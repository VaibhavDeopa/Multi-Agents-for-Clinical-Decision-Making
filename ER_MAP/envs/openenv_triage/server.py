"""
ER_MAP/envs/openenv_triage/server.py
====================================

FastAPI server exposing :class:`TriageOpenEnv` over the OpenEnv HTTP /
WebSocket protocol.

Endpoints:

- ``POST /reset``       reset request (matches ``ResetRequest`` schema)
- ``POST /step``        step request  (matches ``StepRequest`` schema)
- ``GET  /state``       serialized ``TriageState``
- ``GET  /health``      ``{"status": "healthy"}``  (also ``/healthz`` for richer info)
- ``GET  /docs``        OpenAPI / Swagger UI
- ``WS   /ws``          persistent OpenEnv session (mounted from openenv-core)
- ``GET  /web``         optional Gradio HumanAgent UI when ``ENABLE_WEB_INTERFACE=true``

Why a custom HTTP layer (instead of relying entirely on
``openenv.core.env_server.http_server.create_app``):

OpenEnv 0.2.3's HTTP ``/reset`` and ``/step`` handlers spin up a NEW
``Environment`` instance per request and call ``close()`` immediately
after - i.e. HTTP is intentionally stateless and persistent sessions
live on the WebSocket ``/ws`` route. (See
``openenv/core/env_server/http_server.py``: ``reset_handler`` /
``step_handler`` both call ``_env = self._env_factory()`` then
``finally: _env.close()``.) For an episodic env like ER-MAP we need
``/step`` to see the state established by the prior ``/reset``, so we
mount stateful HTTP routes here that share a per-process singleton
``TriageOpenEnv``. We additionally compose in the upstream OpenEnv
FastAPI app so the WebSocket / schema / web-interface routes still
work for OpenEnv-native clients.

References
----------
- ``HTTPEnvServer`` HTTP route bodies (note the ``_env_factory()`` per call):
  https://github.com/meta-pytorch/OpenEnv/blob/main/src/openenv/core/env_server/http_server.py
- echo_env's server layout (``server/app.py``):
  https://github.com/meta-pytorch/OpenEnv/blob/main/envs/echo_env/server/app.py

Stub mode
---------
If no Groq API key is present in the environment, the underlying
``AgentRouter`` already falls back to canned mock responses, so a fresh
HF Space can boot and serve full episodes (with non-LLM nurse/patient
behaviour) without any secrets configured. The ``/healthz`` endpoint
reports ``stub_mode: true`` in that case.
"""

from __future__ import annotations

import logging
import os
import threading
import uuid
from typing import Any, Dict, Optional

from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from .env import TriageOpenEnv
from .models import TriageAction, TriageObservation, TriageState

logging.basicConfig(
    level=os.environ.get("ERMAP_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ER_MAP.openenv_triage.server")


__version__ = "0.1.0"


# ---------------------------------------------------------------------------
# Stub-mode detection
# ---------------------------------------------------------------------------

def _has_any_groq_key() -> bool:
    return any(
        os.environ.get(name)
        for name in (
            "GROQ_API_KEY",
            "GROQ_NURSE_API_KEY",
            "GROQ_PATIENT_API_KEY",
            "GROQ_EMPATHY_JUDGE_API_KEY",
            "GROQ_MEDICAL_JUDGE_API_KEY",
        )
    )


_STUB_MODE = not _has_any_groq_key()
if _STUB_MODE:
    logger.warning(
        "TriageOpenEnv server starting in STUB MODE - no Groq keys found. "
        "Nurse/Patient/Judge LLMs will use canned mock responses. "
        "Set GROQ_API_KEY (or per-role keys) to enable live LLM actors."
    )


# ---------------------------------------------------------------------------
# Pydantic request / response envelopes
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    """HTTP body for ``POST /reset``. Matches OpenEnv's ResetRequest plus
    the ER-MAP-specific ``options`` field."""

    model_config = ConfigDict(extra="allow")

    seed: Optional[int] = Field(default=None, ge=0)
    episode_id: Optional[str] = Field(default=None, max_length=255)
    options: Optional[Dict[str, Any]] = Field(default=None)


class StepRequest(BaseModel):
    """HTTP body for ``POST /step``. The ``action`` field accepts either a
    full ``TriageAction`` payload (preferred) or the legacy raw JSON
    string the gym env consumes."""

    model_config = ConfigDict(extra="allow")

    action: Dict[str, Any] = Field(...)
    timeout_s: Optional[float] = Field(default=None, gt=0)


class StepEnvelope(BaseModel):
    """Wire-format response. Matches OpenEnv 0.2.3's ``StepResponse``
    layout: observation dict + reward + done at the envelope level.
    See ``serialize_observation`` in openenv-core."""

    model_config = ConfigDict(extra="forbid")

    observation: Dict[str, Any]
    reward: Optional[float] = None
    done: bool = False


# ---------------------------------------------------------------------------
# Singleton env management (HTTP routes)
# ---------------------------------------------------------------------------

class _SessionHolder:
    """Thread-safe holder for the single per-process TriageOpenEnv used by
    HTTP routes. WebSocket sessions get their own per-connection env via
    the upstream OpenEnv server."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._env: Optional[TriageOpenEnv] = None
        self._episode_id: Optional[str] = None

    def get(self) -> TriageOpenEnv:
        with self._lock:
            if self._env is None:
                self._env = TriageOpenEnv()
                self._episode_id = str(uuid.uuid4())
            return self._env

    def reset(self) -> TriageOpenEnv:
        with self._lock:
            if self._env is not None:
                try:
                    self._env.close()
                except Exception:  # pragma: no cover
                    logger.debug("Old env close raised", exc_info=True)
            self._env = TriageOpenEnv()
            self._episode_id = str(uuid.uuid4())
            return self._env


_session = _SessionHolder()


def _serialize(obs: TriageObservation) -> StepEnvelope:
    """Mirror ``openenv.core.env_server.serialization.serialize_observation``:
    strip ``done``/``reward``/``metadata`` from the observation dict and
    surface them at the envelope level."""
    obs_dict = obs.model_dump(exclude={"reward", "done", "metadata"})
    return StepEnvelope(
        observation=obs_dict,
        reward=float(obs.reward) if obs.reward is not None else None,
        done=bool(obs.done),
    )


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def build_app() -> FastAPI:
    """Construct the FastAPI app. Exposed for tests / programmatic use."""

    app = FastAPI(
        title="ER-MAP TriageOpenEnv",
        version=__version__,
        description=(
            "OpenEnv-compliant HTTP/WebSocket interface for ER-MAP's "
            "multi-agent emergency-room triage environment. Wraps the "
            "in-house Gymnasium TriageEnv without modifying it, so the "
            "same LoRA trained on the gym env runs unchanged here."
        ),
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    @app.get("/health", tags=["Health"])
    def health() -> Dict[str, Any]:
        """Standard OpenEnv health probe."""
        return {"status": "healthy"}

    @app.get("/healthz", tags=["Health"])
    def healthz() -> Dict[str, Any]:
        """Richer status: version + stub-mode flag (HF Space friendly)."""
        return {
            "status": "healthy",
            "version": __version__,
            "stub_mode": _STUB_MODE,
            "env": "TriageOpenEnv",
        }

    # ------------------------------------------------------------------
    # Reset / Step / State (stateful, per-process singleton)
    # ------------------------------------------------------------------

    @app.post("/reset", response_model=StepEnvelope, tags=["Environment Control"])
    def reset(request: ResetRequest = Body(default_factory=ResetRequest)):
        """Reset the underlying TriageEnv and return the initial obs."""
        env = _session.reset()
        kwargs = request.model_dump(exclude_unset=True)
        try:
            obs = env.reset(**kwargs)
        except Exception as e:
            logger.exception("reset failed")
            raise HTTPException(status_code=500, detail=f"reset failed: {e}")
        return _serialize(obs)

    @app.post("/step", response_model=StepEnvelope, tags=["Environment Control"])
    def step(request: StepRequest):
        """Execute a Doctor action against the active env session."""
        env = _session.get()

        try:
            action = TriageAction.model_validate(request.action)
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"invalid action: {e}")

        try:
            obs = env.step(action, timeout_s=request.timeout_s)
        except Exception as e:
            logger.exception("step failed")
            raise HTTPException(status_code=500, detail=f"step failed: {e}")
        return _serialize(obs)

    @app.get("/state", response_model=StepEnvelope, tags=["State Management"])
    def state():
        """Inspect the current episode state."""
        env = _session.get()
        st: TriageState = env.state
        return StepEnvelope(observation=st.model_dump(), reward=None, done=bool(st.done))

    # ------------------------------------------------------------------
    # Best-effort: mount upstream OpenEnv routes for WebSocket / schema /
    # web-UI parity. We avoid colliding with our own /reset, /step,
    # /state, /health by mounting the upstream app under /openenv.
    # ------------------------------------------------------------------

    try:
        from openenv.core.env_server.http_server import create_app as _oe_create_app

        oe_app = _oe_create_app(
            TriageOpenEnv,
            TriageAction,
            TriageObservation,
            env_name="er_map_triage",
            max_concurrent_envs=int(os.environ.get("MAX_CONCURRENT_ENVS", "8")),
        )
        # Mount under /openenv so OpenEnv-native clients can use
        # ws://host/openenv/ws and the schema routes without colliding
        # with the stateful HTTP routes above.
        app.mount("/openenv", oe_app)
    except Exception as e:  # pragma: no cover
        logger.warning("Failed to mount upstream OpenEnv app at /openenv: %s", e)

    return app


# Module-level ``app`` for ``uvicorn ER_MAP.envs.openenv_triage.server:app``.
app = build_app()


def main() -> None:
    """Entry point for ``python -m ER_MAP.envs.openenv_triage.server``."""
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":  # pragma: no cover
    main()
