"""
ER_MAP/envs/openenv_triage/env.py
=================================

OpenEnv ``Environment`` subclass that delegates to the existing in-process
Gymnasium ``TriageEnv`` so a LoRA trained against the gym env can run
unchanged behind the OpenEnv HTTP/WebSocket protocol.

Citations (verified against the canonical OpenEnv repo & PyPI as of
April 2026):

- ``Environment`` interface (abstract ``reset`` / ``step`` / ``state``):
  https://github.com/meta-pytorch/OpenEnv/blob/main/src/openenv/core/env_server/interfaces.py
  -> ``class Environment(ABC, Generic[ActT, ObsT, StateT]): ...``
  -> ``reset(seed, episode_id, **kwargs) -> ObsT``
  -> ``step(action, timeout_s=None, **kwargs) -> ObsT``  (single-return; reward/done live on the Observation)
  -> ``@property state`` returning the State subclass
  -> ``close()`` is overridable (default no-op).
- ``Action`` / ``Observation`` / ``State`` Pydantic bases:
  https://github.com/meta-pytorch/OpenEnv/blob/main/src/openenv/core/env_server/types.py
- ``serialize_observation`` strips ``reward``/``done``/``metadata`` from
  the observation dict and emits ``{"observation", "reward", "done"}``:
  https://github.com/meta-pytorch/OpenEnv/blob/main/src/openenv/core/env_server/serialization.py
- ``RESERVED_TOOL_NAMES`` (``reset`` / ``step`` / ``state`` / ``close``)
  must NOT be used as MCP tool names:
  https://github.com/meta-pytorch/OpenEnv/blob/main/src/openenv/core/env_server/mcp_types.py
- echo_env exemplar (showing ``MCPEnvironment`` / ``Environment`` class layout):
  https://github.com/meta-pytorch/OpenEnv/tree/main/envs/echo_env
- pyproject pinning ``openenv-core==0.2.3`` (latest as of April 2026):
  https://github.com/meta-pytorch/OpenEnv/blob/main/pyproject.toml

We do NOT subclass ``MCPEnvironment`` because the wrapped env is not a
tool-server: the Doctor agent emits a single structured ``TriageAction``
per step (closer to BlackJack/Wordle semantics than to coding_env's
tools/list+tools/call shape). ``Environment`` is the right base.

Key parity rule
---------------
The gym ``TriageEnv.step(action: str)`` returns the canonical 5-tuple
``(obs_str, reward, done, truncated, info)``. The OpenEnv ``Observation``
has ``done`` and ``reward`` first-class but NO truncated flag, so we fold
``truncated`` and ``info`` (including ``reward_components``) into the
observation's ``metadata`` and ``info`` fields so callers can recover the
exact gym tuple. Same trajectory + same seed -> identical rewards.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Any, Dict, Optional

# OpenEnv 0.2.3 ships the abstract ``Environment`` base under
# ``openenv.core.env_server.interfaces``. See citation block above.
from openenv.core.env_server.interfaces import Environment

from ER_MAP.envs.triage_env import TriageEnv

from .models import TriageAction, TriageObservation, TriageState

logger = logging.getLogger("ER_MAP.openenv_triage.env")


def _stub_env_kwargs() -> Dict[str, Any]:
    """
    Build env kwargs that allow the env to operate without Groq keys.

    The underlying ``AgentRouter`` already has a degraded fallback path
    (``_mock_response``) when no Groq client is configured, so handing it
    empty keys is sufficient. The stub mode is documented in the README.
    """
    return {
        "groq_api_key": os.environ.get("GROQ_API_KEY", "") or None,
        "nurse_api_key": os.environ.get("GROQ_NURSE_API_KEY", "") or None,
        "patient_api_key": os.environ.get("GROQ_PATIENT_API_KEY", "") or None,
        "empathy_judge_api_key": os.environ.get("GROQ_EMPATHY_JUDGE_API_KEY", "") or None,
        "medical_judge_api_key": os.environ.get("GROQ_MEDICAL_JUDGE_API_KEY", "") or None,
        "model": os.environ.get("ERMAP_MODEL", "llama-3.3-70b-versatile"),
    }


def _parse_obs_json(obs_str: str) -> Dict[str, Any]:
    """Best-effort parse of the gym env's JSON observation string."""
    try:
        return json.loads(obs_str)
    except (json.JSONDecodeError, TypeError):
        return {"event": "raw", "raw": obs_str}


class TriageOpenEnv(Environment[TriageAction, TriageObservation, TriageState]):
    """
    OpenEnv-compliant wrapper around ``ER_MAP.envs.triage_env.TriageEnv``.

    Internally instantiates one ``TriageEnv`` per session. Reset/step
    delegate to the gym env unchanged so reward semantics, termination
    rules, and trajectory contents stay byte-identical to the gym
    baseline used by ``training/train_grpo.py``.
    """

    # ``HTTPEnvServer`` instantiates a fresh environment per WebSocket
    # session (see ``http_server.py``). The underlying ``TriageEnv`` keeps
    # its state in instance attributes so concurrent sessions are safe.
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, env_kwargs: Optional[Dict[str, Any]] = None):
        """
        Args:
            env_kwargs: Optional kwargs forwarded to the underlying
                ``TriageEnv`` constructor. Defaults pull Groq keys from
                env vars; missing keys silently fall back to mock mode.
        """
        super().__init__()

        merged_kwargs = _stub_env_kwargs()
        if env_kwargs:
            merged_kwargs.update(env_kwargs)
        self._env_kwargs = merged_kwargs

        # Build the gym env immediately so import-time failures (missing
        # disease DB, etc.) surface synchronously rather than at first
        # /reset request.
        self._env = TriageEnv(**self._env_kwargs)

        self._state = TriageState(
            episode_id=str(uuid.uuid4()),
            step_count=0,
        )
        self._last_truncated: bool = False
        self._last_info: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> TriageObservation:
        """
        Start a new episode.

        ``options`` mirrors the gym ``options`` dict and supports the same
        keys as ``TriageEnv.reset``: ``{"phase": 1|2|3, "difficulty":
        "easy|medium|hard"}``.
        """
        # Honour anything the OpenEnv server passes through that maps to
        # the gym env's options dict (e.g. ``phase`` / ``difficulty``
        # supplied as top-level kwargs by upstream tooling).
        options = dict(options or {})
        for opt_key in ("phase", "difficulty"):
            if opt_key in kwargs and opt_key not in options:
                options[opt_key] = kwargs.pop(opt_key)

        obs_str, info = self._env.reset(seed=seed, options=options or None)

        self._last_truncated = False
        self._last_info = dict(info or {})

        # Refresh state snapshot.
        self._state = TriageState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=int(getattr(self._env, "step_count", 0) or 0),
            done=bool(getattr(self._env, "done", False)),
            truncated=False,
            consent_given=bool(getattr(self._env, "consent_given", False)),
            phase=int(getattr(self._env, "phase", options.get("phase", 1)) or 1),
            ordered_labs=list(getattr(self._env, "ordered_labs", set()) or set()),
            patient_status=str(getattr(self._env, "last_patient_status", "CONTINUE")),
            soap_note=dict(getattr(self._env, "emr", {}) or {}),
            reward_components=dict(getattr(self._env, "reward_components", {}) or {}),
            ground_truth_disease=info.get("ground_truth_disease"),
        )

        return self._build_observation(
            obs_str=obs_str,
            reward=0.0,
            done=False,
            truncated=False,
            info=self._last_info,
        )

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(
        self,
        action: TriageAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> TriageObservation:
        """
        Execute one Doctor turn.

        Converts the typed ``TriageAction`` back into the JSON string the
        gym env expects (preserving the original raw bytes whenever the
        action was created via :meth:`TriageAction.from_json_str`), runs
        the gym ``step``, and re-packages the 5-tuple into a single
        ``TriageObservation``.
        """
        action_str = action.to_json_str()
        obs_str, reward, done, truncated, info = self._env.step(action_str)

        self._last_truncated = bool(truncated)
        self._last_info = dict(info or {})

        # Mirror gym attributes onto the typed state for /state polling.
        self._state = TriageState(
            episode_id=self._state.episode_id,
            step_count=int(getattr(self._env, "step_count", 0) or 0),
            done=bool(done),
            truncated=bool(truncated),
            consent_given=bool(getattr(self._env, "consent_given", False)),
            phase=int(getattr(self._env, "phase", 1) or 1),
            ordered_labs=list(getattr(self._env, "ordered_labs", set()) or set()),
            patient_status=str(getattr(self._env, "last_patient_status", "CONTINUE")),
            soap_note=dict(getattr(self._env, "emr", {}) or {}),
            reward_components=dict(
                info.get("reward_components")
                or getattr(self._env, "reward_components", {})
                or {}
            ),
            ground_truth_disease=self._state.ground_truth_disease,
        )

        return self._build_observation(
            obs_str=obs_str,
            reward=float(reward),
            done=bool(done),
            truncated=bool(truncated),
            info=self._last_info,
        )

    # ------------------------------------------------------------------
    # State / close
    # ------------------------------------------------------------------

    @property
    def state(self) -> TriageState:
        return self._state

    def close(self) -> None:
        """Release the underlying env's router resources."""
        try:
            self._env.close()
        except Exception:  # pragma: no cover - best-effort cleanup
            logger.debug("TriageEnv.close() raised; ignoring.", exc_info=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_observation(
        self,
        obs_str: str,
        reward: float,
        done: bool,
        truncated: bool,
        info: Dict[str, Any],
    ) -> TriageObservation:
        payload = _parse_obs_json(obs_str)
        event = str(payload.get("event", "")) if isinstance(payload, dict) else ""

        # Promote ``truncated`` and the gym ``info`` dict (which carries
        # ``reward_components``) into the observation so OpenEnv clients
        # can reconstruct the exact gym 5-tuple if they need to.
        metadata: Dict[str, Any] = {
            "truncated": bool(truncated),
            "reward_components": dict(info.get("reward_components", {})),
            "step_count": info.get("step_count"),
            "patient_status": info.get("patient_status"),
            "consent_given": info.get("consent_given"),
            "truncation_reason": info.get("truncation_reason"),
        }
        # Drop None values to keep serialized payloads tidy.
        metadata = {k: v for k, v in metadata.items() if v is not None}

        return TriageObservation(
            done=bool(done) or bool(truncated),
            reward=float(reward),
            metadata=metadata,
            raw_observation=obs_str if isinstance(obs_str, str) else json.dumps(obs_str),
            event=event,
            payload=payload if isinstance(payload, dict) else {},
            truncated=bool(truncated),
            info=dict(info or {}),
        )


__all__ = ["TriageOpenEnv"]
