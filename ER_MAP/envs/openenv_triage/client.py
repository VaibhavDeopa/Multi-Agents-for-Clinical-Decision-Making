"""
ER_MAP/envs/openenv_triage/client.py
====================================

Thin HTTP client for the :class:`TriageOpenEnv` FastAPI server.

This client deliberately avoids importing any server-internal modules
(``server.py``, ``env.py``, etc.) so it can be packaged independently
(per the OpenEnv hackathon brief: "clients should never import server
internals").

Wire format matches OpenEnv 0.2.3's HTTP routes registered by
``HTTPEnvServer.register_routes`` -
https://github.com/meta-pytorch/OpenEnv/blob/main/src/openenv/core/env_server/http_server.py:

- ``POST /reset``  body:  {seed?, episode_id?, options?, ...}
                  reply: {observation, reward, done}
- ``POST /step``   body:  {action: <action_fields>, timeout_s?, ...}
                  reply: {observation, reward, done}
- ``GET  /state``  reply: {observation: <state fields>, reward: null, done: false}
- ``GET  /health`` reply: {status: "healthy"}

Returns
-------
``reset`` and ``step`` both return a :class:`StepResult` with the parsed
:class:`TriageObservation`. ``state`` returns a :class:`TriageState`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests

# The client only depends on the public Pydantic models in this package.
# It does NOT import env.py / server.py.
from .models import TriageAction, TriageObservation, TriageState

logger = logging.getLogger("ER_MAP.openenv_triage.client")


@dataclass
class StepResult:
    """
    Parsed envelope returned by ``reset`` and ``step``.

    Mirrors OpenEnv's ``StepResult`` semantics (observation + reward +
    done) without forcing the caller to depend on ``openenv-core`` at
    runtime. Aliases of the underlying gym 5-tuple are also available
    via :attr:`observation`'s ``truncated`` / ``info`` fields.
    """

    observation: TriageObservation
    reward: Optional[float]
    done: bool

    @property
    def truncated(self) -> bool:
        return bool(self.observation.truncated)

    @property
    def info(self) -> Dict[str, Any]:
        return dict(self.observation.info)


class TriageOpenEnvClient:
    """
    Synchronous HTTP client for ER-MAP's OpenEnv-compliant Triage server.

    Example
    -------
    >>> client = TriageOpenEnvClient(base_url="http://localhost:8000")
    >>> client.health()
    {'status': 'healthy', ...}
    >>> result = client.reset(seed=0, options={"phase": 1, "difficulty": "easy"})
    >>> action = TriageAction.from_json_str('{"tool": "read_soap"}')
    >>> result = client.step(action)
    >>> print(result.observation.event, result.reward)
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        *,
        timeout_s: float = 60.0,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self._session = session or requests.Session()

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def health(self) -> Dict[str, Any]:
        resp = self._session.get(self._url("/health"), timeout=self.timeout_s)
        resp.raise_for_status()
        return resp.json()

    def healthz(self) -> Dict[str, Any]:
        """Richer health endpoint exposed by our FastAPI app."""
        resp = self._session.get(self._url("/healthz"), timeout=self.timeout_s)
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # OpenEnv core operations
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        **extra: Any,
    ) -> StepResult:
        body: Dict[str, Any] = {}
        if seed is not None:
            body["seed"] = seed
        if episode_id is not None:
            body["episode_id"] = episode_id
        if options is not None:
            body["options"] = options
        body.update(extra)

        resp = self._session.post(
            self._url("/reset"), json=body, timeout=self.timeout_s
        )
        resp.raise_for_status()
        return self._parse_step_response(resp.json())

    def step(self, action: TriageAction, *, timeout_s: Optional[float] = None) -> StepResult:
        # The server expects the action under "action" (per StepRequest).
        # Use ``model_dump`` so Pydantic-typed fields are JSON-serialized
        # exactly as the server's ``deserialize_action`` expects.
        action_payload = action.model_dump(exclude_none=True, exclude={"metadata"})
        body: Dict[str, Any] = {"action": action_payload}
        if timeout_s is not None:
            body["timeout_s"] = timeout_s

        resp = self._session.post(
            self._url("/step"), json=body, timeout=self.timeout_s
        )
        resp.raise_for_status()
        return self._parse_step_response(resp.json())

    def state(self) -> TriageState:
        resp = self._session.get(self._url("/state"), timeout=self.timeout_s)
        resp.raise_for_status()
        data = resp.json()
        # ``HTTPEnvServer`` wraps the state as
        # ``{"observation": {...state fields...}, "reward": None, "done": ...}``
        # via ``serialize_observation``-equivalent path. Some versions
        # return the state directly; tolerate both.
        if isinstance(data, dict) and "observation" in data and "reward" in data:
            state_dict = data["observation"]
        else:
            state_dict = data
        return TriageState.model_validate(state_dict)

    # ------------------------------------------------------------------
    # Context manager sugar
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> "TriageOpenEnvClient":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_step_response(payload: Dict[str, Any]) -> StepResult:
        obs_dict = payload.get("observation") or {}
        obs = TriageObservation.model_validate(obs_dict)
        reward = payload.get("reward")
        # OpenEnv's serialize_observation strips ``done``/``reward`` from
        # the observation dict, so ``done`` lives at the envelope level.
        done = bool(payload.get("done", False))
        # Make sure obs.done agrees with the envelope.
        if obs.done != done:
            obs.done = done
        if reward is not None and obs.reward is None:
            obs.reward = float(reward)
        return StepResult(observation=obs, reward=reward, done=done)


__all__ = ["TriageOpenEnvClient", "StepResult"]
