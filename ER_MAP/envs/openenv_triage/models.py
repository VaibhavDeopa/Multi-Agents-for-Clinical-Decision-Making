"""
ER_MAP/envs/openenv_triage/models.py
====================================

Pydantic ``Action`` / ``Observation`` / ``State`` models that satisfy the
OpenEnv 0.2.3 interface for the ER-MAP triage environment.

Schema decisions
----------------

Doctor JSON action shape (mirrors ``DOCTOR_TOOLS`` in ``triage_env.py``):

    {"tool": "speak_to",            "target": "nurse|patient", "message": str}
    {"tool": "order_lab",           "test_name": str}
    {"tool": "read_soap",           "section": str}
    {"tool": "update_soap",         "section": str, "content": str}
    {"tool": "terminal_discharge",  "treatment": str, "is_emergency": bool}

The underlying ``TriageEnv.step`` accepts a JSON *string* and parses it
internally. To preserve byte-perfect parity with the gym env we keep the
original raw JSON string on the action (``raw_json``) and replay it
verbatim when stepping. Typed callers (e.g. UI / tests) can construct
the typed dataclass and rely on :meth:`TriageAction.to_json_str` to emit
an equivalent JSON payload.

References
----------
- Pydantic ``Action`` / ``Observation`` / ``State`` base classes:
  https://github.com/meta-pytorch/OpenEnv/blob/main/src/openenv/core/env_server/types.py
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from pydantic import ConfigDict, Field

# OpenEnv base types live in ``openenv.core.env_server.types``. They are
# Pydantic v2 ``BaseModel`` subclasses with sensible defaults
# (``done``/``reward``/``metadata`` on Observation, ``episode_id`` /
# ``step_count`` on State, ``metadata`` on Action).
try:
    from openenv.core.env_server.types import Action, Observation, State
except ImportError:  # pragma: no cover - openenv-core not installed
    # Fallback shims so this module can still be imported in environments
    # that don't have ``openenv-core`` available (e.g. the running Kaggle
    # training image). The wrapper itself will fail at construction time
    # in that case, which is fine.
    from pydantic import BaseModel

    class Action(BaseModel):  # type: ignore[no-redef]
        model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)
        metadata: Dict[str, Any] = Field(default_factory=dict)

    class Observation(BaseModel):  # type: ignore[no-redef]
        model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)
        done: bool = False
        reward: Optional[float] = None
        metadata: Dict[str, Any] = Field(default_factory=dict)

    class State(BaseModel):  # type: ignore[no-redef]
        model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)
        episode_id: Optional[str] = None
        step_count: int = 0


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

# DOCTOR_TOOLS mirrors the same set used by ``triage_env.py``. Kept as a
# module constant so it can be imported by the FastAPI server / client for
# request validation without re-importing the underlying env.
DOCTOR_TOOLS: tuple = (
    "speak_to",
    "order_lab",
    "terminal_discharge",
    "read_soap",
    "update_soap",
)


class TriageAction(Action):
    """
    Doctor's structured action.

    Parity-critical fields (``tool`` is required, the rest are optional and
    only meaningful for specific tools). Extra fields are allowed because
    the LoRA-tuned policy may emit auxiliary keys (``thought``, etc.) that
    the underlying env tolerates.
    """

    # Allow extra LLM-emitted fields (e.g. ``thought``) and accept legacy
    # actions that may include unknown keys; the wrapped env tolerates them.
    model_config = ConfigDict(
        extra="allow",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    tool: str = Field(..., description="Doctor tool name; one of DOCTOR_TOOLS.")
    target: Optional[str] = Field(
        default=None, description="speak_to target ('nurse' or 'patient')."
    )
    message: Optional[str] = Field(
        default=None, description="speak_to message body."
    )
    test_name: Optional[str] = Field(
        default=None, description="order_lab test name."
    )
    treatment: Optional[str] = Field(
        default=None, description="terminal_discharge treatment plan."
    )
    is_emergency: Optional[bool] = Field(
        default=None,
        description="terminal_discharge emergency-classification boolean.",
    )
    section: Optional[str] = Field(
        default=None,
        description="read_soap / update_soap SOAP section (e.g. 'Assessment').",
    )
    content: Optional[str] = Field(
        default=None, description="update_soap content."
    )
    thought: Optional[str] = Field(
        default=None, description="Optional LLM scratchpad text."
    )

    # Round-trip support: when the action originated from a raw JSON
    # string (e.g. an LLM completion) we keep that exact string so the
    # wrapper can replay it verbatim into the gym env, preserving byte
    # parity with the existing reward/parser logic.
    raw_json: Optional[str] = Field(
        default=None,
        description="Original JSON string this action was parsed from (for parity replay).",
    )

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_json_str(cls, s: str) -> "TriageAction":
        """
        Parse a Doctor's free-text JSON action.

        Robust to the same minor malformations the underlying
        ``TriageEnv._parse_doctor_action`` accepts (e.g. JSON embedded in
        prose), but raises ``ValueError`` on unrecoverable malformed input
        so callers can short-circuit before paying the env step cost.
        """
        if not isinstance(s, str):
            raise ValueError("from_json_str expects a string")
        try:
            parsed = json.loads(s.strip())
        except (json.JSONDecodeError, TypeError):
            # Try to extract the first JSON object inside the text, matching
            # ``triage_env._parse_doctor_action`` behaviour.
            import re

            m = re.search(r"\{.*\}", s, re.DOTALL)
            if not m:
                raise ValueError(f"No JSON object found in action: {s!r}")
            try:
                parsed = json.loads(m.group(0))
            except json.JSONDecodeError as e:
                raise ValueError(f"Malformed JSON action: {e}") from e

        if not isinstance(parsed, dict):
            raise ValueError(f"Action JSON must be an object, got {type(parsed).__name__}")
        if "tool" not in parsed:
            raise ValueError("Action JSON must include a 'tool' field")

        # Build the action via Pydantic validation (extra='allow' captures
        # any non-declared LLM keys without dropping them).
        action = cls.model_validate(parsed)
        # Preserve the exact original payload for parity replay.
        action.raw_json = s
        return action

    def to_json_str(self) -> str:
        """
        Serialize back to the JSON string the underlying env consumes.

        If this action was constructed from a raw JSON string, return that
        original payload verbatim (byte-perfect parity). Otherwise emit a
        fresh JSON dump of all non-None declared fields plus any extras.
        """
        if self.raw_json:
            return self.raw_json
        # ``model_dump(exclude_none=True)`` drops unset optional fields so
        # the env sees a clean schema-shaped action rather than a sea of
        # nulls. ``raw_json`` and ``metadata`` are server-side concerns and
        # are excluded from the wire payload.
        payload = self.model_dump(exclude_none=True, exclude={"raw_json", "metadata"})
        return json.dumps(payload)


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class TriageObservation(Observation):
    """
    Doctor-visible observation.

    Mirrors the structured JSON the gym env emits, but exposes the parsed
    fields directly. The original ``raw_observation`` JSON string is also
    retained so policies/clients that were trained against the raw bytes
    can keep using them unmodified.
    """

    model_config = ConfigDict(
        extra="allow",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    raw_observation: str = Field(
        default="",
        description="Original JSON string from TriageEnv (for byte parity).",
    )
    event: str = Field(
        default="",
        description="Event type (e.g. 'episode_start', 'lab_result', 'terminal_win').",
    )
    payload: Dict[str, Any] = Field(
        default_factory=dict,
        description="Parsed JSON content from the gym env.",
    )
    truncated: bool = Field(
        default=False,
        description="Episode truncated due to max_episode_steps (mirrors gym).",
    )
    info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Gym info dict, including reward_components.",
    )


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class TriageState(State):
    """
    Internal episode state surfaced over ``/state`` and the WebSocket
    state message.

    These fields mirror the public attributes of the underlying
    ``TriageEnv`` instance after ``reset()``/``step()``, so frontends and
    debuggers can introspect mid-episode state without poking private
    attributes through HTTP.
    """

    model_config = ConfigDict(
        extra="allow",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    done: bool = Field(default=False)
    truncated: bool = Field(default=False)
    consent_given: bool = Field(default=False)
    phase: int = Field(default=1)
    ordered_labs: List[str] = Field(default_factory=list)
    patient_status: str = Field(default="CONTINUE")
    soap_note: Dict[str, Any] = Field(default_factory=dict)
    reward_components: Dict[str, float] = Field(default_factory=dict)
    ground_truth_disease: Optional[str] = Field(default=None)


__all__ = [
    "TriageAction",
    "TriageObservation",
    "TriageState",
    "DOCTOR_TOOLS",
]
