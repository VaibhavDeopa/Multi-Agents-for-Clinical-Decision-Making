"""
ER_MAP/envs/openenv_triage
==========================

OpenEnv-compliant wrapper around the existing Gymnasium ``TriageEnv``.

This package adapts the in-process gym env to the OpenEnv interface
(``openenv-core>=0.2.3``) without modifying the underlying environment,
so the same LoRA trained against ``ER_MAP.envs.triage_env.TriageEnv``
runs unchanged against ``TriageOpenEnv``.

Public surface
--------------
- :class:`TriageOpenEnv`        - Server-side OpenEnv ``Environment`` subclass.
- :class:`TriageAction`         - Pydantic ``Action`` mirroring the Doctor's JSON schema.
- :class:`TriageObservation`    - Pydantic ``Observation`` exposing the structured Doctor view.
- :class:`TriageState`          - Pydantic ``State`` exposing the internal episode state.
- :class:`TriageOpenEnvClient`  - Thin HTTP client (no server-internal imports).

References
----------
- OpenEnv canonical repo:        https://github.com/meta-pytorch/OpenEnv
- ``Environment`` interface:     ``openenv.core.env_server.interfaces.Environment``
- ``Action`` / ``Observation``:  ``openenv.core.env_server.types``
- echo_env exemplar:             https://github.com/meta-pytorch/OpenEnv/tree/main/envs/echo_env
"""

from __future__ import annotations

from .models import TriageAction, TriageObservation, TriageState

# ``env`` and ``client`` import openenv-core (a heavy dep). Guard those
# behind try/except so plain users of the gym TriageEnv (e.g. the running
# Kaggle training session) are unaffected if openenv-core is not installed.
try:
    from .env import TriageOpenEnv  # noqa: F401
except Exception:  # pragma: no cover - import-time soft fail
    TriageOpenEnv = None  # type: ignore[assignment]

try:
    from .client import TriageOpenEnvClient  # noqa: F401
except Exception:  # pragma: no cover
    TriageOpenEnvClient = None  # type: ignore[assignment]


__all__ = [
    "TriageAction",
    "TriageObservation",
    "TriageState",
    "TriageOpenEnv",
    "TriageOpenEnvClient",
]
