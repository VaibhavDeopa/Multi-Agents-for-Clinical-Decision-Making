# ER_MAP/envs/__init__.py
# Package initializer for the ER-MAP environment suite.

from .triage_env import TriageEnv
from .randomizer import generate_ground_truth, construct_prompts
from .api_router import AgentRouter

# OpenEnv-compliant wrapper. Imported via try/except so users without
# the optional ``openenv-core`` dependency (e.g. the running Kaggle
# training job) keep working unchanged.
try:  # pragma: no cover - import-time soft fail
    from .openenv_triage import TriageOpenEnv  # noqa: F401
except ImportError:
    TriageOpenEnv = None  # type: ignore[assignment]

__all__ = [
    "TriageEnv",
    "TriageOpenEnv",
    "generate_ground_truth",
    "construct_prompts",
    "AgentRouter",
]
