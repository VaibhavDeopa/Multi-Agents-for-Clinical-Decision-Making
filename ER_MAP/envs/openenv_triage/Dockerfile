# Dockerfile for ER-MAP TriageOpenEnv
#
# Pattern follows the recommended HF Spaces / OpenEnv standalone build:
# - slim Python base
# - install openenv-core + project deps from requirements.txt
# - copy the ER_MAP package (so ``ER_MAP.envs.openenv_triage.server:app``
#   resolves at runtime)
# - expose 8000 and run uvicorn
#
# Reference Dockerfile (echo_env, multi-stage uv build):
#   https://github.com/meta-pytorch/OpenEnv/blob/main/envs/echo_env/server/Dockerfile
#
# We use a single-stage slim image here because the project's
# requirements.txt is small and the multi-stage uv flow adds complexity
# that isn't justified for this environment. The HF Space build keeps
# layer caching by separating the requirements install step.

FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000 \
    HOST=0.0.0.0

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Install Python deps (cached when requirements.txt is unchanged) ---
COPY ER_MAP/envs/openenv_triage/server-requirements.txt /tmp/server-requirements.txt
RUN pip install --upgrade pip \
    && pip install -r /tmp/server-requirements.txt

# --- Copy the ER_MAP package ---
# We keep the full ER_MAP package importable so the wrapper can reach
# ``ER_MAP.envs.triage_env`` and supporting modules.
COPY ER_MAP /app/ER_MAP

ENV PYTHONPATH=/app

EXPOSE 8000

# Health check uses the OpenEnv-provided /health route.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "ER_MAP.envs.openenv_triage.server:app", "--host", "0.0.0.0", "--port", "8000"]
