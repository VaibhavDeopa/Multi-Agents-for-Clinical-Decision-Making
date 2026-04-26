# ER-MAP top-level Makefile
#
# Convenience targets for the OpenEnv wrapper. Kept minimal so it does
# not interfere with the existing Kaggle training workflow.

PY ?= python

.PHONY: test-openenv serve-openenv clean

## Run the OpenEnv parity tests (gym vs wrapper, in-proc vs HTTP).
test-openenv:
	$(PY) -m pytest ER_MAP/envs/openenv_triage/tests -v

## Launch the OpenEnv FastAPI server on port 8000 for local inspection.
serve-openenv:
	$(PY) -m uvicorn ER_MAP.envs.openenv_triage.server:app --host 0.0.0.0 --port 8000

clean:
	rm -rf .pytest_cache **/__pycache__
