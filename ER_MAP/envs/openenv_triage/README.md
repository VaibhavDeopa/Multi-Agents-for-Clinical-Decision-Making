# ER-MAP Triage - OpenEnv Wrapper

This package wraps the existing in-house Gymnasium `TriageEnv`
(`ER_MAP/envs/triage_env.py`) so it can be served through the
[OpenEnv](https://github.com/meta-pytorch/OpenEnv) protocol
(`openenv-core>=0.2.3`, latest as of April 2026).

The wrapper does **not** modify the underlying environment - the same
LoRA currently being trained against the gym env on Kaggle runs unchanged
behind this OpenEnv server because both share the exact same `TriageEnv`
instance, reward semantics, and observation format.

## What's in here

| File                       | Purpose                                                                   |
|----------------------------|---------------------------------------------------------------------------|
| `models.py`                | `TriageAction`, `TriageObservation`, `TriageState` (Pydantic v2)          |
| `env.py`                   | `TriageOpenEnv` - `Environment` subclass delegating to `TriageEnv`        |
| `server.py`                | FastAPI app exposed via OpenEnv's `create_app` helper                     |
| `client.py`                | Thin `requests`-based `TriageOpenEnvClient`                               |
| `openenv.yaml`             | OpenEnv manifest (HF Space deployment target)                             |
| `Dockerfile`               | HF Space / Docker image (Python 3.11-slim)                                |
| `server-requirements.txt`  | Container-side pip requirements                                           |
| `tests/test_parity.py`     | pytest parity tests (gym vs wrapper, in-proc vs HTTP)                     |

## Usage

### 1. Local Python (direct, no HTTP)

```python
from ER_MAP.envs.openenv_triage import TriageOpenEnv, TriageAction

env = TriageOpenEnv()
obs = env.reset(seed=0, options={"phase": 1, "difficulty": "easy"})
print(obs.event, obs.payload)

action = TriageAction.from_json_str(
    '{"tool": "speak_to", "target": "patient", "message": "Hello"}'
)
obs = env.step(action)
print(obs.reward, obs.done, obs.info["reward_components"])
```

### 2. Local HTTP (FastAPI + client)

Start the server:

```bash
uvicorn ER_MAP.envs.openenv_triage.server:app --host 0.0.0.0 --port 8000
```

Drive it with the bundled client:

```python
from ER_MAP.envs.openenv_triage import TriageOpenEnvClient, TriageAction

with TriageOpenEnvClient(base_url="http://localhost:8000") as client:
    print(client.health())
    result = client.reset(seed=0, options={"phase": 1, "difficulty": "easy"})
    action = TriageAction.from_json_str('{"tool": "read_soap"}')
    result = client.step(action)
    print(result.reward, result.observation.event)
```

The HTTP wire format matches OpenEnv 0.2.3:

- `POST /reset` body  `{"seed": int?, "episode_id": str?, "options": {...}?}`
- `POST /step`  body  `{"action": {"tool": "...", ...}, "timeout_s": float?}`
- `GET  /state` -> serialized `TriageState`
- `GET  /health` -> `{"status": "healthy"}`
- `GET  /healthz` -> richer status (version + stub-mode flag)

OpenEnv-style WebSocket clients can also connect to `WS /ws` (provided by
`HTTPEnvServer.register_routes`).

### 3. Hugging Face Space deployment

The repo follows the OpenEnv manifest convention; a single command pushes
to a Space:

```bash
# from the repo root
openenv push --repo-id <hf-username>/er-map-triage
```

(See `openenv push --help` from `openenv-core>=0.2.3` for options like
`--private` and `--tag`.)

To deploy manually with Docker:

```bash
docker build -t er-map-triage:latest -f ER_MAP/envs/openenv_triage/Dockerfile .
docker run --rm -p 8000:8000 \
    -e GROQ_API_KEY=$GROQ_API_KEY \
    er-map-triage:latest
```

## Stub mode (keyless deployment)

The underlying `AgentRouter` already gracefully handles the no-Groq case:
when no key is configured, it returns deterministic canned Nurse/Patient
JSON responses (see `AgentRouter._mock_response`) and the LLM judges
return neutral default scores. This is what we call "stub mode".

- **Stub mode** is automatic: leave all `GROQ_*` vars unset.
- **Live mode**: set `GROQ_API_KEY` (and optionally per-role keys
  `GROQ_NURSE_API_KEY`, `GROQ_PATIENT_API_KEY`,
  `GROQ_EMPATHY_JUDGE_API_KEY`, `GROQ_MEDICAL_JUDGE_API_KEY`).

The `/healthz` endpoint reports `{"stub_mode": true|false}` so HF Space
visitors can tell which configuration the running container is in.

## Parity guarantees

`TriageOpenEnv.step` produces **byte-identical** observations and rewards
to the gym `TriageEnv.step` for the same seed and action sequence, because
the wrapper:

1. Replays the original Doctor JSON string verbatim into the gym env
   (via `TriageAction.raw_json`), so `_parse_doctor_action` sees exactly
   the bytes the policy emitted.
2. Forwards the gym 5-tuple `(obs, reward, done, truncated, info)`
   unchanged - `done` and `reward` go on the OpenEnv `Observation`,
   `truncated` and the full `info` dict (including `reward_components`)
   are folded into `Observation.metadata` / `Observation.info`.

This is verified by `tests/test_parity.py`.

## Reserved tool names (OpenEnv brief)

Per the OpenEnv brief, MCP tool names `reset`, `step`, `state`, `close`
are reserved. This wrapper deliberately **does not** subclass
`MCPEnvironment` and does **not** register any MCP tools, so no name
collision is possible. The Doctor's tool names (`speak_to`, `order_lab`,
`read_soap`, `update_soap`, `terminal_discharge`) live inside the
`TriageAction.tool` field, not in any MCP namespace.

## Citations

- `openenv-core==0.2.3` PyPI release - <https://pypi.org/project/openenv-core/>
- `Environment` base class - <https://github.com/meta-pytorch/OpenEnv/blob/main/src/openenv/core/env_server/interfaces.py>
- `Action`/`Observation`/`State` types - <https://github.com/meta-pytorch/OpenEnv/blob/main/src/openenv/core/env_server/types.py>
- `create_app` helper - <https://github.com/meta-pytorch/OpenEnv/blob/main/src/openenv/core/env_server/http_server.py>
- echo_env exemplar (`openenv.yaml`, `Dockerfile`, server layout) - <https://github.com/meta-pytorch/OpenEnv/tree/main/envs/echo_env>
- `RESERVED_TOOL_NAMES` - <https://github.com/meta-pytorch/OpenEnv/blob/main/src/openenv/core/env_server/mcp_types.py>
