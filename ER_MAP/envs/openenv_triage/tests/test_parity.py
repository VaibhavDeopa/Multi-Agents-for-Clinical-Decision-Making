"""
ER_MAP/envs/openenv_triage/tests/test_parity.py
================================================

Parity + smoke tests for the OpenEnv wrapper.

Coverage:
1. Same seed + options + action sequence produces byte-identical rewards,
   terminations, and observation payloads in the wrapped env vs the gym
   ``TriageEnv``.
2. ``TriageAction.from_json_str`` parses well-formed Doctor JSON and
   rejects malformed payloads.
3. End-to-end HTTP round-trip via ``TriageOpenEnvClient`` returns the
   same observation/reward/done as direct in-process env usage.
4. ``openenv.yaml`` is valid YAML and references files that exist on disk.

All tests run without Groq keys: the underlying ``AgentRouter`` falls back
to ``_mock_response`` when no Groq client is configured, so we get
deterministic stubbed Nurse/Patient responses.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import pytest

# Ensure no Groq client gets initialized so the mock-response path is
# active (deterministic stubs). Also wipe the per-role keys.
for _k in (
    "GROQ_API_KEY",
    "GROQ_NURSE_API_KEY",
    "GROQ_PATIENT_API_KEY",
    "GROQ_EMPATHY_JUDGE_API_KEY",
    "GROQ_MEDICAL_JUDGE_API_KEY",
):
    os.environ.pop(_k, None)


from ER_MAP.envs.triage_env import TriageEnv
from ER_MAP.envs.openenv_triage import (
    TriageAction,
    TriageObservation,
    TriageOpenEnv,
    TriageState,
)
from ER_MAP.envs.openenv_triage.client import TriageOpenEnvClient


# A short, deterministic action script that exercises several tools
# without depending on the LLM judges' verdict (we just check parity).
ACTION_SCRIPT = [
    '{"tool": "read_soap"}',
    '{"tool": "speak_to", "target": "patient", "message": "Hello, how are you feeling?"}',
    '{"tool": "order_lab", "test_name": "CBC"}',
    '{"tool": "update_soap", "section": "Assessment", "content": "Likely viral infection."}',
    '{"tool": "terminal_discharge", "treatment": "Rest and hydration", "is_emergency": false}',
]

RESET_KWARGS = {"seed": 42, "options": {"phase": 1, "difficulty": "easy"}}


# ---------------------------------------------------------------------------
# Test 1 - in-process parity
# ---------------------------------------------------------------------------

def _run_gym_episode():
    env = TriageEnv()
    obs, info = env.reset(**RESET_KWARGS)
    trace = [("reset", obs, 0.0, False, False, info)]
    for action in ACTION_SCRIPT:
        if trace[-1][3] or trace[-1][4]:
            break
        obs, reward, done, truncated, info = env.step(action)
        trace.append(("step", obs, reward, done, truncated, info))
    env.close()
    return trace


def _run_openenv_episode():
    env = TriageOpenEnv()
    obs = env.reset(**RESET_KWARGS)
    trace = [("reset", obs.raw_observation, 0.0, False, False, obs.info)]
    for action_str in ACTION_SCRIPT:
        if trace[-1][3] or trace[-1][4]:
            break
        action = TriageAction.from_json_str(action_str)
        obs = env.step(action)
        trace.append(
            ("step", obs.raw_observation, obs.reward, obs.done and not obs.truncated,
             obs.truncated, obs.info)
        )
    env.close()
    return trace


def _strip_volatile(info: dict) -> dict:
    """Drop fields that legitimately differ between two clean episodes
    (none currently expected because the mock LLM path is deterministic
    given a fixed seed, but we keep this hook for forward compatibility)."""
    return {k: v for k, v in (info or {}).items() if k != "_timestamp"}


def test_parity_step_by_step():
    """Wrapped env produces identical rewards, dones, and obs payloads."""
    gym_trace = _run_gym_episode()
    oe_trace = _run_openenv_episode()

    assert len(gym_trace) == len(oe_trace), (
        f"Trace lengths differ: gym={len(gym_trace)} openenv={len(oe_trace)}"
    )

    for i, (g, o) in enumerate(zip(gym_trace, oe_trace)):
        g_kind, g_obs, g_reward, g_done, g_trunc, g_info = g
        o_kind, o_obs, o_reward, o_done, o_trunc, o_info = o

        assert g_kind == o_kind, f"step {i} kind differs"
        assert pytest.approx(g_reward, abs=1e-9) == o_reward, (
            f"step {i} reward differs: gym={g_reward} openenv={o_reward}"
        )
        assert g_done == o_done, f"step {i} done differs"
        assert g_trunc == o_trunc, f"step {i} truncated differs"

        # Compare structured observation content (not the literal string,
        # because dict ordering may diverge in JSON encoding even though
        # the parsed payloads are equal).
        try:
            g_payload = json.loads(g_obs)
            o_payload = json.loads(o_obs)
        except (json.JSONDecodeError, TypeError):
            g_payload, o_payload = g_obs, o_obs
        assert g_payload == o_payload, f"step {i} observation payload differs"

        # reward_components must match exactly (these are the GRPO signal).
        assert _strip_volatile(g_info.get("reward_components", {})) == _strip_volatile(
            o_info.get("reward_components", {})
        ), f"step {i} reward_components differ"


# ---------------------------------------------------------------------------
# Test 2 - TriageAction parsing
# ---------------------------------------------------------------------------

def test_action_from_json_str_well_formed():
    a = TriageAction.from_json_str('{"tool": "speak_to", "target": "nurse", "message": "Hi"}')
    assert a.tool == "speak_to"
    assert a.target == "nurse"
    assert a.message == "Hi"
    # Round-trip preserves the original bytes for parity replay.
    assert a.raw_json is not None
    assert json.loads(a.to_json_str()) == json.loads(a.raw_json)


def test_action_from_json_str_embedded():
    """TriageAction tolerates the same embedded-JSON noise the gym env
    accepts (LLMs sometimes emit prose around their JSON)."""
    a = TriageAction.from_json_str(
        "Sure, here is my action: {\"tool\": \"read_soap\"}"
    )
    assert a.tool == "read_soap"


def test_action_from_json_str_rejects_malformed():
    with pytest.raises(ValueError):
        TriageAction.from_json_str("not even close to JSON")
    with pytest.raises(ValueError):
        # Missing 'tool' field is invalid per the env's parser.
        TriageAction.from_json_str('{"foo": "bar"}')
    with pytest.raises(ValueError):
        TriageAction.from_json_str('{"tool":')  # truncated


# ---------------------------------------------------------------------------
# Test 3 - HTTP round-trip via the FastAPI server
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def http_server():
    """Spin up the FastAPI app on a free port using uvicorn in a thread."""
    pytest.importorskip("uvicorn")
    pytest.importorskip("openenv.core.env_server.http_server")
    import uvicorn

    from ER_MAP.envs.openenv_triage.server import build_app

    port = int(os.environ.get("ERMAP_TEST_PORT", "8765"))
    app = build_app()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # Wait until the server is reachable (max 15s).
    import requests

    base = f"http://127.0.0.1:{port}"
    deadline = time.time() + 15.0
    while time.time() < deadline:
        try:
            r = requests.get(f"{base}/health", timeout=1.0)
            if r.status_code == 200:
                break
        except Exception:
            pass
        time.sleep(0.2)
    else:
        server.should_exit = True
        thread.join(timeout=2.0)
        pytest.skip("uvicorn did not come up within 15s")

    yield base

    server.should_exit = True
    thread.join(timeout=5.0)


def test_http_round_trip_matches_inproc(http_server):
    """Same seed/options/actions through HTTP -> identical reward stream."""
    client = TriageOpenEnvClient(base_url=http_server)

    health = client.health()
    assert health.get("status") == "healthy"

    # Run via HTTP.
    result = client.reset(**RESET_KWARGS)
    http_trace = [("reset", result.observation.raw_observation, 0.0, result.done, result.truncated)]
    for action_str in ACTION_SCRIPT:
        if http_trace[-1][3] or http_trace[-1][4]:
            break
        action = TriageAction.from_json_str(action_str)
        result = client.step(action)
        http_trace.append(
            (
                "step",
                result.observation.raw_observation,
                result.reward,
                result.done and not result.truncated,
                result.truncated,
            )
        )

    # Compare against in-process run.
    oe_trace = _run_openenv_episode()

    assert len(http_trace) == len(oe_trace), (
        f"HTTP trace length {len(http_trace)} != in-proc {len(oe_trace)}"
    )
    for i, (h, o) in enumerate(zip(http_trace, oe_trace)):
        h_kind, h_obs, h_reward, h_done, h_trunc = h
        o_kind, o_obs, o_reward, o_done, o_trunc, _ = o
        assert h_kind == o_kind
        assert pytest.approx(h_reward, abs=1e-9) == o_reward, f"step {i} reward differs over HTTP"
        assert h_done == o_done, f"step {i} done differs over HTTP"
        assert h_trunc == o_trunc, f"step {i} truncated differs over HTTP"
        try:
            assert json.loads(h_obs) == json.loads(o_obs), f"step {i} obs differs over HTTP"
        except (json.JSONDecodeError, TypeError):
            assert h_obs == o_obs


# ---------------------------------------------------------------------------
# Test 4 - openenv.yaml manifest sanity
# ---------------------------------------------------------------------------

def test_openenv_yaml_valid():
    yaml = pytest.importorskip("yaml")
    manifest_path = Path(__file__).resolve().parent.parent / "openenv.yaml"
    assert manifest_path.exists(), f"openenv.yaml not found at {manifest_path}"

    data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)

    # OpenEnv 0.2.x manifest minimum-fields (per echo_env example):
    # spec_version, name, type, runtime, app, port.
    for field in ("spec_version", "name", "type", "runtime", "app", "port"):
        assert field in data, f"openenv.yaml missing required field: {field}"

    # The ``app`` field references ``server.app:app`` (relative module
    # path inside the deployed image). Verify the local file exists.
    app_ref = str(data["app"])
    module_path, _, attr = app_ref.partition(":")
    assert attr == "app", f"openenv.yaml app must end in ':app', got {app_ref}"

    expected_server_path = (
        Path(__file__).resolve().parent.parent
        / (module_path.replace(".", os.sep) + ".py")
    )
    # Some manifests use ``server.app:app`` referring to ``server/app.py``,
    # others (this one) use a flat ``server.py``. Accept either layout.
    flat_alternate = (
        Path(__file__).resolve().parent.parent
        / (module_path.split(".")[-1] + ".py")
    )
    assert expected_server_path.exists() or flat_alternate.exists(), (
        f"openenv.yaml app references missing server module: "
        f"tried {expected_server_path} and {flat_alternate}"
    )

    # Dockerfile should exist so ``openenv push`` can build the Space.
    dockerfile = Path(__file__).resolve().parent.parent / "Dockerfile"
    assert dockerfile.exists(), "Dockerfile missing next to openenv.yaml"
