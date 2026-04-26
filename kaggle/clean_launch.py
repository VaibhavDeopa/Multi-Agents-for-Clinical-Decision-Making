"""
ER-MAP CLEAN TRAINING LAUNCH (Kaggle, T4-safe, 75-episode fixed budget)

Self-contained, idempotent, foolproof. Replaces the old Cell 9 / Cell 11 /
Cell 13 sequence with ONE cell that:

  1. Force-pulls the repo to origin/main (picks up any new fix commits)
  2. Drops the cached ER_MAP module so the next import picks up the fresh disk
  3. Asserts the train_grpo patches are live in the running module (kl-gate
     + use_kl loss branch + phase_episode_budgets parameter)
  4. Sets all hyperparameters EXPLICITLY — does not depend on any earlier
     cell's globals being correct
  5. Frees VRAM aggressively and asserts >= 6 GB free before launch
  6. Runs the Groq pre-flight (routing + 4-key liveness) and asserts all PASS
  7. Calls train() with phase_episode_budgets={1: 20, 2: 25, 3: 30}

Usage from a Kaggle notebook cell:

    exec(open("/kaggle/working/Meta_Finals/kaggle/clean_launch.py").read())

That one line is all you paste. Press play. Walk away for ~4 hours.
"""

import os, sys, gc, subprocess, importlib  # noqa: E401

# =============================================================================
# 1. Force repo to latest commit on origin/main
# =============================================================================
REPO_ROOT = "/kaggle/working/Meta_Finals"
print("[1/7] Updating repo to origin/main...")
subprocess.run(["git", "-C", REPO_ROOT, "fetch", "origin"], check=True)
subprocess.run(["git", "-C", REPO_ROOT, "reset", "--hard", "origin/main"], check=True)
subprocess.run(["git", "-C", REPO_ROOT, "log", "-1", "--oneline"])

# =============================================================================
# 2. Drop cached ER_MAP modules so import picks up the latest disk version
# =============================================================================
print("\n[2/7] Dropping cached modules...")
for _m in list(sys.modules):
    if _m.startswith("ER_MAP"):
        del sys.modules[_m]
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# =============================================================================
# 3. Verify all required patches are live in the running module
# =============================================================================
print("\n[3/7] Verifying patches...")
import inspect
import ER_MAP.training.train_grpo as tg

_train_src = inspect.getsource(tg.train)
assert "if kl_beta > 0.0:" in _train_src, (
    "FAIL: train() missing kl_beta gate. Pull the latest commit on origin/main."
)
assert "phase_episode_budgets" in _train_src, (
    "FAIL: train() missing phase_episode_budgets support."
)
assert "use_kl" in tg.manual_grpo_step.__code__.co_varnames, (
    "FAIL: manual_grpo_step missing 'use_kl' branch."
)
print("  OK — kl_beta gate live")
print("  OK — phase_episode_budgets supported")
print("  OK — use_kl branch in loss function")

# =============================================================================
# 4. EXPLICIT hyperparameters — does not rely on any previous cell's globals
# =============================================================================
print("\n[4/7] Setting hyperparameters (explicit, no Cell 9 dependency)...")

MODEL_NAME            = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
GROUP_SIZE            = 2
LEARNING_RATE         = 5e-6
KL_BETA               = 0.0   # T4-safe: skip reference model load (saves ~5 GB VRAM)
PHASE_EPISODE_BUDGETS = {1: 20, 2: 25, 3: 30}        # 75 episodes total
NUM_EPISODES          = sum(PHASE_EPISODE_BUDGETS.values())
PHASE_REWARD_TARGETS  = {1: 1.2, 2: 1.1, 3: 1.0}      # observational only
PHASE_MIN_WIN_RATE    = 0.20
CONVERGENCE_WINDOW    = 3
EARLY_STOP_ENABLED    = False  # forced off by train() under fixed-budget anyway
OUTPUT_DIR            = "/kaggle/working/er_map_grpo_checkpoints"

# Anti-fragmentation for the GRPO backward pass on T4
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# Groq traffic shaping — 8B for actors, 70B for judges
os.environ["ERMAP_NURSE_MODEL"]            = "llama-3.1-8b-instant"
os.environ["ERMAP_PATIENT_MODEL"]          = "llama-3.1-8b-instant"
os.environ["ERMAP_EMPATHY_JUDGE_MODEL"]    = "llama-3.3-70b-versatile"
os.environ["ERMAP_MEDICAL_JUDGE_MODEL"]    = "llama-3.3-70b-versatile"

# Episode budget controls (read by triage_env)
os.environ["ERMAP_MAX_EPISODE_STEPS"]      = "20"
os.environ["ERMAP_MAX_INTERNAL_EXCHANGES"] = "5"

print(f"  NUM_EPISODES          = {NUM_EPISODES}")
print(f"  PHASE_EPISODE_BUDGETS = {PHASE_EPISODE_BUDGETS}")
print(f"  GROUP_SIZE            = {GROUP_SIZE}")
print(f"  KL_BETA               = {KL_BETA}  (skip ref model)")
print(f"  PHASE_REWARD_TARGETS  = {PHASE_REWARD_TARGETS}  (observational)")

# =============================================================================
# 5. Free VRAM and assert headroom for the model load
# =============================================================================
print("\n[5/7] Freeing VRAM...")
import torch  # noqa: E402

for _name in ("model", "tokenizer", "ref_model", "optimizer"):
    if _name in globals():
        try:
            del globals()[_name]
        except KeyError:
            pass

gc.collect()
torch.cuda.empty_cache()
torch.cuda.ipc_collect()

_free, _total = torch.cuda.mem_get_info(0)
print(f"  VRAM free: {_free/1e9:.2f} / {_total/1e9:.2f} GB")
assert _free / 1e9 >= 6.0, (
    f"FAIL: only {_free/1e9:.2f} GB free; need >= 6 GB. "
    "Do Run -> Restart kernel, then re-run Cell 6 (mount), Cell 7 (secrets), "
    "and this cell. The kernel has unrecoverable VRAM fragmentation."
)

# =============================================================================
# 6. Groq pre-flight (routing + 4-key liveness)
# =============================================================================
print("\n[6/7] Pre-flight: Groq routing + key liveness...")
from ER_MAP.envs.api_router import AgentRouter  # noqa: E402

_router = AgentRouter()
_expected = {
    "nurse":         "llama-3.1-8b-instant",
    "patient":       "llama-3.1-8b-instant",
    "empathy_judge": "llama-3.3-70b-versatile",
    "medical_judge": "llama-3.3-70b-versatile",
}
_all_pass = True
for _role, _exp in _expected.items():
    _actual = _router._models.get(_role, "?")
    _client = _router._clients.get(_role)
    if _client is None:
        print(f"  [SKIP] {_role:14s} -> no Groq client (key missing)")
        _all_pass = False
        continue
    try:
        _resp = _client.chat.completions.create(
            model=_exp,
            messages=[{"role": "user", "content": "Reply with exactly: PING"}],
            max_tokens=4, temperature=0,
        )
        _api_ok = "PING" in (_resp.choices[0].message.content or "").upper()
        _err = ""
    except Exception as _e:
        _api_ok = False
        _err = f" ({type(_e).__name__}: {str(_e)[:80]})"
    _flag = "PASS" if (_actual == _exp and _api_ok) else "FAIL"
    print(f"  [{_flag}] {_role:14s} | model={_actual:25s} | api_ok={_api_ok}{_err}")
    if _flag == "FAIL":
        _all_pass = False
assert _all_pass, "Pre-flight FAILED. Re-run Cell 7 (secrets) and Cell 6 (repo)."

# =============================================================================
# 7. LAUNCH — fixed-budget GRPO training
# =============================================================================
print("\n[7/7] Launching GRPO training (75 episodes, fixed budget)...")
print("=" * 72)
print("  Phase 1 (Tool Mastery)            : 20 episodes")
print("  Phase 2 (Clinical Reasoning)       : 25 episodes")
print("  Phase 3 (Empathetic Negotiation)  : 30 episodes")
print("  Total                             : 75 episodes (~3-5 hours on T4)")
print("  HF Hub backup                     : every 20 episodes")
print("=" * 72)

metrics = tg.train(
    num_episodes=NUM_EPISODES,
    group_size=GROUP_SIZE,
    model_name=MODEL_NAME,
    groq_api_key=os.environ.get("GROQ_NURSE_API_KEY", "")
                 or os.environ.get("GROQ_API_KEY", ""),
    learning_rate=LEARNING_RATE,
    kl_beta=KL_BETA,
    use_wandb=False,
    output_dir=OUTPUT_DIR,
    dry_run=False,
    phase_reward_targets=PHASE_REWARD_TARGETS,
    phase_min_win_rate=PHASE_MIN_WIN_RATE,
    convergence_window=CONVERGENCE_WINDOW,
    early_stop=EARLY_STOP_ENABLED,
    phase_episode_budgets=PHASE_EPISODE_BUDGETS,
)
print("=" * 72)
print(f"\nTRAINING COMPLETE — {len(metrics)} metric records collected.")
print(f"Final LoRA adapter: {OUTPUT_DIR}/final_lora")
print(f"Plots will be rendered by Cell 15 (run it next).")
