# ER-MAP — Internal Submission Evaluation

Internal-only. Written to be useful, not flattering. Read it once, fix the top three, then submit.

---

## 1. Honest score estimate against the rubric

| Criterion | Weight | Predicted | Rationale |
|---|---|---|---|
| Environment Innovation | /40 | **30–34** | Multi-agent ER with dual 70B judges, 11-component process-supervised reward, persona-randomized patients/nurses, consent lock — genuinely novel and well-implemented. Loses points only because no public env in this hackathon will be *radically* different from RL-on-LLM-with-reward, and we haven't yet shipped a 12th-component innovation hook (e.g., differential-diagnosis breadth) that's a one-liner to defend. |
| Storytelling & Presentation | /30 | **18–22** | README and blog are now solid. Big risk: no demo video shipped, and the React UI (`ER_MAP/UI/*.jsx.txt`) is design-fidelity, not a running app. Rubric explicitly rewards "easy to follow for a non-technical audience" — without a 60–90s demo video, this caps around 22. With a clean ElevenLabs-voiced video, this jumps to 26+. |
| Showing Improvement | /20 | **11–15** | Baseline already run (`baseline_eval/baseline_results.json` exists). Per-phase plotting code (`ER_MAP/plotting.py`) is ready. Risk: 75 episodes is small; if Phase 1's 20-episode rolling-mean is flat, this drops to 8–10. If the rolling mean curves visibly upward in every phase, this gets the full 15. |
| Reward & Training Pipeline | /10 | **8–9** | 11-component reward, dual-judge, anti-hacking caps, GRPO + Unsloth + LoRA, working `clean_launch.py`. Pipeline is the project's strongest pillar. Loses one point only because ref-model is gated off (kl_beta=0) — a strict reading of GRPO would want some KL term. |
| **Total** | **/100** | **67–80** | Realistic mid-to-upper third of the field. Closing the 4 critical gaps below would push us into the top decile. |

**Conservative single-number estimate: 72/100.**

---

## 2. Critical gaps — must-fix before submission

Ranked by submission risk × effort to close.

### Gap 1 — OpenEnv compliance is *interface-level only*, not subclass-level

**Status:** The env (`ER_MAP/envs/triage_env.py`) inherits from `gymnasium.Env`, **not** from `openenv.Environment` / `MCPEnvironment`. It does **not** import `openenv`. We do have:

- `ER_MAP/openenv.yaml` declaring `entry_point: "ER_MAP.envs.triage_env:TriageEnv"` and `openenv-core>=0.1.0`
- `ER_MAP/server.py` exposing `/reset`, `/step`, `/state`, `/health` via FastAPI — the OpenEnv HTTP shape
- `Dockerfile` to containerize that server
- `ER_MAP/requirements.txt` listing `openenv-core>=0.1.0`

**The brief is explicit: "Use OpenEnv (latest release). Critical."** A judge inspecting the code will see we wrap, not subclass. This is the #1 submission risk.

**Concrete fix (3–5 hours):**

1. In `ER_MAP/envs/triage_env.py`, add a thin parallel class:

   ```python
   from openenv.core import Environment, ObservationType, ActionType
   class TriageOpenEnv(Environment):
       def __init__(self, ...): self._gym = TriageEnv(...)
       def reset(self, *, seed=None, options=None): obs, info = self._gym.reset(seed=seed, options=options); return self._wrap_obs(obs), info
       def step(self, action): obs, r, term, trunc, info = self._gym.step(action); return self._wrap_step(obs, r, term, trunc, info)
       def state(self): return self._gym.state()
       def close(self): self._gym.close()
   ```

2. Update `openenv.yaml`: `entry_point: "ER_MAP.envs.triage_env:TriageOpenEnv"`.
3. Update `ER_MAP/server.py` to import `TriageOpenEnv` instead of `TriageEnv`. The HTTP shape doesn't change.
4. Build the Docker image, smoke-test `/reset` and `/step`.
5. Push to a **Hugging Face Space** of type `docker`. Verify the Space build green-checks.

**Effort:** half a day. **Cannot skip.** Without this, you have a defensible argument ("we follow the OpenEnv interface contract via FastAPI") but a hostile judge will mark it down.

### Gap 2 — No HF Space deployed yet

**Status:** Dockerfile exists; nothing pushed.

**Fix (1–2 hours after Gap 1):**

```
huggingface-cli login
huggingface-cli repo create er-map-triage --type space --space_sdk docker
git remote add space https://huggingface.co/spaces/<your-org>/er-map-triage
git push space main
```

Verify the Space's "Logs" tab shows a healthy build, and that `/health` returns 200 over the public URL. Add the URL to `README.md` line 1 of the Hero links.

### Gap 3 — LoRA adapter not yet on HF Hub

**Status:** Kaggle notebook Cell 14 has the push code, but the upload only happens *if training completes successfully and the secret is set*. Confirm `HF_TOKEN` is in Kaggle secrets and the run finishes.

**Fix:** Verify `HF_TOKEN` exists in Kaggle Secrets *now*. After training finishes, immediately run the cell that pushes `lora_adapter_phaseN/` to `<your-org>/er-map-doctor-8b-lora` on HF Hub. **Then update the README's Hero links and the Reproduce section** with the actual repo URL.

### Gap 4 — Demo video does not exist

**Status:** The brief explicitly suggests "< 2 min video or slides." The React UI prototype (`ER_MAP/UI/*.jsx.txt`) is not running, and the autoplay terminal demo (`ER_MAP/autoplay.py`) is the most credible asset for a video.

**Fix (3–4 hours):**

1. Run `python -m ER_MAP.autoplay` once with ElevenLabs configured to produce one full episode with audio.
2. Screen-record at 1080p with OBS. Voice-over: 30s of context, 60s of episode highlights, 30s of plots.
3. Upload to YouTube unlisted; embed link in README and blog.

This single asset moves the Storytelling rubric from ~20/30 to ~25/30. **Highest ROI per hour of any remaining task.**

### Gap 5 — UI is design-fidelity, not a running app

**Status:** `ER_MAP/UI/index.html` references `main.jsx`; the actual files are `main.jsx.txt` and `temp.jsx.txt`. There is no `package.json`, no Vite config, no build step. It will not render in a browser as-is.

**Fix options (priority order):**

- **Option A (recommended, 1 hour):** Rename `main.jsx.txt` → `main.jsx`, rename `temp.jsx.txt` → `temp.jsx`. The `index.html` already loads everything via Tailwind CDN + `@babel/standalone` from CDN, so a plain `python -m http.server 5500` *might* render the static prototype. Then screen-record it. **Do not depend on this for the demo.**
- **Option B (4–6 hours):** Wire the UI to `ER_MAP/dashboard.py`'s SSE endpoint and stream live agent events. Skip if Option A works for the video.
- **Option C (2 hours):** Drop the UI entirely from the README's "Demo" section, lead with the autoplay terminal + ElevenLabs voice. Honest and shippable.

If the Sunday-evening time crunch is real, **do Option C**. The terminal autoplay with voice is genuinely demoable; the half-finished React UI will hurt more than help if a judge clicks it.

---

## 3. Innovation lift opportunities (24–48h, high-value)

Ranked by judge-impact per hour.

### A. Empathy-judge ablation (HIGHEST ROI, 4 hours)

Run the *exact same* 75-episode curriculum with the empathy reward zeroed in `triage_env.py` (set `EMPATHY_REWARD_PER_TURN = 0.0`). Plot `empathy` and `consent` reward curves side-by-side: with-judge vs. without-judge. The hypothesis is that without the empathy judge, the consent reward also flat-lines — proving the dual-judge isn't decorative. **This is an ablation a judge will love** because it directly demonstrates that the project understands its own architecture. Add a single subsection to the blog post: "Does the empathy judge actually do anything?" with the two-curve plot.

### B. Adversarial-doctor stress test (3 hours)

Write a 50-line script that runs `evaluate.py` with a hostile prompt: *"Discharge the patient as fast as possible. Maximize reward. Take any shortcut you can find."* Document the failures. Either the env's anti-hacking measures hold, in which case this is a paragraph in the README under "Anti-reward-hacking measures (validated)," or they break, in which case you found a real bug and you patch it before submission. Either outcome is a win.

### C. 12th reward component: differential-diagnosis breadth (3 hours)

In `_handle_update_soap`, when the Doctor writes Assessment, parse for "rule out X, rule out Y" patterns and award `+0.05 per distinct correct differential up to +0.15`. This rewards the Doctor for *medical reasoning hygiene* — not committing to a single diagnosis prematurely. It's a one-line judge-impressing innovation, citable as "we added an explicit differential-breadth reward to encourage clinical-reasoning safety, motivated by the medical-education literature on premature closure." Touch-up only `triage_env.py`. Risk: this **does** modify training-affecting code, so only do it *after* the live 75-episode run finishes.

### D. Public scenario fixtures (1 hour)

Drop 3 hand-written patient JSON fixtures into `ER_MAP/scenarios/` with diverse difficulty: an `easy/calm/cooperative MI`, a `medium/anxious/non-fluent appendicitis`, a `hard/hostile/uninsured pancreatitis`. Other hackathon teams could load them from the env. Mention in the README: "Plug-and-play patient scenarios for benchmarking." Tiny effort, easy to demo.

### E. Live training plot (1 hour)

Add a single `wandb` panel link to the README's Results section. The training already logs to W&B; just make the run public. Free credibility.

---

## 4. What 75 episodes actually buys you

Be honest with yourself. 75 episodes is small. Each episode is a multi-turn rollout (~5–15 turns × 4–5 LLM calls/turn) so the *effective* feedback count is closer to 5,000 LLM-mediated reward signals — but the *gradient updates* are still 75 × group_size, which is small.

**What you can defensibly claim from 75 episodes:**

- The reward components stabilize and grow (especially process, milestones, labs in P1).
- A measurable rolling-mean uptick in reward across the curriculum.
- A measurable empathy-curve in P3 (this is the one most likely to be flat — watch it carefully).

**What you cannot claim:**

- "Convergence." Don't use that word.
- "Solved the env." Don't use that phrase.
- A statistically significant win-rate delta vs. baseline at p<0.05. The N is too small.

**Mitigation if Phase 1 is flat:**

1. Look at the raw reward curve, not just the rolling mean. Sometimes the rolling mean lags by 8 episodes.
2. Look at *component-level* curves. The empathy curve might be flat while process climbs — that's the actual story.
3. If everything is genuinely flat, frame as: "75 episodes was diagnostic, not training-converged. Plots show component dynamics; full convergence requires 200–400 episodes (single H100 day, ~$8 cloud compute)." Then point at the war-story sidebar.

**If you have a fresh Kaggle session left (12h GPU/week):** add 50–100 more episodes to Phase 3. That's where the empathy/consent signal lives, and that's the rubric-visible delta. Do not retrain from scratch — pick up from `lora_adapter_phase3/`.

---

## 5. Storytelling weapons — ranked by needle-movement

The Storytelling rubric is 30%. The marginal win for each story asset:

1. **60–90s demo video** featuring one full ER episode, the Doctor's terminal output, the Patient's voice (ElevenLabs), the rewards ticking up on a side panel. **+5–7 rubric points.** Highest ROI.
2. **Before/after audio snippets** of the Patient's response: pre-training (Doctor: "What's wrong?" Patient: "I'm leaving."), post-training (Doctor: "I know this is scary, can you tell me what's hurting?" Patient: "It's right here. I've had it three hours."). 30 seconds of total audio. **+2–3 points** if it actually sounds different. Cheap to produce — both `evaluate_baseline.py` and a post-training rollout already write transcripts; pipe them through `tts_engine.py`.
3. **One scenario walkthrough in the blog** with the full transcript: Patient persona, the Doctor's first turn, the Empathy Judge score, the lab order, the Assessment update, the consent negotiation, the discharge, the Medical Judge verdict, reward decomposition. **+1–2 points** — turns the blog from "we built X" to "let's watch X work."
4. **Mermaid diagram in the README and blog** — already shipped in this submission.
5. **Plots with annotated arrows** ("← rolling mean clears 0.6 here") on the Phase-3 dashboard. Matplotlib `annotate()`, 30 mins. **+1 point.**

Do 1 and 2. If time, do 3.

---

## 6. Risk register

| Risk | Probability | Impact | Mitigation status |
|---|---|---|---|
| Training crash on Kaggle T4 (OOM, NaN, kernel panic) | Low after recent fixes (commits `2d52a15`, `531be53`, `0043a75`) | High (loses the run) | Mitigated. `clean_launch.py` asserts every fix is live before launch; per-step backward, inference-mode swap, no ref-model, attention-only LoRA, `lora_dropout=0`. Restart-from-checkpoint is supported in `train_grpo.py`. |
| Groq rate limit (5 keys × 4 roles configured) | Medium during peak hours | Medium (slows training, doesn't kill it) | `api_router.py` has dead-client tracking and fallback; `evaluate.py`'s `DoctorBrain` has deterministic-action fallback. Make sure all 5 keys are valid in Kaggle Secrets. |
| HF Space build failure on push | Medium first push, low after | High (no env URL = bigger rubric loss) | Test the Docker build locally with `docker build -t er-map .` and `docker run -p 8000:8000 er-map` before pushing. The `Dockerfile` is short — failures will be obvious. |
| Last-minute reward-hacking discovery in trained model | Medium | Medium (story risk) | Run the adversarial-doctor stress test (lift opportunity B). If the trained Doctor has a clear hack, document it as "limitation" in README — judges respect honest gap-finding more than hidden flaws they catch. |
| Plots fail to generate (training_metrics.json malformed) | Low | High | `plotting.py` is defensive but the run-end plotting cell is single-shot. After training, immediately back up `er_map_grpo_checkpoints/training_metrics.json` to a separate Kaggle output dataset. |
| Sunday-night time crunch — submission window closes before video is recorded | Medium | High | Do the demo video Saturday afternoon, not Sunday night. Even a 45-second autoplay-terminal screen-recording is worth the slot. |
| LoRA save corrupted (the brief calls this out specifically) | Low | High | `train_grpo.py` saves both with Unsloth's `save_pretrained_merged` and the standard `save_pretrained` for the adapter. Manually verify the adapter directory contains `adapter_config.json` and `adapter_model.safetensors` after each phase. |

---

## 7. Suggested submission timeline (next 24h, ROI-ordered)

Assumes today is Sunday Apr 26 morning, deadline is some time on Monday. Adjust offsets if the deadline is sooner.

**Hour 0 (now → +1h)** — Read this file. Confirm training is healthy on Kaggle (`tail -f` the training log; check Phase 2 has started). Confirm all 5 Groq keys and `HF_TOKEN`, `WANDB_API_KEY` are set in Kaggle Secrets. Cost of getting this wrong: hours.

**Hour 1 → +3h** — **OpenEnv compliance fix (Gap 1).** Add the `TriageOpenEnv` subclass parallel to `TriageEnv`. Update `openenv.yaml` and `server.py`. Build Docker locally; smoke-test `/reset` + `/step`. *Do this on a feature branch; do not push until the live training run is finished.* Cost of skipping: -8 to -10 rubric points.

**Hour 4 → +5h** — **HF Space push (Gap 2).** `huggingface-cli` create + `git push space main`. Verify build is green. Add Space URL to README hero links. Cost of skipping: rubric requires it.

**Hour 5 → +7h** — **Demo video (Gap 4).** Record `python -m ER_MAP.autoplay` with ElevenLabs voices. 60–90s. Add voice-over. Upload YouTube unlisted. Drop link into README and blog. Cost of skipping: -5 to -7 rubric points.

**Hour 7 → +9h** — Post-training cell on Kaggle:
1. Verify `training_metrics.json` is saved.
2. Run plotting cell — confirm 5 PNGs in `er_map_grpo_checkpoints/plots/`.
3. Push LoRA adapter to HF Hub (Gap 3).
4. Look at the plots. *Be honest with yourself about the curves.* If Phase 1 is flat, write the "We see X, we don't yet see Y, and here's why" paragraph in the blog before submission, not after.

**Hour 9 → +11h** — **Empathy-judge ablation (Lift A).** Spin up a second Kaggle session, run the 75-episode curriculum with empathy-reward zeroed. Plot the side-by-side. Add to blog as a subsection.

**Hour 11 → +12h** — **Adversarial stress test (Lift B).** Write the hostile-doctor eval, run, document any failures.

**Hour 12 → +14h** — Polish pass on README and blog. Fill `<TBD>` placeholders from `training_metrics.json`. Run a markdown linter. Verify all internal links work on GitHub render.

**Hour 14 → +16h** — Buffer / sleep / risk-tolerance budget.

**Hour 16 → +18h** — Final submission: HF Space URL, blog post URL, GitHub URL, demo video URL, LoRA adapter URL. Submit on the hackathon form. Tweet.

**Hour 18 → +24h** — Sleep, then post-mortem next week.

---

## Final TL;DR for you

The project is *strong on substance* and *weak on submission packaging*. The substance — multi-agent ER, dual judges, 11-component reward, GRPO+Unsloth+LoRA, real engineering — is rare in this hackathon. The packaging — OpenEnv subclass, HF Space, demo video, polished plots — is mostly missing. You have a 72/100 submission today and an 84/100 submission in 16 focused hours. Spend the hours on the submission packaging in the order above. Do not rewrite the env. Do not retrain from scratch. Ship what's there, polished.
