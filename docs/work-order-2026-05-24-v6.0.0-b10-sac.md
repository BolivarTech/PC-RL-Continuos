<!--
Author: Julian Bolivar (jbolivarg)
Version: 1.0.0
Date: 2026-05-24
-->

# Work order — B10 validation of pc-rl-core v6.0.0 (canonical SAC) on the PC-Inv_Pendulum harness

> **For the PC-Inv_Pendulum harness owner/agent.** Cross-repo handoff (mandate R1: the library is
> fixed; the harness adapts its agent construction + tunes hyperparameters against B10). pc-rl-core
> v6.0.0 replaces the continuous on-policy score-function path with **canonical off-policy SAC**.
> This work order says how to bump the harness to the SAC API, a concrete starting config, how to
> run **B10**, the pass criterion, and the escalation levers if B10 misses.

## 0. Status of the library (upstream)

- Branch `feature/v6.0.0-sac-continuous` (HEAD `fd14d60`). NOT merged to main, NOT tagged/published
  (publish is held until B10 passes — that is the purpose of this work order).
- Pre-merge gate PASSED: `/requesting-code-review` clean-to-go + MAGI **STRONG GO (3-0)**. The actor
  pathwise gradient and `∇_a Q` are finite-difference-verified including at saturation; in-library
  directional guards (μ_raw bounded, pathwise drives μ, σ-narrowing, Q learns, auto-temp) pass.
- **B10 is the authoritative deterministic-convergence check** (R10/R12): the minimal in-library
  Pendulum cannot reproduce the stochastic-solves gap, so convergence is proven only here.
- The harness already depends on the library via `pc-rl-core = { path = "../PC-RL-Core" }`, so with
  PC-RL-Core checked out on this branch the harness builds against v6.0.0 directly.

## 0b. ⚠ CRITICAL CORRECTION (2026-05-25, from Step-0 in-library diagnostic — read before tuning)

The first B10 attempt MISSED. An upstream white-box diagnostic
(`diagnose_sac_contextual_bandit`, an in-library contextual bandit with state-dependent interior
optima) **root-caused it: the recommended `lr = 3e-4` in §3 was ~10× TOO LOW for this library's
plain-SGD framework** (pc-rl-core uses fixed-lr SGD, NOT Adam; 3e-4 is the *Adam*-standard, whose
effective step is far larger than a raw SGD step). At `lr = 3e-4` the Q-critic never learns a
discriminative surface (argmax_a Q stuck at the action-grid edge, Q(a*)≈Q(−a*)) → the actor gets no
signal → μ stays random/saturated and σ never collapses — EXACTLY the symptoms the first B10 run
reported. At **`lr = 3e-3` (×10)** the diagnostic SAC converges cleanly: Q learns the correct interior
optima, μ tracks argmax_a Q, σ collapses to ~0.15–0.27, α settles. The mechanism is correct; the lr was
the bottleneck. The first run swept `target_entropy`/`alpha_lr`/capacity/activation but NOT the main
actor/Q-critic lr (because §3 pinned 3e-4) — so it never hit the working regime.

**ACTION: re-run B10 with `actor.lr_weights = q_critic.lr ≈ 3e-3` (sweep 1e-3…1e-2).** §3 below is
corrected accordingly. Caveat (honest): the diagnostic is single-step (no temporal horizon), so the
higher lr is NECESSARY and high-confidence, but Pendulum's 200-step horizon is untested in-library — if
B10 still misses *after* the lr fix, the next diagnostic is the horizon/credit-assignment path (n-step
Q targets), not the core mechanism.

## 1. B10 pass criterion (unchanged)

`multi_seed` 10×500: **deterministic** eval (Play mode, `tanh(μ_raw)`, no exploration noise) mean
clears ≈ −500 with **≥ 5/10 seeds > −400** on Pendulum-v1.

## 2. The v6.0.0 SAC API surface — what changed vs v5 (BREAKING for continuous)

The harness's agent construction (`src/agent.rs` / `src/config.rs`) must be updated. Continuous mode
is now SAC; the old continuous fields are gone/ignored:

- **Actor emits μ AND log_σ:** set `actor.output_size = 2 * action_dim` (Pendulum action_dim = 1 ⇒
  `output_size = 2`). `actor.output_activation = Activation::Linear` (required). σ is learned
  per-state; **`policy_sigma` is IGNORED** in SAC mode.
- **New required config field `q_critic: Some(QCriticConfig{ state_dim, action_dim, hidden_layers, lr })`**
  — `state_dim` must equal `actor.input_size`; `action_dim` is the action dim (1).
- **Replay REQUIRED:** `replay_training_capacity > 0` and `replay_batch_size > 0` (≤ capacity). SAC
  forces `positive_only = false` internally for continuous (Pendulum rewards are ≤ 0).
- **New optional fields:** `target_entropy: Option<f64>` (default `None` ⇒ `−action_dim`),
  `log_alpha_init: f64` (default 0.0 ⇒ α₀ = 1), `alpha_lr: f64` (default 0.001),
  `learning_starts: usize` (default 0 ⇒ warmup = batch_size). `polyak_tau` reused (default 0.005).
- **Rejected for continuous SAC at construction:** `actor_hysteresis`/`critic_hysteresis = true`,
  `td_steps > 0`, `gae_lambda = Some(..)` (leave None / false), `q_critic = None`,
  `output_size != 2*action_dim`, non-Linear output. `apply_config` cannot change q_critic topology.
- New public items: `QCritic`, `QCriticConfig`, `QCriticWeights`, `QCriticCpu`, `Layer::input_gradient`.
- **Action mapping:** the policy outputs `a = tanh(a_raw) ∈ (−1, 1)`; Pendulum torque ∈ [−2, 2], so
  scale the executed action by ×2 (the harness already does action scaling — keep it; the deterministic
  eval action is `2 * tanh(μ_raw)`).

## 3. Recommended STARTING config for Pendulum-v1 (tune from here against B10)

Standard SAC-for-Pendulum values, adapted to the PC actor's cost. **These are a starting point — the
harness tunes them against B10 (R1).**

```
gamma                    = 0.99            # Pendulum standard (NOT 0.95)
actor.input_size         = 3               # (cosθ, sinθ, θ̇)
actor.hidden_layers      = [64, 64] (Tanh) # or [32,32] for speed; softsign also fine
actor.output_size        = 2               # 2 * action_dim
actor.output_activation  = Linear
actor.max_steps          = 5–10            # PC inference depth. LOWER than 20 to cut B10 wall-clock
                                           #   (T0 spike: 15.7 ms/update @ max_steps=20 → ~4.4 h for 10×500;
                                           #    max_steps=5 roughly quarters the actor-inference cost)
actor.lr_weights         = 3e-3            # CORRECTED (was 3e-4): SGD framework needs ~10× the Adam-standard lr (see §0b); sweep 1e-3…1e-2
q_critic.state_dim       = 3
q_critic.action_dim      = 1
q_critic.hidden_layers   = [64, 64] (Tanh) # or [256,256] if compute allows
q_critic.lr              = 3e-3            # CORRECTED (was 3e-4): same SGD-vs-Adam reason (§0b) — this is the lr that starved the critic in run 1; sweep 1e-3…1e-2
replay_training_capacity = 100_000         # 1e5 (1e6 also fine if memory allows)
replay_recent_capacity   = 0
replay_batch_size        = 256             # SAC standard; must be ≤ capacity
learning_starts          = 1_000           # collect ~1k transitions before learning
polyak_tau               = 0.005
target_entropy           = None            # ⇒ −action_dim = −1.0 (Pendulum standard); tune if needed
log_alpha_init           = 0.0             # α₀ = 1.0
alpha_lr                 = 3e-3            # raised with the rest (SGD framework, §0b); the temperature was NOT the bottleneck — the actor/Q lr was
# leave these at SAC-safe values:
gae_lambda = None ; td_steps = 0 ; actor_hysteresis = false ; critic_hysteresis = false
```

- **Reward/observation normalization: ENABLE from the first B10 run** (decided, per MAGI Caspar) —
  do NOT defer it as a later lever. Normalize observations (running mean/std) and scale/normalize the
  reward; Pendulum's raw reward ∈ [≈−16.3, 0]. This materially affects SAC convergence.
- **One SAC update per environment step** after `learning_starts` (the harness's `step_continuous`
  call already drives `sac_learn_step` internally — just keep stepping).
- Training budget: 500 episodes × 200 steps = 100k env steps per seed (matches B10's 10×500). SAC on
  Pendulum typically solves in ~10–20k steps, so 100k is ample.

## 4. How to run B10

```
# PC-RL-Core checked out on feature/v6.0.0-sac-continuous (the path dep picks it up):
cd ../PC-Inv_Pendulum
# 1) Update src/agent.rs / src/config.rs / src/cli.rs to build the SAC config in §3.
# 2) Build + run the 10×500 sweep:
cargo run --release --bin multi_seed
# (add CLI flags for the SAC hyperparameters if you want to sweep them, mirroring the v5 anneal CLI)
```
Verdict = deterministic eval over the final window, NaN-aware: PASS needs **≥ 5/10 seeds eval > −400**.
Record per-seed deterministic eval means in a results doc (mirror `docs/results_v5_alpha_sigma_anneal.md`).

## 5. Escalation levers if B10 misses (in priority order)

The in-library SAC mechanism is correct (FD-verified) regardless of B10, so a miss is a TUNING/harness
matter, not a library bug. Try in order:

1. **Actor + Q-critic learning rate FIRST (the confirmed run-1 root cause, §0b)** — `actor.lr_weights`
   and `q_critic.lr` at **~3e-3, sweep 1e-3…1e-2**. This is the lever the first run missed; the Step-0
   diagnostic proves SAC fails at 3e-4 and converges at 3e-3 in-library. Only after the lr is in the
   working regime do the other knobs matter: `target_entropy`, `alpha_lr`, `learning_starts`,
   `replay_batch_size`, `polyak_tau`, `actor.max_steps` (PC depth), hidden sizes.
2. **Reward/obs normalization** — confirm it is ON and correct (running stats); try reward scaling.
3. **σ / log_σ diagnostics (MAGI Caspar)** — log `mean σ` and `mean|μ_raw|` over training so a σ
   instability (σ collapsing too fast → premature exploitation, or never collapsing → never commits)
   is caught early rather than only as a missed convergence number. The library exposes
   `sac_skipped_actor_updates()` / `sac_skipped_critic_updates()` getters — log them too (nonzero ⇒
   non-finite updates being skipped).
3b. **PC actor topology coupling** — the actor's μ and log_σ share the PC hidden trunk; if σ behaves
   pathologically, try a wider/deeper actor or different activation (softsign) before concluding.
4. **n-step Q targets** — if single-step bootstrapping is too slow, a (harness-side) n-step return for
   the Q target can help; this is the next structural lever.
5. **Last resort (library change, new SBTDD cycle):** prioritized replay, distributional Q, or a
   separate μ/log_σ head — only if 1–4 are exhausted.

## 6. After B10

- **PASS (≥5/10 > −400):** report back; upstream then merges `feature/v6.0.0-sac-continuous` to main,
  tags `v6.0.0`, and publishes (CD). v6.0.0 = the canonical-SAC convergence fix, validated.
- **MISS:** report the per-seed numbers + the σ/μ_raw diagnostics + which levers were tried; upstream
  decides between further harness tuning vs a library escalation (a new SBTDD cycle).

## 7. Reference

- Upstream branch `feature/v6.0.0-sac-continuous` @ `fd14d60`; spec `sbtdd/spec-behavior.md`; plan
  `planning/claude-plan-tdd.md` (both gitignored locally upstream). CHANGELOG `## [6.0.0]`.
- Prior B10 evidence: `docs/results_v410.md` (v4.1.0), `docs/results_v5_alpha_sigma_anneal.md` (v5.0.0
  — Option A annealing exhausted, motivated this SAC rewrite).
