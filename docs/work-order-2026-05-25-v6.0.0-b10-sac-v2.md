<!--
Author: Julian Bolivar (jbolivarg)
Version: 2.0.0
Date: 2026-05-25
-->

# Work order v2 — B10 of pc-rl-continuos v6.0.0 (canonical SAC): random warmup + commitment-annealing

> **For the PC-Inv_Pendulum harness owner/agent.** SUPERSEDES `work-order-2026-05-24-v6.0.0-b10-sac.md`
> (+§0b lr fix). After 7 in-library diagnostics, the root cause of the B10 miss is understood and a small
> library fix has landed upstream. This work order says exactly what to set and run for B10.

## 0. What the diagnostics established (so you don't re-chase dead ends)

The v6.0.0 SAC **code is correct** (pre-merge gate MAGI STRONG GO; FD-verified gradients; single-step
convergence proven for INTERIOR and BOUNDARY optima; the twin Q-critic provably learns Q^π with corr 0.92
given good (s,a) coverage). The B10 miss is **NOT** any of these (all ruled out by diagnostics):

- NOT the optimizer — **Adam was tested (critic, actor, both) and REFUTED**; plain SGD learns Q^π fine. Do
  NOT pursue Adam.
- NOT value-learning capacity — the critic represents Q^π correctly given coverage.
- NOT boundary commitment — single-step SAC reaches saturated optima cleanly.

**The actual bottleneck = the explore→commit balance:** (a) the random initial actor gives poor early
(s,a) COVERAGE → the critic only learns Q near the bad policy's actions → weak/misleading ∇_a Q; and
(b) the entropy temperature must ANNEAL α LOW so the DETERMINISTIC μ commits — if α stays high, the
entropy term (≈2α) dominates ∇_a Q and μ never commits (the deterministic eval stays poor / σ rebounds —
exactly the symptom in your first two B10 runs). **The earlier "sustained exploration" guidance
(target_entropy = −0.5, keep α high) was BACKWARDS — it prevents commitment. You need exploration EARLY
(coverage) then α annealed LOW (commitment).**

## 1. Library change that LANDED upstream (use it)

pc-rl-continuos now does **uniform-random ACTION warmup** during `learning_starts`: while the replay buffer has
< `learning_starts` transitions, the continuous-SAC agent executes/records actions drawn `Uniform(−1,1)` in
the squashed action space (proper coverage), then switches to the policy. **Set `learning_starts > 0` to
enable it** (default 0 = off). This is the coverage half of the fix; it's automatic once you set it.

## 2. Recommended config (the corrected levers)

```
gamma                    = 0.99
actor.lr_weights         = 3e-3          # SGD framework needs ~10x the Adam-standard (established)
q_critic.lr              = 3e-3
actor.hidden_layers      = [64,64] (Tanh); output_size = 2; output_activation = Linear; max_steps = 5-10
q_critic.hidden_layers   = [64,64]
replay_training_capacity = 100_000 ; replay_batch_size = 256
learning_starts          = 5_000        # NOW drives uniform-random warmup coverage (set it > 0!)
polyak_tau               = 0.005
# ── THE COMMITMENT LEVER (this is what the first runs got wrong): ──
target_entropy           = SWEEP −1 → −2 → −4   # MORE NEGATIVE forces α to anneal LOW → σ shrinks → μ commits.
                                                #   −1 (= −action_dim) may be too high (σ never commits, your run 1/2 symptom).
alpha_lr                 = 3e-3          # let α track the (lower) target promptly
log_alpha_init           = 0.0
# Optional explore→commit SCHEDULE (caller-side, like your v5 α/σ anneal): start target_entropy ≈ −1 for
# the first ~30% of training (coverage), then ramp to ≈ −4 (commitment). Mutating these per-episode is fine.
# leave: gae_lambda = None ; td_steps = 0 ; actor/critic hysteresis = false
```
- **Reward/obs normalization: ON** from the first run (keep as before).
- DEFER n-step (it gave only modest help in diagnostics; it's a FALLBACK if the commitment tuning still misses).

## 3. Instrument these (to know WHY if it misses)

Per eval, log (you already have σ/μ_raw/skip from v1):
- **`mean |∇_a Q|` vs `2·α`** at a few probe states — if `2α ≫ |∇_a Q|`, the entropy term is dominating →
  the policy can't commit → lower `target_entropy` / raise `alpha_lr`. (This was the decisive in-library
  signal: ratio 14–120× → no commitment.) The library exposes the Q critics; compute ∇_a Q via the
  action-value gradient at the deterministic action.
- **σ trajectory** — it MUST shrink over training (explore→exploit). If it plateaus high, α isn't annealing
  → make `target_entropy` more negative.
- `sac_skipped_actor_updates()` / `sac_skipped_critic_updates()` (should be 0).

## 4. Run B10

```
cd ../PC-Inv_Pendulum   # PC-RL-Continuos checked out on feature/v6.0.0-sac-continuous (path-dep)
# set config per §2 (learning_starts=5000; sweep target_entropy −1/−2/−4); ensure obs/reward norm ON
cargo run --release --bin multi_seed
```
PASS = deterministic eval ≥ 5/10 seeds > −400. Write results to `docs/results_v6_sac_b10_v2.md`.

## 5. If it still misses (in order)

1. **target_entropy more negative / faster α anneal** — the σ-doesn't-commit symptom is the #1 suspect;
   push commitment harder. Confirm via the `|∇_a Q|` vs `2α` log that α actually drops below the gradient.
2. **Larger `learning_starts`** (more random-action coverage, e.g. 10_000–20_000) and/or bigger actor/critic.
3. **n-step Q targets** — the deferred structural lever (would be a new upstream library feature; report back).
4. The PC-actor off-policy interaction is the residual wildcard (non-standard; flagged) — report the
   instrumentation and upstream will investigate.

## 6. After B10

PASS → upstream merges `feature/v6.0.0-sac-continuous` to main, tags `v6.0.0`, publishes. MISS → report the
per-seed numbers + the §3 instrumentation (esp. `|∇_a Q|` vs `2α` and the σ trajectory) → upstream decides
n-step vs further investigation. Library branch `feature/v6.0.0-sac-continuous`; the warmup feature is the
GREEN commit `186dfec`.
