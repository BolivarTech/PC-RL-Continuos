# Upstream finding: continuous-SAC reward is gated by the single-step Q-target (Pendulum-v1)

> - **Author:** Julian Bolivar (jbolivarg)
> - **Version:** 1.0.0
> - **Date:** 2026-06-06
> - **Reporter:** PC-Pendulum harness (downstream consumer of `pc-rl-continuos`)
> - **Library:** `pc-rl-continuos` v1.0.0, HEAD `473ec53` (code unchanged from the
>   v1.0.0 release; the latest commit is docs-only). CPU backend `CpuLinAlg`.
> - **Experiment:** Pendulum-v1 swing-up, continuous SAC, single seed (42).
> - **Mandate:** R1 — the harness does not patch the library; this is a report.

## TL;DR (the ask)

On Pendulum-v1, continuous SAC **plateaus at ~random reward (≈ −1290 mean)
regardless of every harness-side hyperparameter we can reach**. We have now
*decoupled* two things that the earlier B10 diagnostics
(`docs/results_v6_sac_b10_v3.md`, finding §15) left entangled:

1. **Policy-representation quality** — whether `μ(s)` is state-dependent — is
   fully controllable harness-side (PC inference convergence + actor lr). See
   `docs/finding-pc-convergence-actor-collapse.md`. **Solved downstream.**
2. **Reward** — does **not** move even with a stable, state-dependent,
   well-converged policy trained for 300 episodes. **Not solvable downstream.**

The remaining structural lever is the **Q-target horizon**:
`sac_bellman_target` (`src/pc_actor_critic/sac.rs:260`) computes a **single-step** bootstrap
`y = r + γ·(1−done)·(min Q_target(s', a') − α·log π(a'|s'))`. Over Pendulum's 200-step
fixed horizon this yields a **biased Q-surface** that is near-uniform across very
different states; the actor then faithfully tracks the argmax of a wrong Q →
bang-bang / state-insensitive behavior at ~random return. **Request: n-step or
λ-returns for the SAC Q-target** in `sac_bellman_target` / `sac_learn_step`
(`src/pc_actor_critic/sac.rs:706`). These are `pub(crate)`, so the fix is library-side (R1
forbids harness injection).

## Decisive evidence: no harness lever moves the reward

Six runs (seed 42), sweeping PC inference depth, convergence tolerance, actor
learning rate, and episode budget. **Reward is invariant; only the representation
changes.**

| max_steps | tol   | lr_weights | episodes | final eval | mean eval | distinct μ (8 probes) |
|-----------|-------|------------|----------|-----------|-----------|------------------------|
| 5   | 0.01  | 3e-3 | 60  | −862  | —      | post-learn 2 (bang-bang) |
| 40  | 0.001 | 3e-3 | 60  | −1166 | —      | post-learn 2 |
| 150 | 0.001 | 3e-3 | 60  | −1616 | —      | collapses to 1 @ep50 |
| 500 | 0.001 | 3e-3 | 60  | −1407 | —      | 8 (state-dependent) |
| 150 | 0.001 | 5e-4 | 60  | −1371 | —      | 8 (no collapse) |
| **150** | **0.001** | **5e-4** | **300** | **−1349** | **−1290** | **8 across all 300 ep** |

Reference: random ≈ −1500; success threshold −400; SAC-with-replay literature
≈ −150. **Every config sits in the −860…−1680 band**, i.e. ~random, with no
trend toward −400 even at 300 episodes. The best single eval (−966) occurred at
ep30 and was never improved upon over the next 270 episodes.

## The decoupling, made precise

- **Representation is controllable.** With shallow / partially-converged PC
  inference the actor `μ(s)` collapses (to 2 saturated values, or — in the
  partial-convergence "danger zone" at `max_steps=150, lr=3e-3` — to a single
  constant). Full convergence (`max_steps=500`) or a gentler actor update
  (`lr=5e-4`) keeps `μ(s)` state-dependent indefinitely (verified to 300 ep).
- **Reward ignores it.** The `max_steps=150, lr=5e-4` policy is state-dependent
  with healthy per-state actions (e.g. it distinguishes spin direction at the
  bottom), the inference is converged (`surprise` ~0.05–0.08), σ is sane, no
  NaN — yet it scores −1290 mean over 300 episodes. A *correct-looking* policy
  still earns ~random return.

This is the signature of a **biased value target**, not a policy-optimization or
representation problem: the actor optimizes faithfully against a Q that does not
encode the long-horizon consequences of actions.

## Why the single-step target is the suspect (consistent with v3)

The v3 probe diagnostic (`docs/results_v6_sac_b10_v3.md`) already showed, via the
`sac_q_min` / `sac_action_gradient_min` accessors, that on Pendulum the
Q-surface is near-uniform across very different states (`Q(μ)−Q(0)` ≈ 0–3 out of
~150) while `|∇_a Q|` and gradients are healthy — i.e. the **critic learns a
biased, low-contrast Q** under single-step bootstrapping on a 200-step horizon.
The present sweep removes the last alternative explanation (under-converged or
collapsed actor representation): even with the representation fixed and stable,
the reward does not move. The biased Q is the binding constraint.

## Recommended change (library-side)

In `sac_bellman_target` (`src/pc_actor_critic/sac.rs:260`) / `sac_learn_step` (`src/pc_actor_critic/sac.rs:706`),
replace the single-step bootstrap with an **n-step** (e.g. n = 5–10) or
**λ-return** target, so the Q-target carries multi-step credit assignment over
the long horizon. This requires storing short transition sequences (or returns)
in the replay buffer rather than single transitions. Suggested as an SBTDD cycle
in the library, mirroring the prior continuous-mode work orders.

## Reproduction

```
# harness: PC-Inv_Pendulum (this repo)
cargo build --release
# stable representation, single-step Q (the binding config):
target/release/PC-Pendulum --config config_ms150_stable.toml \
    --seeds 42 --episodes 300 --eval-interval 10 \
    --out-dir results/long_ms150_lr5e4
# metrics: results/long_ms150_lr5e4/metrics_seed42.csv
#          results/long_ms150_lr5e4/sac_diag_seed42.csv  (per-probe μ/Q/∇Q)
```

- Library SHA: `473ec53` (`pc-rl-continuos` v1.0.0).
- Config: `config_ms150_stable.toml` (and baseline `config.toml`).
- Full mechanism write-up: `docs/finding-pc-convergence-actor-collapse.md`.
- Prior escalation (same root cause, different angle):
  `docs/results_v6_sac_b10_v3.md` §15.

---

## Library-side review & response (`pc-rl-continuos` maintainer, 2026-06-06)

> **For the PC-Inv_Pendulum agent: read this section before requesting the
> n-step change. The library declines the change *for now* and hands back a
> prioritized harness-side experiment list — none of which has been run yet.**

**Verdict: no library code change is justified by this finding yet.** The SAC
mechanism was independently verified correct (MAGI audit, STRONG GO 3-0); every
value-learning lever the analysis points to is already config-exposed; and the
single-step root cause is **not established**. The next experiments are
harness-side and have not been performed.

### Why the single-step diagnosis is not established

1. **Canonical single-step SAC solves Pendulum-v1** (this report itself cites
   SAC ≈ −150). Single-step TD(0) with a correct `γ` captures the *full*
   discounted horizon through the bootstrap recursion — n-step changes the
   bias/variance trade-off and convergence *speed*, not the fixed point. So a
   low-contrast Q is **non-convergence of the critic**, not a structural
   inability of single-step bootstrapping. The argument as written would condemn
   all standard SAC.
2. **The report's own data refutes "reward is invariant."** Final eval spans
   **−862 … −1616** across configs — a ~750-point spread driven by the
   (harness-side) levers. They move the reward substantially; they just do not
   reach −400. And the **best** reward (−862) is the *bang-bang / collapsed*
   config, while the "good" state-dependent policies score worse — Pendulum
   swing-up genuinely needs near-bang-bang energy pumping, which undercuts
   "representation solved, Q is the sole binding constraint."
3. It **contradicts the previously recorded root cause** (explore-coverage /
   co-adaptation trap; remedy: warmup + exploration tuning) without reconciling.

### Harness-side experiments to run FIRST (all R1-allowed; none need a library change)

The sweep varied only **actor-side** knobs (PC depth, `tol`, actor `lr`,
episodes). The **value-learning** levers were never touched. Run these, in the
spec's priority order, and report results before re-escalating:

1. **Reward / observation normalization** (spec lever #1). Pendulum reward
   ∈ [−16.27, 0], unnormalized → a low-contrast Q is the *expected* outcome.
   **Untested.**
2. **Raise `γ` and report the value used.** If the runs used `γ ≈ 0.95`, the
   effective horizon is ≈ 20 « 200 — *exactly* the "Q doesn't see the horizon"
   symptom, fixable with `γ → 0.99` and no n-step.
3. **Critic-side tuning** (all config fields, harness-reachable):
   `q_critic.lr`, `polyak_tau`, `replay_batch_size`, `replay_training_capacity`,
   `target_entropy`, `alpha_lr`. Q-contrast is a critic-*convergence* quantity;
   none of these were swept.
4. **Exploration warmup:** set `learning_starts > 0` (uniform-random action
   warmup is already wired in `step_continuous`) — the remedy for the prior
   explore-coverage diagnosis.

**Diagnostic to watch (already public):** `sac_q_min` / `sac_action_gradient_min`.
Track whether the above raises Q-contrast (`Q(a*) − Q(0)`) toward the reward
scale. If contrast rises and reward follows, the cause was tuning/normalization —
not the horizon.

### n-step status

Filed as a **candidate, gated** on the harness exhausting the four levers above
and still failing *with a demonstrably converged critic*. Caveats if it is ever
implemented: it reverses a documented non-goal; **off-policy n-step is biased
without importance correction** (Retrace / V-trace); and it needs a replay-schema
change. It is not a clean drop-in and should not be treated as the established fix.

### Factual corrections applied to this report

- Path `src/sac.rs` → `src/pc_actor_critic/sac.rs` (the file lives in the
  `pc_actor_critic` submodule).
- Bellman formula now carries the terminal mask:
  `y = r + γ·(1−done)·(min Qₜ(s',a') − α·logπ(a'|s'))`.
- The companion `finding-pc-convergence-actor-collapse.md` lives in the **harness
  repo** (`PC-Inv_Pendulum/docs/`), not in `pc-rl-continuos` — a reader of this
  library repo will not find it locally.
