<!--
Author: Julian Bolivar (jbolivarg)
Version: 3.0.0
Date: 2026-05-26
-->

# Work order v3 — B10 of pc-rl-continuos v6.0.0 (canonical SAC): saturation-lock REFUTED upstream → instrument and confirm

> **For the PC-Inv_Pendulum harness owner/agent.** SUPERSEDES `work-order-2026-05-25-v6.0.0-b10-sac-v2.md`.
> v2 landed the random-action warmup, lr=3e-3 SGD fix, and the commitment-annealing `target_entropy`
> sweep — `multi_seed` produced the stochastic-SOLVES milestone (s44 best_train ≈ −4) but the
> DETERMINISTIC μ locked at a constant saturated action (results_v6_sac_b10_v2.md). v3 closes the
> in-library investigation that v2 left open and tells you exactly what to instrument and run next.

## 0. What the upstream Step-0 diagnostic established (no Pendulum guesswork)

After the v2 hand-off the leading hypothesis was H-Jac (tanh-Jacobian attenuation × low α → μ
locks at the boundary state-INDEPENDENTLY). An in-library diagnostic
(`diagnose_sac_saturation_lock`, single-step contextual bandit, 4 probe states with `a*∈{+1,−1,+1,−1}`
— state-dependent BOUNDARY optima, same lr/batch/topology as your v2 config) compared the two
regimes that differ ONLY in `target_entropy`:

| | EXPLORE_HIGH `te=−0.5` | COMMIT_LOW `te=−4.0` |
|---|---|---|
| final α | 0.169 | **0.019** (annealed low as designed) |
| `err = mean\|μ_det − a*\|` | 0.220 | **0.150** (closer to the saturated optimum) |
| `var(μ_raw across states)` | 1.10 | **1.85** (MORE state-dependent) |
| mean `\|μ_raw\|` | 1.05 | 1.35 (genuinely saturated) |
| `\|∇_a Q\|` @ μ_det / @ a=0 | 0.07 (atten = 14×) | 0.06 (atten = 16×) |
| **var ratio (high/low)** | **0.597** | — |

**H-Jac REFUTED.** The tanh Jacobian IS attenuated ~14× at saturation, but commitment proceeds
anyway — COMMIT_LOW achieves *lower* error AND *higher* state-variance than EXPLORE_HIGH (the
opposite of what H-Jac predicted). **H-Rep REFUTED.** var(μ_raw) = 1.10 and 1.85 are well above
zero — the actor genuinely conditions on state in both regimes.

**The library SAC works on single-step.** With the v2 config it reaches state-dependent saturated
optima. The Pendulum failure mode is in something the single-step bandit does NOT have, and the
only remaining structural difference is the **200-step horizon + single-step bootstrap for the
soft-Bellman target**. The §5-3 deferred lever — n-step (or λ-) Q returns — is now the only
candidate consistent with the evidence.

## 1. Library API exposed for harness instrumentation (the v2 ask is satisfied)

Q-critics were `pub(crate)` in v6.0.0 baseline — v3 ships three public methods so you can compute
`|∇_a Q|` vs `2α` and probe `Q(s,a)` directly without unsafe access:

```rust
PcActorCritic::sac_alpha() -> Option<f64>
PcActorCritic::sac_q_min(state: &[f64], action: &[f64]) -> Option<f64>
PcActorCritic::sac_action_gradient_min(state: &[f64], action: &[f64]) -> Option<Vec<f64>>
```

`sac_q_min` returns `min(Q1, Q2)`. `sac_action_gradient_min` returns `∇_a min(Q1, Q2)` from the
critic that currently realises the min (clipped-double-Q). All three return `None` outside SAC
mode. Contract test: `test_sac_public_accessors_discriminate_mode_and_match_min_critic`. The
diagnostic itself is reproducible upstream as
`cargo test --release -- --ignored --nocapture diagnose_sac_saturation_lock`.

## 2. What to do next (sequenced)

### 2a. Re-run the v2 sweep with `|∇_a Q|` vs `2α` instrumentation enabled

Same v2 config (`learning_starts=5000`, `lr=3e-3`, `target_entropy=-2.0`, reduced batch 64 /
[32,32], obs/reward normalization on, polyak τ=0.005, alpha_lr=3e-3 — the configuration that gave
the cleanest v2 signal: monotonic σ shrink, stochastic SOLVES, deterministic lock at constant
saturated). Per eval, for **the deterministic action** `μ_det = tanh(μ_raw)` evaluated at the
probe state(s) of your choice (e.g., a small fixed set of 4–8 Pendulum states reset from canonical
initial conditions):

```
log per eval:
  alpha           = agent.sac_alpha().unwrap()           // scalar
  two_alpha       = 2 * alpha
  mu_det          = (run Play / agent.act(..., Play))    // scalar (action_dim=1)
  q_min_mu        = agent.sac_q_min(state, &[mu_det]).unwrap()
  q_min_0         = agent.sac_q_min(state, &[0.0]).unwrap()      // interior reference
  grad_mu         = agent.sac_action_gradient_min(state, &[mu_det]).unwrap()[0].abs()
  grad_0          = agent.sac_action_gradient_min(state, &[0.0]).unwrap()[0].abs()
  ratio_grad_2a   = grad_mu / two_alpha
  atten           = grad_mu / grad_0       // expect ~0.05–0.10 at saturation (matches upstream)
```

**What to look for (the decisive comparison):**

- If `grad_mu` is genuinely tiny AND `var(μ_raw)` across probe states is tiny on the LIVE
  Pendulum (despite single-step bandit success), then there IS a Pendulum-specific commitment
  failure beyond H-Jac/H-Rep — escalate with the per-state numbers.
- If `grad_mu` is comparable to the bandit's (`~0.15`) and `var(μ_raw)` is non-trivial, then the
  deterministic μ has the same magnitude of pathwise signal as in the working bandit — the
  failure is NOT a magnitude problem at saturation. **This is the predicted regime under the
  multi-step-credit-assignment hypothesis.** Move to 2b.

This is the diagnostic v2 promised but couldn't run; v3 makes it possible. Report numbers per
probe state per eval (CSV in `results/`).

### 2b. If 2a confirms the Pendulum failure isn't a saturation magnitude problem: stand down — upstream owns n-step

If the Pendulum `|∇_a Q|` vs `2α` and `var(μ_raw)` numbers match the bandit's working regime,
**no further harness tuning is on the table**. The remaining structural lever — multi-step Q
targets in the soft-Bellman backup — is internal to `sac_learn_step` and `sac_bellman_target` in
the library (R1 forbids editing pc-rl-continuos). Upstream will start a SBTDD cycle for n-step (or λ-)
returns in the SAC critic; v3 is the trigger.

### 2c. If 2a uncovers something else (unlikely but possible)

If the Pendulum probe shows a regime the bandit DOESN'T cover — e.g., `grad_mu ≈ 0` because of
specific Pendulum geometry, or `var(μ_raw) ≈ 0` despite COMMIT_LOW — that REOPENS the upstream
investigation. Report it; we'll cut a fresh diagnostic against the new symptoms.

## 3. What NOT to do (rule out, do not re-chase)

- Do NOT sweep `target_entropy` further — v2 already showed `te=−2` is the most stable; the
  single-step diagnostic confirms COMMIT_LOW is the correct regime.
- Do NOT push lr higher than `3e-3` — the §0b Step-0 diagnostic established this as the working
  range; the saturation-lock diagnostic confirmed convergence at the same lr.
- Do NOT pursue saturation-region "fixes" (clamp μ_raw, soft penalty on |μ_raw|, Jacobian
  compensation, Adam on actor) — the in-library diagnostic refutes saturation as the cause.

## 4. After 2a + 2b

PASS (B10 unblocked by n-step + the upstream SBTDD cycle) → upstream merges
`feature/v6.0.0-sac-continuous` to main, tags v6.0.0, publishes.

MISS still — report the 2a instrumentation in full. v3 explicitly does NOT plan ahead of that
report; the data shapes the next move.

## 5. Reference

- Upstream branch `feature/v6.0.0-sac-continuous` (HEAD will be the v3 commits).
- In-library diagnostic source: `src/pc_actor_critic/mod.rs::diagnose_sac_saturation_lock`.
- Public API source: `src/pc_actor_critic/sac.rs` (`sac_alpha` / `sac_q_min` /
  `sac_action_gradient_min`).
- Prior B10 evidence: `docs/results_v6_sac_b10.md`, `docs/results_v6_sac_b10_lr_fix.md`,
  `docs/results_v6_sac_b10_v2.md` (gitignored locally; harness-side authoritative).
