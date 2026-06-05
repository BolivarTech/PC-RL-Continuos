<!--
Author: Julian Bolivar (jbolivarg)
Version: 1.0.0
Date: 2026-05-26
-->

# v6.0.0 SAC v3 §2a probe diagnostic — saturation-magnitude RULED OUT on Pendulum. §2b activated: stand down for upstream n-step.

> Follow-up to `docs/results_v6_sac_b10_v2.md`. Work order v3 (2026-05-26) refuted the v2 H-Jac
> (tanh-Jacobian attenuation × low α → state-INDEPENDENT lock) hypothesis upstream via a single-step
> diagnostic with state-dependent boundary optima: COMMIT_LOW (te=−4) reaches LOWER error AND HIGHER
> state-variance than EXPLORE_HIGH (te=−0.5), the opposite of H-Jac. v3 then shipped 3 public SAC
> accessors (`sac_alpha`, `sac_q_min`, `sac_action_gradient_min`) so the harness can run the
> "Pendulum vs the working bandit" comparison the v2 ask required but couldn't deliver.
>
> **Result: §2b is decisively activated.** Pendulum's `|∇_a Q|` is ~10× LARGER than the bandit's
> (not tiny), the tanh-Jacobian attenuation is ABSENT in the Q-critic (atten ratio ≈ 1.0, not 0.07),
> the entropy term is dominated by the Q-gradient (`grad_μ / 2α = 38–54×`), and `var(μ_raw)` is
> 2–3× HIGHER than the bandit's working regime. Pendulum is NOT a saturation/magnitude problem.
> The only remaining structural difference with the working bandit is **single-step bootstrap over a
> 200-step horizon** → upstream n-step / λ-returns is the right (and now only) next lever.

---

## 1. What the harness shipped (work order v3 §2a)

- `src/probe_diag.rs` — 8 canonical Pendulum probe states (up/down/sides × still/spinning) +
  `run_probe_diag(&Agent, &ObsNormalizer, &probes)` using the v3 SAC accessors `sac_alpha`,
  `sac_q_min`, `sac_action_gradient_min`; `ProbeDiagCsv` sink (one file per seed).
- `src/training.rs` wires the diagnostic into `train_seed` — per eval, the agent's deterministic
  policy is probed at the 8 states and `(α, μ_det, q_min_μ, q_min_0, |grad_μ|, |grad_0|)` is logged
  to `results/<out_dir>/sac_diag_seed{S}.csv`.
- TDD: RED (stub returns zeros → finite-α assert fails) → GREEN (real impl using the 3 accessors)
  → wire. Suite: **44/44**, clippy/fmt/build/doc clean. Reused the work-order v2 most-stable config
  (reduced batch 64, [32,32], warmup 5000, lr 3e-3, te=−2) so the comparison is apples-to-apples.

## 2. The decisive numbers (3 seeds × 300 eps, final eval, mean over 8 probes)

| seed | α | 2α | mean `|grad_μ|` | mean `|grad_0|` | atten = `grad_μ / grad_0` | `grad_μ / 2α` | `var(μ_raw)` |
|---|---|---|---|---|---|---|---|
| 42 | 0.0067 | 0.013 | **0.72** | 0.74 | **0.96** | **54×** | **4.64** |
| 43 | 0.0105 | 0.021 | **0.89** | 1.01 | **0.89** | **43×** | **5.65** |
| 44 | 0.0133 | 0.027 | **1.01** | 1.02 | **0.98** | **38×** | **4.01** |
| _upstream bandit COMMIT_LOW_ | — | — | _0.06_ | — | _0.06 (≈ 1/16×)_ | _modest_ | _1.85_ |

**Pendulum's numbers exceed (or match) the working bandit's COMMIT_LOW regime in every dimension
relevant to the magnitude / commitment hypothesis.**

## 3. Interpretation

- **`|∇_a Q|` is LARGE, ~10× the bandit's.** The Q-critic has plenty of input-gradient signal at
  the deterministic action. NOT a magnitude failure.
- **`atten ≈ 1.0` — no tanh-Jacobian attenuation visible in the Q-critic's gradient.** The bandit
  showed a ~14× attenuation at saturation (atten ratio 0.07). Here the gradient at `μ_det` is
  essentially the same as at `a = 0`. The Q-critic's input gradient is decoupled from the actor's
  tanh squash — exactly as the upstream diagnostic predicted (and as expected for `∇_a Q` queried
  on the critic network directly).
- **The entropy term is dominated (`grad_μ / 2α = 38–54×`).** α annealed low (0.007–0.013) per the
  commitment lever; `2α ≈ 0.02` is negligible vs the Q-gradient ~0.9. The work order's failure mode
  "if `2α ≫ |∇_a Q|`, the policy can't commit" is **inverted** here — the policy CAN commit
  (and does), `2α ≪ |∇_a Q|`.
- **`var(μ_raw) = 4–5.6` is HIGH** — 2–3× the bandit's COMMIT_LOW. The actor genuinely conditions
  on state.

But: **deterministic eval is still −1278 (0/3 > −400).** And looking at the per-probe table for
each seed, `μ_det` clusters into only **two saturated values** (e.g., seed 42: −0.985 / +0.955),
and the Q-critic's preference between `μ_det` and `a = 0` is **tiny** (`q_min_μ − q_min_0 ≈ 0–3`
out of ~−150 magnitude). The high `var(μ_raw)` is the actor flipping between two saturated
constants across states — bang-bang with a coarse / wrong sign mapping — not a refined
state-conditioned policy. The Q-surface itself is uniform to within a few units across very
different probe states (~−145 to −165 across up/down/sides/spinning).

**The single-step bootstrap on a 200-step horizon is biasing the Q-surface.** The Q-critic learns
a Q that supports a "max torque, sign depends on state" policy that scores best in single-step
look-ahead but is poor over the actual 200-step return. The actor commits hard and accurately to
`argmax_a Q(s,·)` of that biased Q — which is exactly the "commitment" the v2 lever is supposed to
produce, but it's commitment to the wrong actions.

## 4. Conclusion — §2b stand down

Per work order v3 §2b: *"If the Pendulum `|∇_a Q|` vs `2α` and `var(μ_raw)` numbers match the
bandit's working regime, no further harness tuning is on the table. The remaining structural lever
— multi-step Q targets in the soft-Bellman backup — is internal to `sac_learn_step` and
`sac_bellman_target` in the library (R1 forbids editing pc-rl-continuos). Upstream will start a SBTDD
cycle for n-step (or λ-) returns in the SAC critic; v3 is the trigger."*

Pendulum's numbers don't just match the working bandit's regime — they **exceed** it on
magnitude (`|∇_a Q|`) and on state-variance (`var(μ_raw)`). This is the §2b scenario, decisively.
**Standing down.** Upstream owns the next move (n-step / λ-returns in the soft-Bellman target).

What the harness has produced as the trigger evidence:
- 5 work-order-prescribed configurations exhausted with the same deterministic plateau
  (`results_v6_sac_b10.md` §13/§14, `results_v6_sac_b10_lr_fix.md`, `results_v6_sac_b10_v2.md`,
  this doc).
- The stochastic policy SOLVES (s44 `best_train = −4`); the deterministic actor commits to a
  saturated bang-bang on a biased Q-surface (this doc).
- The single-step diagnostic shows the library SAC works correctly on a single-step task; the
  v3 probe diagnostic confirms Pendulum is NOT a saturation/magnitude/commitment failure.
- `sac_skipped_*` = 0 across all runs (numerically sound).

The only remaining structural difference between the bandit (which converges) and Pendulum
(which doesn't) is the **200-step horizon × single-step Q bootstrap**. That is now the upstream
SBTDD target.

## 5. Reproduction

```
# Standard config.toml is v2-corrected (lr 3e-3, learning_starts 5000, target_entropy -2.0):
cargo run --release --bin multi_seed -- --config config_sac_warmup_te2.toml --seeds 42,43,44 --episodes 300
# Per eval: results/<out>/sac_diag_seed{S}.csv has columns
#   episode,probe_idx,sx,sy,sz,alpha,mu_det,q_min_mu,q_min_0,grad_mu,grad_0
# 8 probe states defined in src/probe_diag.rs::pendulum_probe_states.
# Compare grad_mu vs 2*alpha and atten=grad_mu/grad_0 to the upstream bandit's working regime
# (COMMIT_LOW: grad ~0.06, atten ~0.06, var(mu_raw) ~1.85).
```
pc-rl-continuos v6.0.0 `536bad1` (no library code change — R1). Deterministic under fixed seeds.

## 6. Ladder

| Run | Result | Key signal |
|---|---|---|
| v6 lr=3e-4 | rigid σ ~0.8, μ random | lr 10× too low (Adam vs SGD) |
| v6 lr=3e-3 | σ transient, μ saturates | lr fix necessary, horizon caveat triggered |
| v6 + warmup + te=−2 | stochastic SOLVES (s44 −4), μ saturated constant | actor commits, but to the wrong action |
| **v6 + warmup + te=−2 + v3 probe diag** | **`grad_μ` 10× bandit, atten ≈ 1, var(μ_raw) 4–5; eval −1278** | **§2b: not saturation/magnitude → upstream n-step** |
