# PC inference dynamics: free-energy minimisation and amortised inference under RL

This note documents how the predictive-coding (PC) actor settles its inference
loop, how the RL weight update relates to that settling, and an empirical study
of the claim *"the more trained the weights, the fewer cycles needed to reduce
free energy."*

The reproducing diagnostic is the ignored test
`pc_actor_critic::sac_learning_guards::probe_pc_inference_steps_vs_training`
(run with `cargo test --release --lib probe_pc_inference_steps_vs_training -- --ignored --nocapture`).

---

## 1. The actor "deliberates" by reducing free energy — confirmed

`PcActor::infer` (`src/pc_actor.rs`) runs an iterative loop. Each step:

1. **Top-down prediction** — `prediction = transpose_forward(state_above)`: the
   layer above generates an expectation of the layer below.
2. **Error** — `error = prediction − target` (mismatch with the current state).
3. **State update** — `h += alpha · error`: the hidden state moves toward the
   top-down prediction.
4. **Surprise** — `surprise_score = rms_error(errors)`: the scalar free-energy
   proxy across layers.
5. **Convergence** — stop when `surprise < tol` (after `min_steps`), else run to
   `max_steps`.

This is gradient-style minimisation of inter-layer prediction error (free
energy) over the **states** (not the weights). `InferResult` reports `y_conv`
(converged output), `steps_used`, `surprise_score` and `converged`.

## 2. RL updates the weights on the *stable* state — confirmed

The agent uses an E-step → M-step structure:

- `step_continuous` / `sac_actor_update` call `infer()` **first** (settle the
  dynamics).
- The converged `InferResult` (`y_conv`, `hidden_states`, `prediction_errors`)
  is then handed to `apply_actor_update_and_bookkeeping → update_weights`.

So the RL weight adjustment operates on the **already-settled** representation —
the *equilibrium-snapshot backprop* strategy. The pathwise SAC gradient is not
differentiated through the inference iterations; it is applied to the fixed-point
snapshot. The output layer uses pure backprop; hidden layers blend backprop with
the PC error via `local_lambda` (`delta = λ·backprop + (1−λ)·pc_error`).

## 3. "More trained ⇒ fewer cycles" — conditionally true (empirical)

We measured the average number of inference cycles (`steps_used`) over 25 fixed
probe states **before vs after** 20 000 online RL steps (seed 7), across five
`(max_steps, tol, local_lambda)` regimes:

| Regime | max_steps | tol | λ | before (steps / surp / conv) | after (steps / surp / conv) | Δsteps | Verdict |
|--------|-----------|-----|---|------------------------------|-----------------------------|--------|---------|
| R1 (canonical helper) | 20 | 0.01 | 1.0 | 20.00 / 0.062 / 0% | 20.00 / 0.079 / 0% | **+0.00** | never converges (ceiling); free energy ↑ |
| R2 | 80 | 0.01 | 1.0 | 80.00 / 0.030 / 0% | 78.92 / 0.024 / 12% | −1.08 | mostly at ceiling |
| **R3** | **80** | **0.05** | **1.0** | 34.08 / 0.049 / **100%** | 30.44 / 0.049 / **100%** | **−3.64 (−11%)** | ✅ **claim holds** |
| R4 | 80 | 0.05 | **0.9** | 34.08 / 0.049 / 100% | 80.00 / 0.089 / 0% | **+45.92** | ✗ destabilises |
| R5 | 200 | 0.02 | **0.9** | 113.84 / 0.020 / 96% | 187.80 / 0.020 / 100% | **+73.96** | ✗ destabilises |

### Reading

- **The intuition is the canonical predictive-coding *amortised inference*
  property, and it is real here — but regime-dependent.** It manifests cleanly
  only in **R3**, where convergence is reachable (`conv = 100%` both before and
  after, so `steps_used` reflects the true cycles-to-converge) and the weights
  are trained by **pure backprop (`λ = 1.0`)**. There, training cuts the cycle
  count by **−11%** (34 → 30). Mechanism: as the policy weights learn structure,
  the feed-forward pass initialises the hidden state closer to the inference
  fixed point, so fewer refinement cycles are needed. This is **emergent**, not
  enforced — there is no explicit free-energy term in the RL loss.

- **The canonical helper config (R1: `max_steps = 20`, `tol = 0.01`) is *not* in
  that regime.** The loop never reaches `tol` (always 20 steps, `conv = 0%`), so
  the effect cannot appear — and under pure RL pressure the free energy actually
  **drifts up** (0.062 → 0.079): the RL gradient optimises the policy output,
  which can worsen forward/top-down consistency.

- **Counterintuitively, injecting the PC error into the weight update
  (`λ < 1`, R4/R5) does not help — it destabilises the inference** (cycles
  explode, convergence collapses). The PC-error term is blended with the SAC
  pathwise gradient, which has a different scale and direction; at `λ = 0.9`
  the mix pushes the weights into a less-contractive coupling and the loop stops
  converging.

### Practical implication

To exploit the "fewer cycles with training" property, the config must

1. allow real convergence — raise `max_steps` and/or loosen `tol` so the loop
   reaches `surprise < tol` (the shipped `max_steps = 20 / tol = 0.01` runs the
   loop to its ceiling every step without converging, wasting compute), and
2. keep `local_lambda = 1.0` (pure backprop); blending the PC error into the SAC
   weight update is counterproductive in this setting.

## Caveats

- This is the **minimal in-library scale**: one hidden layer of 18 units, a
  9-dimensional input, and a toy reward (penalise `|a₀|`). The downstream
  PC-Pendulum harness may use a different actor topology / parameters, so the
  production regime can differ. The **qualitative law** (holds ⟺ convergent +
  `λ = 1.0`) is general; the magnitudes are toy-scale.
- The diagnostic is deterministic (fixed seed 7, deterministic probes) so the
  table is reproducible; only structural invariants are asserted in the test —
  the amortisation *magnitude* is not, as it is seed/scale dependent.
