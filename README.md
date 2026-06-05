# PC-RL-Continuos

[![CI](https://github.com/BolivarTech/PC-RL-Continuos/actions/workflows/ci.yml/badge.svg)](https://github.com/BolivarTech/PC-RL-Continuos/actions/workflows/ci.yml)
[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange.svg)](https://www.rust-lang.org)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE-MIT)

**Continuous-action** Deliberative Predictive Coding (DPC) reinforcement learning, implemented entirely in Rust with **zero ML-framework dependencies**.

The policy is a **predictive-coding actor** that *deliberates before acting* — it runs an iterative top-down/bottom-up free-energy-minimization loop instead of a single feedforward pass. On continuous action spaces this actor is trained by **canonical Soft Actor-Critic (SAC)**: a reparameterized squashed-Gaussian policy driven by a twin action-value critic, with automatic entropy temperature and off-policy replay.

> **Lineage.** This project is the continuous-control line of the DPC architecture. It is built on, and shares its foundation with, the discrete framework [**PC-RL-Core**](https://github.com/BolivarTech/PC-RL-Core) (Tic-Tac-Toe-validated, REINFORCE + V-critic). PC-RL-Continuos keeps the predictive-coding actor but replaces the discrete on-policy machinery with off-policy SAC for deterministic continuous-policy convergence.

The library is **backend-agnostic**: all linear algebra is abstracted behind a `LinAlg` trait, enabling future GPU backends (CUDA/wgpu) without touching the RL logic.

## Why SAC on a predictive-coding actor

On continuous control, a deterministic policy `μ` must converge onto the optimum the exploration discovers. A score-function (REINFORCE) gradient is **degenerate at the saturated `tanh` squash boundary** where many control optima live (max torque, max force): `tanh` is flat there, so changing `μ_raw` does not change the action — the mean is decoupled from the advantage signal and never converges.

SAC's **pathwise / reparameterization gradient** solves this:

```
∇_μ J = ∇_a Q(s, a) · ∂a/∂μ_raw
```

The action-value critic `Q(s, a)` supplies a low-variance *directional* signal that pins the mean to the value optimum — exactly what the score-function cannot. This requires the full canonical SAC recipe: twin Q critics, Polyak target networks, a replay buffer, a reparameterized squashed-Gaussian actor with learned per-state σ, and automatic entropy temperature.

## Installation

```toml
[dependencies]
pc-rl-core = "6.0"
```

## Quick Start (continuous SAC)

```rust
use pc_rl_core::{
    CpuLinAlg, PcActorCritic, PcActorCriticConfig, PcActorConfig,
    QCriticConfig, Activation, LayerDef, SelectionMode,
};
use pc_rl_core::pc_actor_critic::ActionSpace;

// --- Actor: a predictive-coding network emitting [μ_raw | log_σ_raw] ---
// output_size = 2 * action_dim, output_activation = Linear (required for SAC).
let action_dim = 1; // e.g. Pendulum-v1 torque
let state_dim  = 3;

let actor = PcActorConfig {
    input_size: state_dim,
    output_size: 2 * action_dim,          // μ_raw and log_σ_raw heads
    hidden_layers: vec![LayerDef { size: 64, activation: Activation::Softsign }],
    output_activation: Activation::Linear, // MUST be Linear in SAC mode
    alpha: 0.03,
    tol: 0.01,
    min_steps: 1,
    max_steps: 5,
    lr_weights: 3e-4,
    synchronous: true,
    temperature: 1.0,
    local_lambda: 1.0,
    residual: false,
    rezero_init: 0.001,
};

// --- Twin Q action-value critics Q(s, a) → scalar ---
let q_critic = QCriticConfig {
    state_dim,
    action_dim,
    hidden_layers: vec![LayerDef { size: 64, activation: Activation::Tanh }],
    lr: 3e-4,
};

let mut config = PcActorCriticConfig {
    action_space: ActionSpace::Continuous, // ⇒ canonical SAC
    q_critic: Some(q_critic),              // required when continuous
    polyak_tau: 0.005,                     // soft target-network update rate
    target_entropy: None,                  // None ⇒ −action_dim (standard SAC)
    log_alpha_init: 0.0,                   // α₀ = exp(0) = 1.0
    alpha_lr: 3e-4,                        // temperature learning rate
    replay_training_capacity: 100_000,     // off-policy replay buffer (required > 0)
    replay_batch_size: 256,
    gamma: 0.99,
    ..Default::default()
};

let backend = CpuLinAlg::new();
let mut agent = PcActorCritic::new(backend, config, 42)?;

// --- Training loop: one off-policy SAC step per environment step ---
let mut state = env.reset();
loop {
    // Samples a ~ tanh(μ_raw + σ·ε), stores the transition, runs a SAC
    // mini-batch update (twin-Q soft Bellman + pathwise actor + temperature),
    // and Polyak-updates the target nets.
    let action = agent.step_continuous(&state, reward, done)?;
    let (next_state, reward, done) = env.apply(&action);
    state = next_state;
    if done { state = env.reset(); }
}

// --- Deterministic evaluation: Play returns tanh(μ_raw), no noise ---
let (action, _infer) = agent.act_continuous(&state, SelectionMode::Play)?;
# Ok::<(), pc_rl_core::PcError>(())
```

## Canonical SAC — design

| Component | Behavior |
|---|---|
| **Reparameterized squashed-Gaussian actor** | The PC actor's converged output `y_conv` (size `2·action_dim`) splits into `μ_raw` and `log_σ_raw` (clamped to `[−5, 2]`). σ is **learned per state**. Sample `a = tanh(μ_raw + σ·ε)`, `ε ~ N(0, I)`. Play returns `tanh(μ_raw)`. |
| **Twin Q critics (`QCritic`)** | Two independent `Q(s, a)` networks. `min(Q1, Q2)` is used for both the actor update and the Bellman target (clipped double-Q, mitigates overestimation). |
| **`∇_a Q` (backprop-to-input)** | New capability: the gradient of the scalar Q output w.r.t. the action input — the directional signal that drives the pathwise actor gradient. |
| **Target networks (Polyak)** | `Q1ₜ, Q2ₜ` soft-updated each step: `θₜ ← (1−τ)·θₜ + τ·θ`, `τ = polyak_tau`. Used only for the soft-Bellman target. |
| **Off-policy replay** | Uniform buffer of `(s, a_raw, r, s', done)` transitions. Each step samples a `replay_batch_size` mini-batch. `replay_training_capacity > 0` is **required** in SAC mode. |
| **Automatic temperature α** | A learned scalar `log_α` tuned toward `H_target` (default `−action_dim`): `J(α) = −α·(logπ + H_target)`. Configurable via `log_alpha_init`, `alpha_lr`, `target_entropy`. |
| **log-prob with tanh-Jacobian** | `logπ(a\|s) = log N(a_raw; μ, σ²) − Σ log(1 − tanh²(a_raw))` — the squashed-Gaussian correction, generalized to learned σ. |

**Per-step update order:** collect transition → twin-Q soft-Bellman (MSE) → pathwise actor (`α·logπ − min Q`) → temperature → Polyak soft target update.

## Architecture

### Core components

- **`PcActor<L: LinAlg>`** — policy network with the predictive-coding inference loop (top-down prediction / bottom-up error until convergence), residual skip connections, and surprise scoring. Emits `[μ_raw | log_σ_raw]` in SAC mode.
- **`QCritic<L: LinAlg>`** — action-value critic `Q(s, a): (state ⊕ action) → scalar`. Exposes `forward`, `update` (MSE), and `∇_a Q` via backprop-to-input. Twin instances form the clipped double-Q.
- **`PcActorCritic<L: LinAlg>`** — integrated agent. Continuous mode (`ActionSpace::Continuous`) runs canonical SAC; `step_continuous` / `act_continuous` are the entry points.
- **`Layer<L: LinAlg>`** — dense layer with forward, transpose-forward (PC top-down), backward, and `input_gradient` (backprop-to-input).
- **`LinAlg` trait** — backend-agnostic linear algebra (31 instance methods). Default: `CpuLinAlg`.

### Predictive-coding inference

Instead of a single feedforward pass, the actor runs an iterative loop where higher layers generate top-down predictions of lower-layer states. The inter-layer prediction error (*surprise*) drives hidden-state updates until convergence (`alpha`, `tol`, `max_steps`). The converged output is the policy's `(μ_raw, log_σ_raw)`. The PC actor is **never** replaced by a feedforward MLP — SAC adapts to it.

### Key SAC config surface

| Field | Meaning |
|---|---|
| `action_space: ActionSpace::Continuous` | Selects the SAC path. |
| `q_critic: Option<QCriticConfig>` | Twin-Q critic topology (`state_dim`, `action_dim`, hidden layers, lr). Required `Some` when continuous. |
| `polyak_tau: f64` | Soft target-network update rate, `(0, 1]`. |
| `target_entropy: Option<f64>` | `H_target`; `None` ⇒ `−action_dim`. |
| `log_alpha_init: f64` | Initial log-temperature (`α₀ = exp(log_alpha_init)`). |
| `alpha_lr: f64` | Temperature learning rate. |
| `replay_training_capacity: usize` | Replay buffer size (must be `> 0`). |
| `replay_batch_size: usize` | SAC mini-batch size. |
| `gamma: f64` | Discount factor. |

### Type aliases

```rust
type PcActorCpu       = PcActor<CpuLinAlg>;
type QCriticCpu       = QCritic<CpuLinAlg>;
type PcActorCriticCpu = PcActorCritic<CpuLinAlg>;
type LayerCpu         = Layer<CpuLinAlg>;
```

## Project structure

```
PC-RL-Continuos/
├── src/
│   ├── linalg/
│   │   ├── mod.rs                  # LinAlg trait (backend-agnostic)
│   │   ├── cpu.rs                  # CpuLinAlg (Vec<f64> + Matrix)
│   │   └── golub_kahan.rs          # Golub-Kahan SVD (O(n^3))
│   ├── activation.rs               # Tanh, ReLU, Sigmoid, ELU, Softsign, Linear
│   ├── error.rs                    # PcError crate-wide error type
│   ├── matrix.rs                   # Dense matrix, softmax, clipping helpers
│   ├── layer.rs                    # Layer<L> with PC top-down + input_gradient
│   ├── pc_actor.rs                 # PcActor<L> inference loop (policy network)
│   ├── q_critic.rs                 # QCritic<L> action-value critic + ∇_a Q
│   ├── mlp_critic.rs               # MlpCritic<L> (discrete V-critic, untouched)
│   ├── pc_actor_critic/            # Integrated agent (directory submodule)
│   │   ├── mod.rs                  # act/step, step_continuous, act_continuous
│   │   ├── sac.rs                  # SAC learn step: twin-Q, pathwise actor, α, Polyak
│   │   ├── replay.rs               # Off-policy replay buffer + continuous transitions
│   │   ├── config.rs               # PcActorCriticConfig + ActionSpace + serde defaults
│   │   ├── control.rs              # Plasticity / hysteresis control surface
│   │   ├── ewma.rs                 # EwmaTracker + PlasticityState
│   │   ├── hysteresis.rs           # Dual-EWMA FROZEN/PLASTIC state machine
│   │   ├── fisher.rs               # FisherState<L> for EWC regularization
│   │   └── trajectory.rs           # TrajectoryStep<L> + ActivationCache<L>
│   └── serializer.rs               # JSON persistence (Q nets, target nets, log_α)
├── docs/
│   ├── pc_actor_critic_paper.md    # DPC architecture paper
│   ├── pc_inference_intuitive_guide.md
│   ├── experiment_pendulum_v1_spec.md
│   └── continuous_space_test_harness_guide.md
└── Cargo.toml
```

## Validation

Deterministic continuous-policy convergence is validated **downstream** by the
PC-Pendulum harness on **Pendulum-v1** (`multi_seed` 10×500): the deterministic
evaluation mean clears ≈ −500 with ≥ 5/10 seeds > −400. In-library tests are
directional/mechanism guards (pathwise gradient drives `μ_raw` toward saturated
optima; `μ_raw` stays bounded; log-prob Jacobian; auto-temperature sign), not a
convergence proof.

## Documentation

| Document | Content |
|---|---|
| [docs/pc_actor_critic_paper.md](docs/pc_actor_critic_paper.md) | Formal DPC architecture spec and mathematical justification |
| [docs/pc_inference_intuitive_guide.md](docs/pc_inference_intuitive_guide.md) | Conversational walkthrough of PC inference + learning |
| [docs/experiment_pendulum_v1_spec.md](docs/experiment_pendulum_v1_spec.md) | Spec for the PC-Pendulum validation harness (continuous, Pendulum-v1) |
| [docs/continuous_space_test_harness_guide.md](docs/continuous_space_test_harness_guide.md) | Configuring and running the continuous harness |
| [CHANGELOG.md](CHANGELOG.md) | Per-release changes and migration notes |

## Dependencies

- `serde` / `serde_json` — serialization
- `rand` — random number generation
- `chrono` — timestamps

No PyTorch, TensorFlow, or any ML framework. Pure Rust from scratch.

## Testing

```bash
cargo nextest run                          # fast suite
cargo nextest run -- --ignored             # slow SAC learning guards
cargo test --doc                           # doctests
cargo clippy --all-targets -- -D warnings  # lint
```

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT License ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.
