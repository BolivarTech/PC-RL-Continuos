# Continuous Action Space — Test Harness Handoff Guide (v4.1.0)

**Audience:** The agent/developer who will build the **downstream consumer
binary** (e.g. `PC-Pendulum`) that validates `pc-rl-continuos` v4.1.0's continuous
action mode against real dynamics.

**Purpose:** Give that agent the *exact, code-verified* public API contract of
the continuous mode so it can program the test harness without guessing. The
two experiment specs describe *what* to run; this guide pins down *how to call
the library*.

**Source of truth:** All signatures, field names, defaults, and validation
rules below were read directly from the v4.1.0 source (`src/lib.rs`,
`src/pc_actor_critic/{mod.rs,config.rs}`, `src/serializer.rs`). If anything here
disagrees with `docs/experiment_pendulum_v1_spec.md`, **this guide wins** —
the spec predates the final API and contains a few simplifications that won't
compile (noted in §7).

---

## 1. What is the inverted-pendulum experiment?

The **Pendulum-v1 swing-up** task: a single rigid rod hangs from a fixed pivot
under gravity. The agent applies a continuous **torque** to the pivot each step.
Starting from a random angle (typically hanging down), it must **swing the rod
up and balance it upright**, then hold it there.

- It is the canonical "hello world" of *continuous control* — the continuous
  analogue of CartPole. There is no win/lose; the agent just accumulates a
  per-step **cost** (negative reward) and tries to minimize it.
- **State** `[cos θ, sin θ, θ̇]` (3-D). **Action** scalar torque `u ∈ [−2, 2]`
  N·m (1-D). **Reward** `−(θ² + 0.1·θ̇² + 0.001·u²)` (≤ 0; best per-step = 0 at
  upright, still, zero torque).
- Episodes are a fixed 200 steps, no termination.
- **Why this experiment matters:** v4.1.0 ships the continuous Gaussian-policy
  *code* and passes synthetic gradient tests, but it has **never been run
  against real dynamics**. Pendulum-v1 is the lowest-friction proof that the
  policy gradient `δ = td_error · (μ − a)/σ²` actually trains a working policy.
  If it can't solve Pendulum, there are bugs synthetic tests didn't catch.

Full physics, reward, episode structure, phases, success criteria, and a ~30-LOC
reference `Pendulum` struct are in **`docs/experiment_pendulum_v1_spec.md`**.
A 4-D action variant (cart force + reach) is in
**`docs/experiment_cartpole_continuous_spec.md`**. Read those for experiment
design; read *this* guide for the API.

---

## 2. Dependency

```toml
# Consumer Cargo.toml
[dependencies]
pc-rl-continuos = "4"          # v4.0.0+
rand = "0.8"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
toml = "0.8"              # optional: if you load config from TOML
chrono = "0.4"           # optional: metrics timestamps
```

`pc-rl-continuos` has no ML-framework dependencies and is CPU-only in v4.0.0
(`CpuLinAlg`). A GPU backend is planned but not shipped — do not depend on it.

---

## 3. Public API surface you will use

Everything is re-exported from the crate root (`pc_rl_continuos::…`).

```rust
use pc_rl_continuos::{
    PcActorCritic, PcActorCriticConfig, ActionSpace, CpuLinAlg,
    SelectionMode,                       // re-exported from pc_actor
    PcActorConfig, MlpCriticConfig, LayerDef, Activation,
    save_agent, load_agent, save_checkpoint,
};
```

### 3.1 Construction — `PcActorCritic::new`

```rust
pub fn new(backend: L, config: PcActorCriticConfig, seed: u64)
    -> Result<Self, PcError>
```

- `backend`: pass `CpuLinAlg::new()`.
- `seed`: seeds the agent's internal `StdRng`. **This is the only place a seed
  goes** — action sampling (Box-Muller) and weight init draw from it.
  Same seed + same call sequence ⇒ identical trajectory (reproducibility).
- Returns `Err(PcError::ConfigValidation(..))` if the config violates any rule
  in §5. **Always handle the `Result`** — do not `.unwrap()` in the harness body
  except in throwaway smoke code.

```rust
let agent = PcActorCritic::new(CpuLinAlg::new(), config, 42)?;
```

### 3.2 Training step — `step_continuous`

```rust
pub fn step_continuous(&mut self, state: &[f64], reward: f64, done: bool)
    -> Result<Vec<f64>, PcError>
```

This is the **main training entry point**. One call per environment step. It:

1. Runs PC inference on `state` to get the policy mean `μ(s)`.
2. **If a previous transition is buffered**, performs a TD(0) update using the
   Gaussian-policy gradient `δ_j = td_error · (μ_j − a_j)/σ²`, plus
   surprise-scaled LR, optional hysteresis, etc.
3. **Samples** and returns the action `a = μ + σ·ε`, `ε ~ N(0, I)` via
   Box-Muller from the internal RNG. `σ = config.policy_sigma`.
4. Stashes `(state, action, infer)` so the *next* call closes the TD bootstrap.
5. On `done == true`, clears all transient episode state.

**Calling convention (canonical RL timing):** pass the *current* `state` and the
reward earned from the *previous* action. On the first step of an episode pass
`reward = 0.0`.

```rust
let action = agent.step_continuous(&state, last_reward, done)?;
```

> **The returned action is tanh-squashed to `[−1, 1]`.** Internally the actor
> outputs an unbounded mean `μ_raw` (`output_activation` must be `Linear`),
> samples `a_raw = μ_raw + σ·ε`, then returns `tanh(a_raw)` — bounded by
> construction. **No clamp is needed.** Map affinely to your physical range
> (see §6).

There is also `step_continuous_raw_device(&mut self, …) -> Result<L::Vector, …>`
— identical flow but returns the backend-native vector. On `CpuLinAlg`,
`L::Vector == Vec<f64>`, so it's equivalent. It exists only as a forward-compat
hook for a future GPU backend. **Use `step_continuous` for the harness.**

### 3.3 Inference only (evaluation) — `act_continuous`

```rust
pub fn act_continuous(&mut self, state: &[f64], mode: SelectionMode)
    -> Result<(Vec<f64>, InferResult<L>), PcError>
```

No learning, no stored-state mutation. Use it for evaluation rollouts.

| `mode` | Returns | RNG advanced? |
|---|---|---|
| `SelectionMode::Play` | deterministic `μ(s)` (no noise) | No |
| `SelectionMode::Training` | `μ + σ·ε` (stochastic) | Yes (1 draw/dim) |

```rust
let (action, infer) = agent.act_continuous(&state, SelectionMode::Play)?;
// infer.surprise_score, infer.y_conv, etc. available for metrics.
```

> Note `Play` mode returns `tanh(μ_raw)` — already in `[−1, 1]`. Apply the
> same affine map from §6 in evaluation as in training; no clamp needed.

### 3.4 Persistence — serializer free functions

```rust
pub fn save_agent<L>(agent: &PcActorCritic<L>, path: &str,
                     episode: usize, metrics: Option<TrainingMetrics>)
    -> Result<(), PcError>

pub fn save_checkpoint<L>(agent: &PcActorCritic<L>, dir: &str,
                          episode: usize, metrics: Option<TrainingMetrics>)
    -> Result<PathBuf, PcError>   // auto-names checkpoint_ep{N}_{timestamp}.json

pub fn load_agent(path: &str, backend: CpuLinAlg)
    -> Result<(PcActorCritic, AgentMetadata), PcError>
```

```rust
save_agent(&agent, "checkpoints/ep_0050.json", 50, None)?;
let (agent, meta) = load_agent("checkpoints/ep_0050.json", CpuLinAlg::new())?;
```

JSON format. `action_space` and `policy_sigma` are serialized, so a loaded agent
stays in continuous mode.

---

## 4. Building the config (two valid paths)

`PcActorCriticConfig` has ~50 fields. **It does NOT derive `Default`**, so you
**cannot** write `PcActorCriticConfig { actor, critic, ..Default::default() }`.
Pick one of these:

### Path A (recommended) — deserialize from TOML/JSON, let serde fill defaults

Every field *except* `actor` and `critic` has a `#[serde(default = …)]`. So a
config file only needs `actor`, `critic`, and the few continuous knobs you want
to override; serde supplies the rest.

```toml
# config.toml
[actor]
input_size = 3
output_size = 1
output_activation = "linear"   # MUST be linear — library squashes internally via tanh
alpha = 0.03
tol = 0.01
min_steps = 1
max_steps = 5
lr_weights = 0.005
synchronous = true
temperature = 1.0
local_lambda = 0.99
residual = false
rezero_init = 0.001
hidden_layers = [{ size = 32, activation = "tanh" }]

[critic]
input_size = 35            # 3 state + 32 latent_concat (= sum of actor hidden sizes)
output_activation = "linear"
lr = 0.005
hidden_layers = [{ size = 64, activation = "tanh" }]

# --- continuous mode knobs ---
action_space = "Continuous"
policy_sigma = 0.3
gae_lambda = 0.95          # GAE eligibility trace for multi-step credit assignment
policy_entropy_coeff = 0.1 # v4.2.0 entropy temperature α (default-on); bounds μ_raw at the
                           # squash boundary so the DETERMINISTIC policy converges. 0.0 = v4.1.0
                           # behavior. Read per-step + runtime-mutable like policy_sigma — anneal
                           # it caller-side over training (lower α as the policy commits).

# --- override any other default as needed ---
gamma = 0.99
adaptive_surprise = true
surprise_buffer_size = 400
entropy_coeff = 0.0
scale_floor = 0.1
scale_ceil = 2.0
```

```rust
let cfg: PcActorCriticConfig =
    toml::from_str(&std::fs::read_to_string("config.toml")?)?;
```

> **`critic.input_size` must equal `actor.input_size + Σ(actor hidden sizes)`.**
> The critic receives the raw state concatenated with every actor hidden-layer
> activation (`latent_concat`). For one 32-unit hidden layer on a 3-D state:
> `3 + 32 = 35`. **As of v4.0.1**, getting this wrong makes `new()` return
> `PcError::ConfigValidation` naming `critic.input_size`. **In v4.0.0 exactly**,
> the mismatch is accepted at construction and instead panics later in
> `MlpCritic::forward` on the first critic forward pass (second `step_continuous`
> call) — pin v4.0.1+ to get the eager, recoverable error.

### Path B — full struct literal in Rust

Specify **all** fields. A complete, compiling literal is in the
`PcActorCriticConfig` doc example (`src/pc_actor_critic/config.rs`, the
`/// # Examples` block) and in the `base_config()` test fixture in the same
file — copy one of those and flip `action_space` / `policy_sigma`.

---

## 5. Continuous-mode validation rules (enforced in `new()`)

When `action_space == ActionSpace::Continuous`, `new()` returns
`Err(PcError::ConfigValidation)` unless **all** of these hold:

| Field | Rule in continuous mode | Reason |
|---|---|---|
| `policy_sigma` | **must be `> 0.0` and finite** | it's the Gaussian σ; `/σ²` in the gradient |
| `policy_entropy_coeff` | **must be `>= 0.0` and finite** | v4.2.0 entropy temperature α; bounds `μ_raw` (default-on `0.1`; `0.0` = v4.1.0). Distinct from discrete `entropy_coeff`. A constant α is an always-on restoring force that continuously biases `μ_raw` toward zero — callers SHOULD anneal α downward over training (start higher, decay toward a small floor as the policy commits) to let the mean escape to its optimal value. The field is read per-step and runtime-mutable like `policy_sigma`, enabling caller-side annealing. |
| `distillation_lambda_polyak` | **must be `0.0`** | KL distillation undefined for raw continuous output |
| `distillation_lambda_frozen` | **must be `0.0`** | same reason |
| `gae_lambda` | `None` (TD(0)) **or** `Some(λ)` where `0 < λ < 1` | GAE eligibility trace supported in v4.1.0; `Some(0.95)` recommended |
| `td_steps` | **must be `0`** | continuous TD(n) not implemented |
| `replay_training_capacity` | **must be `0`** | `replay_learn` rejects continuous transitions → buffer would be write-only |
| `replay_recent_capacity` | **must be `0`** | same reason |
| `entropy_coeff` | any value **allowed but inert** | this is the DISCRETE (softmax) entropy coeff — inert in continuous. For continuous entropy use `policy_entropy_coeff` (v4.2.0) |

Continuous mode uses a **constant learning rate** — the surprise→LR modulation
(M1) is bypassed for continuous policy learning. No replay, no distillation, no
TD(n). The two experiment specs already set all of the above correctly.

---

## 6. Action scaling (affine map — no clamp needed)

`step_continuous` and `act_continuous` return tanh-squashed actions in
`[−1, 1]`. Map affinely to the environment's physical range. For Pendulum
(`output_size = 1`, torque range `[−2, 2]`):

```rust
let action = agent.step_continuous(&state, last_reward, done)?;   // tanh(a_raw) ∈ [-1,1]
let torque = action[0] * 2.0;                                      // [-1,1] → [-2,2]
let (next_state, reward, done) = env.step(torque);
```

No clamp is required — tanh already bounds the output. For an N-D action
space, scale each component to its own range with `action[i] * range_i`.
Apply the **same** mapping in `act_continuous` evaluation.

---

## 7. Corrections vs. `experiment_pendulum_v1_spec.md`

The pendulum spec is correct on experiment design but its Rust snippets predate
the final API. When you implement, apply these fixes:

1. **No `..Default::default()` / no `// ... rest of defaults`.** The config has
   no `Default` impl — use Path A (TOML+serde) or a full literal (§4).
2. **Construction.** Use `PcActorCritic::new(CpuLinAlg::new(), cfg, seed)?`. The
   seed is the 3rd arg of `new`, not a separate setter.
3. **`step_continuous` and `act_continuous` return tanh-squashed actions in
   `[−1, 1]`** — no clamp needed. Use `Activation::Linear` for the actor
   `output_activation` (NOT Tanh). Map affinely to the physical range (§6).
4. **`critic.input_size = actor.input_size + Σ hidden sizes`** (the spec's
   `3 + 32` is right *because* its actor has one 32-unit layer; recompute if you
   change topology).
5. Field/type names in §3–§5 of *this* guide are the authoritative spellings
   (`ActionSpace::Continuous`, `SelectionMode::Play`/`Training`, `LayerDef`,
   `Activation::Linear`, etc.).

---

## 8. Minimal smoke test to program first

Before the full training loop, get this compiling and passing — it exercises the
entire continuous integration surface in ~20 lines and catches API/dimension
mistakes immediately:

```rust
#[test]
fn continuous_smoke() {
    use pc_rl_continuos::*;

    // Build a tiny continuous config (Path B literal, or load a TOML).
    let cfg: PcActorCriticConfig =
        toml::from_str(include_str!("../config.toml")).unwrap();
    assert_eq!(cfg.action_space, ActionSpace::Continuous);

    let mut agent = PcActorCritic::new(CpuLinAlg::new(), cfg, 42).unwrap();

    // One episode of random states; verify finite, in-range-after-clamp actions.
    let mut last_reward = 0.0;
    for step in 0..200 {
        let state = vec![0.5_f64.cos(), 0.5_f64.sin(), 0.1]; // dummy 3-D obs
        let done = step == 199;
        let action = agent.step_continuous(&state, last_reward, done).unwrap();
        assert_eq!(action.len(), 1);
        assert!(action[0].is_finite(), "action must be finite");
        assert!((-1.0..=1.0).contains(&action[0]), "tanh-squashed action must be in [-1,1]");
        let torque = action[0] * 2.0;  // affine map to [-2,2]; no clamp needed
        assert!((-2.0..=2.0).contains(&torque));
        last_reward = -0.1; // dummy
    }

    // Deterministic Play mode: same state twice → same action.
    let s = vec![1.0, 0.0, 0.0];
    let (a1, _) = agent.act_continuous(&s, SelectionMode::Play).unwrap();
    let (a2, _) = agent.act_continuous(&s, SelectionMode::Play).unwrap();
    assert_eq!(a1, a2, "Play mode must be deterministic");

    // Round-trip persistence.
    save_agent(&agent, "smoke.json", 0, None).unwrap();
    let _ = load_agent("smoke.json", CpuLinAlg::new()).unwrap();
}
```

If this passes, the API wiring is correct and you can build the real
`Pendulum` env + training loop per `experiment_pendulum_v1_spec.md`.

---

## 9. Success / failure signals (from the spec)

- **Works:** multi-seed mean reward (final 100 eps) ≥ 3× better than random
  (~−1500); ≥ 5/10 seeds reach > −400; no NaN/Inf/panics; reproducible under
  fixed seed; surprise score decreases over training.
- **Reveals a v4.1.0 bug (report back to this repo):** all seeds stuck at
  ~−1500 (gradient-sign/numerical issue); most seeds NaN; reward improves then
  catastrophically collapses. Include seed, `pc-rl-continuos` commit SHA, and the
  metrics CSV in the bug report.

---

## 10. GRAD_CLIP / σ saturation and the Pendulum PASS bar

The continuous-mode gradient update per output dimension is:

```
delta = td_error · (μ_raw − a_raw) / σ²  =  −td_error · ε / σ
```

With small `policy_sigma` (e.g. 0.1) and Pendulum-scale `td_error` values
(which can be in the tens early in training), the per-element delta can
routinely saturate `GRAD_CLIP = 5.0` — the global weight-update clip used
across the crate. When most updates are at the clip boundary, the effective
gradient direction is preserved but the magnitude is uniformly compressed,
slowing learning.

**If the in-library smoke tests pass but the Pendulum harness PASS bar (≥ 5/10
seeds reach mean reward > −400) is missed**, the saturation levers are (in
order of impact):

1. **Raise `policy_sigma`** — smaller `1/σ²` factor reduces delta magnitude
   directly. Try 0.3–0.5 before other changes.
2. **Lower `lr_weights`** — dampens the weight step post-clip.
3. **More episodes** — slower convergence is still convergence; budget
   300–500 episodes before concluding non-convergence.
4. **Center / normalize rewards before passing to `step_continuous`** — if
   `td_error` magnitudes are chronically large, dividing rewards by a running
   std-dev (e.g. over the last 100 episode sums) brings them into a range
   where GRAD_CLIP rarely bites. This is a harness-side normalization; the
   library itself does not normalize inputs.

**Merge gate:** the in-library unit and integration tests (716 tests, including
the continuous smoke and gradient-sign tests) are the merge gate for
`pc-rl-continuos`. The Pendulum harness result is the **external validation** check
and lives in a separate `PC-Pendulum` repository. Failing the Pendulum PASS bar
does not block a library release but should be investigated before the result is
cited as empirical evidence that the continuous mode works on real dynamics.

---

**TL;DR for the implementing agent:** add `pc-rl-continuos = "4"`; build a
`PcActorCriticConfig` with `action_space = "Continuous"`, `policy_sigma > 0`,
`output_activation = "linear"`, and `gae_lambda = 0.95` (TOML+serde is
easiest); `PcActorCritic::new(CpuLinAlg::new(), cfg, seed)?`; loop
`step_continuous(&state, prev_reward, done)?` and **scale affinely** (e.g.
`action[0] * 2.0`) — no clamp needed, the library returns tanh-squashed actions
in `[−1, 1]`; evaluate with `act_continuous(&state, SelectionMode::Play)?`;
checkpoint with `save_agent`/`load_agent`. Keep replay/TD(n)/distillation off.
Get §8 green first.
