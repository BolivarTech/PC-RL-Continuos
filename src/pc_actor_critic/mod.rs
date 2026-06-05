// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-03-25

//! Integrated PC Actor-Critic agent.
//!
//! Combines [`PcActor`] for action selection via predictive coding inference
//! with the twin [`QCritic`](crate::QCritic) action-value critics for canonical
//! Soft Actor-Critic (SAC) value estimation on continuous action spaces.
//!
//! Generic over a [`LinAlg`] backend `L`. Defaults to [`CpuLinAlg`].

use std::collections::VecDeque;

use rand::rngs::StdRng;

use crate::error::PcError;
use crate::linalg::cpu::CpuLinAlg;
use crate::linalg::LinAlg;
use crate::mlp_critic::{MlpCritic, MlpCriticConfig};
use crate::pc_actor::{InferResult, PcActor, PcActorConfig};

pub mod config;

pub use config::*;

pub mod ewma;

pub use ewma::*;

pub mod hysteresis;

pub use hysteresis::*;

pub mod fisher;

pub use fisher::*;

pub mod trajectory;

pub use trajectory::{ActivationCache, TrajectoryStep};

pub mod replay;

mod control;

mod sac;

/// Default cooldown (in learning steps) between consecutive `rollback_hard()` calls.
///
/// Prevents thrashing when the caller repeatedly reverts the actor to the
/// frozen champion. Set to 0 via [`PcActorCritic::set_rollback_hard_cooldown`]
/// to disable the cooldown entirely.
pub const DEFAULT_ROLLBACK_HARD_COOLDOWN: u64 = 100;

/// Minimum value for the actor's log standard deviation in learned-σ SAC mode
/// (v6.0.0). Clamps `log σ` from below so `σ` never collapses to zero.
pub const LOG_SIG_MIN: f64 = -5.0;

/// Maximum value for the actor's log standard deviation in learned-σ SAC mode
/// (v6.0.0). Clamps `log σ` from above so `σ` stays in a numerically safe range.
pub const LOG_SIG_MAX: f64 = 2.0;

/// Learning-path mode for [`PcActorCritic::learn_continuous_inner`].
///
/// Distinguishes on-policy updates (which must maintain GAE traces,
/// TD-error buffers, cooldown counters and the EWC Fisher estimate)
/// from replay-driven off-policy updates (which must NOT mutate
/// online-only state so the replay batch does not contaminate the
/// agent's view of its current trajectory).
///
/// Introduced in Phase 2 of the self-recovery plan to prepare the
/// internal learn path for a future `replay_learn` caller.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LearnMode {
    /// On-policy update: all bookkeeping side effects enabled.
    Online,
    /// Off-policy replay update: skip GAE trace update, td_error
    /// buffer push, Fisher lifecycle and cooldown counter increments.
    ///
    /// Constructed by [`PcActorCritic::replay_learn`] for each transition
    /// drawn from the replay buffer.
    Replay,
}

/// Integrated PC Actor-Critic agent.
///
/// Combines a predictive coding actor with the twin Q action-value critics
/// for canonical Soft Actor-Critic (SAC) on continuous action spaces.
///
/// Generic over a [`LinAlg`] backend `L`. Defaults to [`CpuLinAlg`].
#[derive(Debug)]
pub struct PcActorCritic<L: LinAlg = CpuLinAlg> {
    /// The PC actor network.
    pub(crate) actor: PcActor<L>,
    /// The MLP critic (value function).
    pub(crate) critic: MlpCritic<L>,
    /// Agent configuration.
    pub config: PcActorCriticConfig,
    /// Random number generator for action selection.
    rng: StdRng,
    /// Circular buffer of recent surprise scores for adaptive thresholds.
    surprise_buffer: VecDeque<f64>,
    /// Backend used for linear algebra operations.
    pub(crate) backend: L,
    /// Previous observation (transient, not serialized).
    state_prev: Option<L::Vector>,
    /// Previous discrete action taken (transient, not serialized).
    /// Mutually exclusive with [`Self::action_prev_continuous`]: only the
    /// field matching `config.action_space` is populated by the active
    /// step path.
    action_prev: Option<usize>,
    /// Previous continuous action vector taken (transient, not serialized).
    /// Holds the pre-squash `a_raw = μ_raw + σ·ε` from the prior
    /// `step_continuous` call so the next call can build the Gaussian-policy
    /// gradient `(μ − a_raw)/σ²`. The returned/executed action is
    /// `tanh(a_raw)`, but the gradient must use the unbounded `a_raw`.
    /// Mutually exclusive with [`Self::action_prev`].
    action_prev_continuous: Option<Vec<f64>>,
    /// Previous inference result (transient, not serialized).
    infer_prev: Option<InferResult<L>>,
    /// Previous valid actions mask (transient, not serialized).
    /// `None` = all actions valid (used by `step()`), `Some` = masked (used by `step_masked()`).
    valid_actions_prev: Option<Vec<usize>>,
    /// Actor hysteresis state machine (None when disabled).
    actor_hysteresis: Option<HysteresisState>,
    /// Critic hysteresis state machine (None when disabled).
    critic_hysteresis: Option<HysteresisState>,
    /// Steps the actor has been in PLASTIC state during the current phase.
    actor_plastic_step_counter: u64,
    /// Steps the critic has been in PLASTIC state during the current phase.
    critic_plastic_step_counter: u64,
    /// Consecutive steps the critic has been FROZEN.
    critic_frozen_steps: u64,
    /// Consecutive steps the actor has been FROZEN.
    actor_frozen_steps: u64,
    /// Circular buffer of recent |TD errors| for critic adaptive scale.
    td_error_buffer: VecDeque<f64>,
    /// Last TD error from learn_continuous (transient, for hysteresis).
    last_td_error: f64,
    /// Precomputed per-hidden-layer decay factors for actor (M3a).
    actor_decay_factors: Vec<f64>,
    /// Precomputed per-hidden-layer decay factors for critic (M3a).
    critic_decay_factors: Vec<f64>,
    /// Per-layer prediction error EMA for adaptive consolidation (M3b, actor only).
    layer_error_ema: Vec<f64>,
    /// Per-layer Fisher information state for actor EWC (empty when ewc_lambda=0).
    actor_fisher: Vec<FisherState<L>>,
    /// Per-layer Fisher information state for critic EWC (empty when ewc_lambda=0).
    critic_fisher: Vec<FisherState<L>>,
    /// Whether the last actor PLASTIC phase was reliable (>= min_fisher_phase steps).
    actor_last_phase_reliable: bool,
    /// Whether the last critic PLASTIC phase was reliable (>= min_fisher_phase steps).
    critic_last_phase_reliable: bool,
    /// Output-level eligibility trace for GAE(λ). Empty when gae_lambda=None.
    /// Not serialized — transient mid-episode state.
    actor_trace: Vec<f64>,
    /// Polyak-averaged target actor for KL distillation.
    /// `Some` when `distillation_lambda_polyak > 0`, `None` otherwise.
    /// Updated via soft Polyak averaging after each actor weight update.
    pub(crate) polyak_target: Option<PcActor<L>>,
    /// Frozen champion actor for KL distillation.
    /// `Some` when `distillation_lambda_frozen > 0`, `None` otherwise.
    /// Never updated automatically — stays byte-exact until explicit
    /// `champion_update()` call (Task 4).
    pub(crate) frozen_champion: Option<PcActor<L>>,
    /// Cooldown window (learning steps) between consecutive `rollback_hard()` calls.
    /// 0 disables the cooldown entirely.
    pub(crate) rollback_hard_cooldown_steps: u64,
    /// Steps since the last successful `rollback_hard()` call.
    /// Initialized to `u64::MAX` so the first call is always allowed.
    pub(crate) steps_since_last_rollback_hard: u64,
    /// Dual-compartment replay buffer (Phase 2). `None` when
    /// `replay_training_capacity == 0` at construction, otherwise a
    /// freshly-allocated empty buffer sized per config. Populated by
    /// auto-record in [`step`](Self::step) / [`step_masked`](Self::step_masked)
    /// and consumed by [`replay_learn`](Self::replay_learn).
    pub(crate) replay_buffer: Option<crate::pc_actor_critic::replay::ReplayBuffer>,
    /// Monotonic counter of `replay_learn` calls where the td_error
    /// clamp was binding (MAGI R5 W5). Exposed via
    /// [`PcActorCritic::replay_clamp_count`].
    pub(crate) replay_clamp_count: u64,
    /// Log-temperature for SAC automatic entropy tuning (v6.0.0).
    ///
    /// `α = exp(log_alpha)` is guaranteed `> 0`. Initialized from
    /// `config.log_alpha_init` in `new()` for SAC mode (else `0.0`).
    /// Updated by `sac_temperature_update()`. Serialized as
    /// `Option<f64>` in the save file (present for SAC agents, absent
    /// for discrete / pre-v6 files); restored directly on load, falling
    /// back to `config.log_alpha_init` when absent.
    pub(crate) log_alpha: f64,
    /// SAC twin Q-critics (v6.0.0). `Some` in continuous SAC mode
    /// (`action_space == Continuous && q_critic.is_some()`). `None`
    /// for discrete agents and pre-v6 continuous agents without
    /// `q_critic` config.
    pub(crate) q1: Option<crate::q_critic::QCritic<L>>,
    /// SAC twin Q-critic 2 (v6.0.0). See [`Self::q1`].
    pub(crate) q2: Option<crate::q_critic::QCritic<L>>,
    /// Polyak-averaged soft target copy of `q1` (v6.0.0). Updated via
    /// `polyak_update_targets()` after every critic update. `None`
    /// when `q1` is `None`.
    pub(crate) q1_target: Option<crate::q_critic::QCritic<L>>,
    /// Polyak-averaged soft target copy of `q2` (v6.0.0). See [`Self::q1_target`].
    pub(crate) q2_target: Option<crate::q_critic::QCritic<L>>,
    /// Monotonic counter of SAC critic update steps skipped due to non-finite
    /// intermediate values (non-finite actions, log-prob, Q-values, or Bellman
    /// target). Each skipped transition increments this by one; the counter
    /// never resets. Exposed for diagnostics.
    pub(crate) sac_skipped_critic_updates: u64,
    /// Monotonic counter of SAC actor update steps skipped due to non-finite
    /// delta values (non-finite Q-gradient, log-prob, or intermediate values).
    /// Each skipped transition increments this by one; the counter never resets.
    /// Exposed for diagnostics (T11/T12).
    pub(crate) sac_skipped_actor_updates: u64,
}

/// Numerical-stability epsilon for the tanh-Jacobian log term near the squash boundary.
///
/// Added to `(1 − tanh²(a_raw))` before taking the logarithm so the Jacobian
/// correction remains finite even when `|a_raw|` is very large (tanh ≈ ±1).
/// Must equal `1e-6` — pinned by [`test_squashed_log_prob_matches_reference`].
const SQUASH_JAC_EPS: f64 = 1e-6;

/// Log-probability of the tanh-squashed diagonal Gaussian policy at `a = tanh(a_raw)`,
/// with the tanh-Jacobian correction, for learned per-dimension `σ = exp(log_sigma)`.
///
/// ```text
/// logπ(a|s) = Σ_j [ logN(a_raw_j; μ_j, σ_j²) − log(1 − tanh²(a_raw_j) + ε_stab) ]
/// ```
///
/// where `logN(x; μ, σ²) = −0.5·((x−μ)/σ)² − log σ − 0.5·log(2π)`.
///
/// The `ε_stab = SQUASH_JAC_EPS` term prevents the Jacobian log from diverging to −∞
/// at the squash boundary, keeping `logπ` finite for any finite `a_raw`.
///
/// Consumed by the SAC critic soft-Bellman target and the automatic-temperature
/// update (T10/T11); added here as a pure free function so those tasks can call it
/// without owning the full `PcActorCritic` context.
///
/// # Arguments
///
/// * `mu_raw` — unbounded policy mean, one value per action component.
/// * `log_sigma` — log standard deviation (clamped to `[LOG_SIG_MIN, LOG_SIG_MAX]`
///   before this call), one value per action component.
/// * `a_raw` — pre-squash sample `μ_raw + σ·ε`, one value per action component.
///
/// # Returns
///
/// The scalar `logπ(a|s)` summed over all action components.
fn squashed_log_prob(mu_raw: &[f64], log_sigma: &[f64], a_raw: &[f64]) -> f64 {
    let half_log_2pi = 0.5 * (2.0 * std::f64::consts::PI).ln();
    let mut lp = 0.0;
    for j in 0..mu_raw.len() {
        let s = log_sigma[j]; // log σ
        let z = (a_raw[j] - mu_raw[j]) / s.exp();
        lp += -0.5 * z * z - s - half_log_2pi;
        let t = a_raw[j].tanh();
        lp -= (1.0 - t * t + SQUASH_JAC_EPS).ln();
    }
    lp
}

/// Reparameterized SAC actor DESCENT delta on `(mu_raw, log_sigma_raw)`,
/// of length `2 * action_dim`, minimising `L = α·logπ(a|s) − min(Q1,Q2)(s,a)`.
///
/// Under reparameterisation the Gaussian score terms cancel (ε is constant
/// in μ and log σ). The entropy gradient uses the ε_stab-consistent
/// tanh-Jacobian derivative so the formula is exact at saturation.
///
/// Descent direction per component `j` (n = action_dim):
///
/// ```text
/// jac     = 1 − tanh²(a_raw[j])
/// jac_ent = 2·t·jac / (jac + ε_stab)   where t = tanh(a_raw[j])
/// σ       = exp(log_sigma[j])
/// δμ[j]      = α·jac_ent − g_a[j]·jac
/// δlog_σ[j]  = α·(−1 + jac_ent·σ·ε[j]) − g_a[j]·jac·σ·ε[j]
/// ```
///
/// # Arguments
///
/// * `mu` — unbounded policy mean μ_raw, length `n`.
/// * `log_sigma` — log standard deviation (clamped), length `n`.
/// * `a_raw` — pre-squash sample `μ + σ·ε`, length `n`.
/// * `eps` — the fixed reparameterisation noise `ε = (a_raw − μ) / σ`, length `n`.
/// * `g_a` — `∂ min(Q1,Q2) / ∂a` evaluated at the squashed action, length `n`.
/// * `alpha` — SAC entropy temperature `α ≥ 0`.
///
/// # Returns
///
/// Descent delta of length `2n`: first `n` entries are `δμ`, next `n` are `δlog_σ`.
pub(crate) fn sac_actor_delta(
    mu: &[f64],
    log_sigma: &[f64],
    a_raw: &[f64],
    eps: &[f64],
    g_a: &[f64],
    alpha: f64,
) -> Vec<f64> {
    let n = mu.len();
    let mut delta = vec![0.0; 2 * n];
    for j in 0..n {
        let t = a_raw[j].tanh();
        let jac = 1.0 - t * t;
        let sigma = log_sigma[j].exp();
        let jac_ent = 2.0 * t * jac / (jac + SQUASH_JAC_EPS);
        // μ-half: entropy gradient − Q pathwise gradient
        delta[j] = alpha * jac_ent - g_a[j] * jac;
        // log_σ-half: entropy gradient − Q pathwise gradient
        delta[n + j] = alpha * (-1.0 + jac_ent * sigma * eps[j]) - g_a[j] * jac * sigma * eps[j];
    }
    delta
}

/// Sample one standard-normal variate via Box–Muller (cosine half).
///
/// Reuses the same sampling convention as the v4.1.0 `act_continuous` Training
/// arm: `u1 ∈ [ε, 1]`, `u2 ∈ [0, 1)`, `ε = (-2·ln u1)^½ · cos(2πu2)`.
/// No new dependencies — only `rand::Rng::gen_range` from the existing `rand = "0.8"` dep.
///
/// # Arguments
///
/// * `rng` — any `rand::Rng` implementor (typically `StdRng`).
///
/// # Returns
///
/// A single `f64` drawn from `N(0, 1)`.
fn sample_standard_normal(rng: &mut impl rand::Rng) -> f64 {
    let u1: f64 = rng.gen_range(f64::EPSILON..=1.0);
    let u2: f64 = rng.gen_range(0.0..1.0);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Split a `2·action_dim` actor output into `(μ_raw, clamp(log_σ_raw, LOG_SIG_MIN, LOG_SIG_MAX))`.
///
/// The dual-head actor emits `[μ_raw | log_σ_raw]` concatenated into `y_conv`.
/// This helper partitions the vector and clamps `log_σ` to `[LOG_SIG_MIN, LOG_SIG_MAX]`
/// so that `σ = exp(log_σ)` stays in a numerically safe range.
///
/// # Arguments
///
/// * `y_conv` — actor convergence output of length `2 * action_dim`.
/// * `action_dim` — number of action components.
///
/// # Returns
///
/// `(mu_raw, log_sigma_clamped)`, each of length `action_dim`.
fn split_mu_log_sigma(y_conv: &[f64], action_dim: usize) -> (Vec<f64>, Vec<f64>) {
    let mu = y_conv[..action_dim].to_vec();
    let log_sigma = y_conv[action_dim..2 * action_dim]
        .iter()
        .map(|&x| x.clamp(LOG_SIG_MIN, LOG_SIG_MAX))
        .collect();
    (mu, log_sigma)
}

/// Deterministic (Play) action: `tanh(μ_raw)` for each action component.
///
/// Used in `SelectionMode::Play` — no RNG, no exploration noise.
///
/// # Arguments
///
/// * `mu_raw` — unbounded mean output from the actor, one value per action component.
///
/// # Returns
///
/// Squashed action vector where each element ∈ (−1, 1).
fn deterministic_squashed_action(mu_raw: &[f64]) -> Vec<f64> {
    mu_raw.iter().map(|&m| m.tanh()).collect()
}

/// Reparameterized sample: returns `(a_raw, a = tanh(a_raw))`.
///
/// Samples `ε ~ N(0, I)` via Box–Muller (factored into [`sample_standard_normal`]),
/// computes `a_raw = μ_raw + exp(log_σ) · ε`, and squashes to `a = tanh(a_raw) ∈ (−1, 1)`.
///
/// # Arguments
///
/// * `mu_raw` — unbounded mean, one value per action component.
/// * `log_sigma` — clamped log standard deviation, one value per action component.
/// * `rng` — any `rand::Rng` implementor.
///
/// # Returns
///
/// `(a_raw, a)` where `a_raw` is the pre-squash sample and `a ∈ (−1, 1)` is the executed action.
fn sample_squashed_action(
    mu_raw: &[f64],
    log_sigma: &[f64],
    rng: &mut impl rand::Rng,
) -> (Vec<f64>, Vec<f64>) {
    let mut a_raw = Vec::with_capacity(mu_raw.len());
    for (i, &m) in mu_raw.iter().enumerate() {
        let eps = sample_standard_normal(rng);
        a_raw.push(m + log_sigma[i].exp() * eps);
    }
    let a = a_raw.iter().map(|&ar| ar.tanh()).collect();
    (a_raw, a)
}

/// Draws a uniform-random action in the squashed space for SAC warmup.
///
/// Each action component `a_j` is sampled from `Uniform(−0.999, 0.999)`,
/// keeping coverage uniform in action space. The pre-squash value is computed
/// as `a_raw_j = atanh(a_j)`. Sampling in the squashed space (then `atanh`)
/// rather than sampling `a_raw` uniformly avoids the clustering of
/// `tanh(a_raw)` at ±1 that arises from uniform sampling in raw space.
///
/// # Parameters
///
/// * `action_dim` — number of action components.
/// * `rng` — any `rand::Rng` implementor (agent's seeded `StdRng`).
///
/// # Returns
///
/// `(a_raw, a)` where `a_raw` holds `atanh(a_j)` and `a ∈ (−0.999, 0.999)`.
fn sample_uniform_squashed_action(
    action_dim: usize,
    rng: &mut impl rand::Rng,
) -> (Vec<f64>, Vec<f64>) {
    let squashed: Vec<f64> = (0..action_dim)
        .map(|_| rng.gen_range(-0.999_f64..=0.999_f64))
        .collect();
    let a_raw: Vec<f64> = squashed.iter().map(|&a| a.atanh()).collect();
    (a_raw, squashed)
}

impl<L: LinAlg> PcActorCritic<L> {
    /// Builds the four SAC Q-critic slots (`q1`, `q2`, `q1_target`, `q2_target`).
    ///
    /// Returns `(None, None, None, None)` when `q_critic_cfg` is `None` (discrete
    /// mode or pre-v6 continuous without SAC). When `Some`, constructs two
    /// independent critics from separate RNG draws and clones each to a matching
    /// target via `from_weights`.
    ///
    /// # Errors
    ///
    /// Propagates any `PcError` from `QCritic::new` or `QCritic::from_weights`.
    // Four Option<QCritic<L>> in the return tuple is intentional — each slot has
    // a distinct role (live q1/q2, target q1/q2) and will be accessed individually.
    #[allow(clippy::type_complexity)]
    fn build_sac_critics(
        backend: &L,
        q_critic_cfg: Option<crate::q_critic::QCriticConfig>,
        rng: &mut impl rand::Rng,
    ) -> Result<
        (
            Option<crate::q_critic::QCritic<L>>,
            Option<crate::q_critic::QCritic<L>>,
            Option<crate::q_critic::QCritic<L>>,
            Option<crate::q_critic::QCritic<L>>,
        ),
        PcError,
    > {
        let cfg = match q_critic_cfg {
            Some(c) => c,
            None => return Ok((None, None, None, None)),
        };

        let q1 = crate::q_critic::QCritic::new(backend.clone(), cfg.clone(), rng)?;
        let q2 = crate::q_critic::QCritic::new(backend.clone(), cfg.clone(), rng)?;

        // Clone targets from live critics via round-trip through weights —
        // QCritic has no Clone impl, so from_weights is the safe copy path.
        let q1_target =
            crate::q_critic::QCritic::from_weights(backend.clone(), cfg.clone(), q1.to_weights())?;
        let q2_target =
            crate::q_critic::QCritic::from_weights(backend.clone(), cfg, q2.to_weights())?;

        Ok((Some(q1), Some(q2), Some(q1_target), Some(q2_target)))
    }

    /// Returns the eligibility trace length: output_size when GAE enabled, 0 otherwise.
    fn gae_trace_len(config: &PcActorCriticConfig) -> usize {
        if config.gae_lambda.is_some() {
            config.actor.output_size
        } else {
            0
        }
    }

    /// Precomputes per-hidden-layer decay factors for actor and critic (M3a),
    /// plus per-layer error EMA initialization for adaptive consolidation (M3b).
    ///
    /// Returns `(actor_decay_factors, critic_decay_factors, layer_error_ema)`.
    fn compute_decay_factors(config: &PcActorCriticConfig) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n_ah = config.actor.hidden_layers.len();
        let actor_decay_factors: Vec<f64> = (0..n_ah)
            .map(|i| config.consolidation_decay.powi((n_ah - 1 - i) as i32))
            .collect();

        let n_ch = config.critic.hidden_layers.len();
        let critic_decay_factors: Vec<f64> = (0..n_ch)
            .map(|i| {
                config
                    .critic_consolidation_decay
                    .powi((n_ch - 1 - i) as i32)
            })
            .collect();

        let layer_error_ema = if config.adaptive_consolidation {
            vec![0.0; n_ah]
        } else {
            Vec::new()
        };

        (actor_decay_factors, critic_decay_factors, layer_error_ema)
    }

    /// Returns the effective `positive_only` flag for the replay buffer.
    ///
    /// Continuous SAC must retain ALL transitions regardless of reward sign
    /// (Pendulum-v1 rewards are always ≤ 0; a positive-only filter would leave the
    /// buffer permanently empty and block learning).  Forces `false` for
    /// `ActionSpace::Continuous`; honours `config.replay_positive_only` unchanged
    /// for discrete agents.
    ///
    /// Shared by `new()` and `apply_config()` so the override is guaranteed in
    /// both construction paths.
    fn effective_positive_only(config: &PcActorCriticConfig) -> bool {
        if config.action_space == ActionSpace::Continuous {
            false
        } else {
            config.replay_positive_only
        }
    }

    /// Builds an optional `HysteresisState` from config parameters.
    ///
    /// Returns `Some(fresh Plastic state)` when enabled, `None` when disabled.
    /// Shared by `new()` and `apply_config()` to avoid construction duplication.
    fn build_hysteresis(
        enabled: bool,
        fast_window: usize,
        slow_window: usize,
        wake_fraction: f64,
        sleep_fraction: f64,
    ) -> Option<HysteresisState> {
        if enabled {
            Some(HysteresisState {
                fast: EwmaTracker::new(fast_window),
                slow: EwmaTracker::new(slow_window),
                state: PlasticityState::Plastic,
                wake_fraction,
                sleep_fraction,
                min_initial_plastic: slow_window as u64,
            })
        } else {
            None
        }
    }

    /// Allocates Fisher state for a set of layers when EWC is enabled.
    ///
    /// Returns one `FisherState` per layer when `ewc_lambda > 0`, or an empty
    /// `Vec` when EWC is disabled (zero overhead).
    /// Shared by `new()` and `apply_config()`.
    fn build_fisher_for_layers(
        backend: &L,
        layers: &[crate::layer::Layer<L>],
        ewc_lambda: f64,
    ) -> Vec<FisherState<L>> {
        if ewc_lambda > 0.0 {
            layers
                .iter()
                .map(|layer| {
                    let rows = backend.mat_rows(&layer.weights);
                    let cols = backend.mat_cols(&layer.weights);
                    let bias_size = backend.vec_len(&layer.bias);
                    FisherState::new(backend, rows, cols, bias_size)
                })
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Computes the minimum Fisher phase length from `fisher_ema_beta`.
    ///
    /// Returns `ceil(1 / (1 - beta))` when EWC is enabled, 0 otherwise.
    fn min_fisher_phase(config: &PcActorCriticConfig) -> u64 {
        if config.ewc_lambda > 0.0 {
            (1.0 / (1.0 - config.fisher_ema_beta)).ceil() as u64
        } else {
            0
        }
    }

    /// Allocates the Polyak + Frozen distillation anchor slots based on the
    /// configured distillation lambdas.
    ///
    /// Returns `(polyak_target, frozen_champion)`. Each slot is
    /// `Some(actor.clone())` when its corresponding `distillation_lambda_*`
    /// is strictly positive, `None` otherwise. A lambda of exactly `0.0`
    /// disables the slot entirely — no `PcActor` clone is allocated.
    ///
    /// This is the single authoritative allocation site for both anchors:
    /// every `PcActorCritic` constructor (`new`, `crossover`, `from_parts`)
    /// and `apply_config` delegates here so the lambda-based slot-presence
    /// invariant is guaranteed by construction. Centralising this logic
    /// resolves the DRY concern raised by MAGI Gate A W2.
    fn allocate_anchor_slots(
        config: &PcActorCriticConfig,
        actor: &PcActor<L>,
    ) -> (Option<PcActor<L>>, Option<PcActor<L>>) {
        let polyak_target = if config.distillation_lambda_polyak > 0.0 {
            Some(actor.clone())
        } else {
            None
        };
        let frozen_champion = if config.distillation_lambda_frozen > 0.0 {
            Some(actor.clone())
        } else {
            None
        };
        (polyak_target, frozen_champion)
    }

    /// Validates a [`PcActorCriticConfig`] for internal consistency.
    ///
    /// Checks gamma range, surprise buffer size, scale floor/ceil ordering,
    /// hysteresis fractions, consolidation decay bounds, EWC parameter bounds,
    /// td_steps validity, and gae_lambda/td_steps mutual exclusion.
    ///
    /// Shared by [`new()`](Self::new) and [`apply_config()`](Self::apply_config) to ensure
    /// identical validation rules. Does NOT validate topology match (that
    /// requires an existing agent).
    ///
    /// # Errors
    ///
    /// Returns [`PcError::ConfigValidation`] with a descriptive message on
    /// the first failing check.
    fn validate_config(config: &PcActorCriticConfig) -> Result<(), PcError> {
        // Per-network f64 fields: reject NaN/Inf early to prevent confusing
        // topology-mismatch errors downstream (NaN != NaN is always true).
        if !config.actor.lr_weights.is_finite() {
            return Err(PcError::ConfigValidation(format!(
                "actor lr_weights must be finite, got {}",
                config.actor.lr_weights
            )));
        }
        if !config.actor.alpha.is_finite() {
            return Err(PcError::ConfigValidation(format!(
                "actor alpha must be finite, got {}",
                config.actor.alpha
            )));
        }
        if !config.actor.tol.is_finite() {
            return Err(PcError::ConfigValidation(format!(
                "actor tol must be finite, got {}",
                config.actor.tol
            )));
        }
        if !config.actor.temperature.is_finite() {
            return Err(PcError::ConfigValidation(format!(
                "actor temperature must be finite, got {}",
                config.actor.temperature
            )));
        }
        if !config.actor.local_lambda.is_finite() {
            return Err(PcError::ConfigValidation(format!(
                "actor local_lambda must be finite, got {}",
                config.actor.local_lambda
            )));
        }
        if !config.actor.rezero_init.is_finite() {
            return Err(PcError::ConfigValidation(format!(
                "actor rezero_init must be finite, got {}",
                config.actor.rezero_init
            )));
        }
        if !config.critic.lr.is_finite() {
            return Err(PcError::ConfigValidation(format!(
                "critic lr must be finite, got {}",
                config.critic.lr
            )));
        }
        if !config.scale_floor.is_finite() {
            return Err(PcError::ConfigValidation(format!(
                "scale_floor must be finite, got {}",
                config.scale_floor
            )));
        }
        if !config.scale_ceil.is_finite() {
            return Err(PcError::ConfigValidation(format!(
                "scale_ceil must be finite, got {}",
                config.scale_ceil
            )));
        }

        if !(0.0..=1.0).contains(&config.gamma) {
            return Err(PcError::ConfigValidation(format!(
                "gamma must be in [0.0, 1.0], got {}",
                config.gamma
            )));
        }
        if config.adaptive_surprise && config.surprise_buffer_size < 10 {
            return Err(PcError::ConfigValidation(format!(
                "surprise_buffer_size must be >= 10 when adaptive_surprise is enabled, got {}",
                config.surprise_buffer_size
            )));
        }
        if config.scale_floor < 0.0 {
            return Err(PcError::ConfigValidation(format!(
                "scale_floor must be >= 0.0, got {}",
                config.scale_floor
            )));
        }
        if config.scale_ceil <= config.scale_floor {
            return Err(PcError::ConfigValidation(format!(
                "scale_ceil must be > scale_floor, got scale_ceil={} scale_floor={}",
                config.scale_ceil, config.scale_floor
            )));
        }

        // Validate actor hysteresis fractions
        if config.actor_hysteresis {
            if config.actor_wake_fraction <= 0.0 {
                return Err(PcError::ConfigValidation(format!(
                    "actor_wake_fraction must be > 0.0 when actor_hysteresis enabled, got {}",
                    config.actor_wake_fraction
                )));
            }
            if config.actor_sleep_fraction <= 0.0 || config.actor_sleep_fraction >= 1.0 {
                return Err(PcError::ConfigValidation(format!(
                    "actor_sleep_fraction must be in (0.0, 1.0) when actor_hysteresis enabled, got {}",
                    config.actor_sleep_fraction
                )));
            }
        }

        // Validate critic hysteresis fractions
        if config.critic_hysteresis {
            if config.critic_wake_fraction <= 0.0 {
                return Err(PcError::ConfigValidation(format!(
                    "critic_wake_fraction must be > 0.0 when critic_hysteresis enabled, got {}",
                    config.critic_wake_fraction
                )));
            }
            if config.critic_sleep_fraction <= 0.0 || config.critic_sleep_fraction >= 1.0 {
                return Err(PcError::ConfigValidation(format!(
                    "critic_sleep_fraction must be in (0.0, 1.0) when critic_hysteresis enabled, got {}",
                    config.critic_sleep_fraction
                )));
            }
        }

        // Validate consolidation decay (M3a)
        if !(0.0..=1.0).contains(&config.consolidation_decay) {
            return Err(PcError::ConfigValidation(format!(
                "consolidation_decay must be in [0.0, 1.0], got {}",
                config.consolidation_decay
            )));
        }
        if !(0.0..=1.0).contains(&config.critic_consolidation_decay) {
            return Err(PcError::ConfigValidation(format!(
                "critic_consolidation_decay must be in [0.0, 1.0], got {}",
                config.critic_consolidation_decay
            )));
        }

        // Validate adaptive consolidation params (M3b)
        if config.adaptive_consolidation {
            if config.consolidation_sigmoid_k <= 0.0 {
                return Err(PcError::ConfigValidation(format!(
                    "consolidation_sigmoid_k must be > 0.0 when adaptive_consolidation enabled, got {}",
                    config.consolidation_sigmoid_k
                )));
            }
            if config.consolidation_ema_beta <= 0.0 || config.consolidation_ema_beta >= 1.0 {
                return Err(PcError::ConfigValidation(format!(
                    "consolidation_ema_beta must be in (0.0, 1.0), got {}",
                    config.consolidation_ema_beta
                )));
            }
            if config.consolidation_error_threshold <= 0.0 {
                return Err(PcError::ConfigValidation(format!(
                    "consolidation_error_threshold must be > 0.0, got {}",
                    config.consolidation_error_threshold
                )));
            }
        }

        // Validate EWC parameters (M4)
        if config.ewc_lambda < 0.0 {
            return Err(PcError::ConfigValidation(format!(
                "ewc_lambda must be >= 0.0, got {}",
                config.ewc_lambda
            )));
        }
        if config.ewc_lambda > 0.0 {
            if !(0.0..=1.0).contains(&config.fisher_decay) {
                return Err(PcError::ConfigValidation(format!(
                    "fisher_decay must be in [0.0, 1.0], got {}",
                    config.fisher_decay
                )));
            }
            if config.fisher_ema_beta <= 0.0 || config.fisher_ema_beta >= 1.0 {
                return Err(PcError::ConfigValidation(format!(
                    "fisher_ema_beta must be in (0.0, 1.0), got {}",
                    config.fisher_ema_beta
                )));
            }
        }

        if config.td_steps == 1 {
            return Err(PcError::ConfigValidation(
                "td_steps=1 is not supported — use 0 for TD(0) or >= 2 for multi-step".to_string(),
            ));
        }

        if let Some(lambda) = config.gae_lambda {
            if !(0.0..=1.0).contains(&lambda) {
                return Err(PcError::ConfigValidation(format!(
                    "gae_lambda must be in [0.0, 1.0], got {lambda}"
                )));
            }
            if config.td_steps > 0 {
                return Err(PcError::ConfigValidation(
                    "gae_lambda and td_steps > 0 are mutually exclusive".to_string(),
                ));
            }
        }

        // Validate Polyak distillation parameters
        if !config.distillation_lambda_polyak.is_finite() || config.distillation_lambda_polyak < 0.0
        {
            return Err(PcError::ConfigValidation(format!(
                "distillation_lambda_polyak must be finite and >= 0.0, got {}",
                config.distillation_lambda_polyak
            )));
        }
        if !config.polyak_tau.is_finite() || !(0.0..=1.0).contains(&config.polyak_tau) {
            return Err(PcError::ConfigValidation(format!(
                "polyak_tau must be finite and in [0.0, 1.0], got {}",
                config.polyak_tau
            )));
        }
        if config.distillation_lambda_polyak > 0.0 && config.polyak_tau == 0.0 {
            return Err(PcError::ConfigValidation(
                "polyak_tau must be > 0.0 when distillation_lambda_polyak > 0".to_string(),
            ));
        }
        if !config.distillation_lambda_frozen.is_finite() || config.distillation_lambda_frozen < 0.0
        {
            return Err(PcError::ConfigValidation(format!(
                "distillation_lambda_frozen must be finite and >= 0.0, got {}",
                config.distillation_lambda_frozen
            )));
        }

        // Phase 2: replay buffer validation.
        if config.replay_recent_capacity > 0 && config.replay_training_capacity == 0 {
            return Err(PcError::ConfigValidation(
                "replay_recent_capacity > 0 requires replay_training_capacity > 0".to_string(),
            ));
        }
        if config.replay_training_capacity > 0 && config.replay_batch_size == 0 {
            return Err(PcError::ConfigValidation(
                "replay_batch_size must be > 0 when replay buffer is enabled".to_string(),
            ));
        }
        // If the batch is larger than the buffer it can never be filled → silent
        // no-learning.  Reject early so the misconfiguration is surfaced at
        // construction rather than discovered at the first sac_learn_step call.
        //
        // Scoped to continuous SAC only: discrete replay semantics differ
        // (the buffer is optional, and discrete agents with batch > capacity
        // are valid pre-v6 configurations that must remain constructable).
        if config.action_space == ActionSpace::Continuous
            && config.replay_training_capacity > 0
            && config.replay_batch_size > config.replay_training_capacity
        {
            return Err(PcError::ConfigValidation(format!(
                "replay_batch_size ({}) exceeds replay_training_capacity ({}); \
                 the buffer can never accumulate a full batch and will never \
                 produce a learning update. Reduce replay_batch_size or \
                 increase replay_training_capacity.",
                config.replay_batch_size, config.replay_training_capacity,
            )));
        }

        // v6.0.0 — canonical SAC continuous-mode rules (replaces v4/v5 on-policy rules).
        if config.action_space == ActionSpace::Continuous {
            // Hysteresis machinery is bypassed entirely in the SAC continuous
            // learning path; enabling it would silently do nothing and mislead
            // the caller.  Reject early so the misconfiguration is surfaced at
            // construction rather than discovered as a no-op at runtime.
            if config.actor_hysteresis || config.critic_hysteresis {
                return Err(PcError::ConfigValidation(
                    "actor_hysteresis/critic_hysteresis are not supported in continuous SAC \
                     mode (v6.0.0); the SAC learning path bypasses hysteresis machinery. \
                     Set actor_hysteresis=false and critic_hysteresis=false, or use \
                     ActionSpace::Discrete."
                        .to_string(),
                ));
            }

            // policy_sigma is IGNORED by SAC (σ is learned from the dual-head actor)
            // but the field must remain finite to pass the general f64 check.
            if !config.policy_sigma.is_finite() {
                return Err(PcError::ConfigValidation(format!(
                    "policy_sigma ({}) must be finite when action_space == Continuous.",
                    config.policy_sigma
                )));
            }

            // KL distillation is undefined for raw continuous output.
            if config.distillation_lambda_polyak > 0.0 {
                return Err(PcError::ConfigValidation(format!(
                    "distillation_lambda_polyak ({}) is not supported in \
                     continuous action space (KL is undefined for raw \
                     output). Set to 0.0 or use ActionSpace::Discrete.",
                    config.distillation_lambda_polyak
                )));
            }
            if config.distillation_lambda_frozen > 0.0 {
                return Err(PcError::ConfigValidation(format!(
                    "distillation_lambda_frozen ({}) is not supported in \
                     continuous action space — same reason as Polyak.",
                    config.distillation_lambda_frozen
                )));
            }

            // TD(n) flush is not supported for continuous mode.
            if config.td_steps != 0 {
                return Err(PcError::ConfigValidation(format!(
                    "td_steps ({}) > 0 is not supported in continuous action space. \
                     Set to 0 or use ActionSpace::Discrete.",
                    config.td_steps
                )));
            }

            // SAC requires Linear output — the actor emits unbounded μ and log_σ;
            // a bounded activation re-introduces the vanishing-gradient trap.
            if config.actor.output_activation != crate::activation::Activation::Linear {
                return Err(PcError::ConfigValidation(format!(
                    "continuous action space requires actor.output_activation == Linear \
                     (the actor emits μ and log_σ; actions are tanh-squashed internally). \
                     Got {:?}.",
                    config.actor.output_activation
                )));
            }

            // SAC requires a Q-critic.
            let q_cfg = match &config.q_critic {
                Some(q) => q,
                None => {
                    return Err(PcError::ConfigValidation(
                        "continuous action space (SAC) requires q_critic to be Some(..). \
                         Set q_critic with a valid QCriticConfig."
                            .to_string(),
                    ));
                }
            };

            // actor.output_size must equal 2 * action_dim (μ head + log_σ head).
            let action_dim = q_cfg.action_dim;
            let expected_output = 2 * action_dim;
            if config.actor.output_size != expected_output {
                return Err(PcError::ConfigValidation(format!(
                    "continuous SAC requires actor.output_size == 2 * q_critic.action_dim \
                     = 2 * {action_dim} = {expected_output}, got {} (actor emits μ and \
                     log_σ heads).",
                    config.actor.output_size
                )));
            }

            // q_critic.state_dim must match actor.input_size.
            if q_cfg.state_dim != config.actor.input_size {
                return Err(PcError::ConfigValidation(format!(
                    "q_critic.state_dim ({}) must equal actor.input_size ({}) so the \
                     Q-critic receives the same observation as the actor.",
                    q_cfg.state_dim, config.actor.input_size
                )));
            }

            // SAC requires a replay buffer.
            if config.replay_training_capacity == 0 {
                return Err(PcError::ConfigValidation(
                    "continuous action space (SAC) requires replay_training_capacity > 0. \
                     SAC is an off-policy algorithm and needs a replay buffer."
                        .to_string(),
                ));
            }
            if config.replay_batch_size == 0 {
                return Err(PcError::ConfigValidation(
                    "continuous action space (SAC) requires replay_batch_size > 0 when \
                     replay buffer is enabled."
                        .to_string(),
                ));
            }

            // Validate target_entropy if provided.
            if let Some(te) = config.target_entropy {
                if !te.is_finite() {
                    return Err(PcError::ConfigValidation(format!(
                        "target_entropy ({te}) must be finite when set. \
                         Use None to let the library use the −action_dim heuristic."
                    )));
                }
            }

            // Validate polyak_tau (must be strictly > 0 for SAC soft target updates).
            if config.polyak_tau <= 0.0 {
                return Err(PcError::ConfigValidation(format!(
                    "polyak_tau ({}) must be > 0.0 in continuous SAC mode \
                     (soft target network updates require a positive mixing rate).",
                    config.polyak_tau
                )));
            }

            // Validate temperature learning rate and initial log-temperature.
            if !config.alpha_lr.is_finite() || config.alpha_lr <= 0.0 {
                return Err(PcError::ConfigValidation(format!(
                    "alpha_lr ({}) must be finite and > 0.0 in continuous SAC mode.",
                    config.alpha_lr
                )));
            }
            if !config.log_alpha_init.is_finite() {
                return Err(PcError::ConfigValidation(format!(
                    "log_alpha_init ({}) must be finite in continuous SAC mode.",
                    config.log_alpha_init
                )));
            }
        }

        // Tolerance-based sentinel check + NaN/Infinity rejection + upper bound
        // per MAGI Melchior + Caspar Checkpoint 2 iter 2 hardening requirement.
        // The actor and critic replay-floor fields share the same tri-state
        // sentinel semantics; the helper dedupes the rule (v3.0.0 refactor).
        let upper_bound = 10.0 * config.scale_ceil;
        Self::validate_replay_floor("scale_floor_replay", config.scale_floor_replay, upper_bound)?;
        Self::validate_replay_floor(
            "critic_floor_replay",
            config.critic_floor_replay,
            upper_bound,
        )?;

        // v4.0.1 hardening — the critic consumes latent_concat: the raw state
        // concatenated with every actor hidden-layer activation. Enforce the
        // derived-size invariant at construction so a wrong critic.input_size
        // fails here as a recoverable ConfigValidation error, instead of as a
        // lazy panic in MlpCritic::forward on the first critic forward pass.
        let expected_critic_input = config.actor.input_size
            + config
                .actor
                .hidden_layers
                .iter()
                .map(|l| l.size)
                .sum::<usize>();
        if config.critic.input_size != expected_critic_input {
            return Err(PcError::ConfigValidation(format!(
                "critic.input_size ({}) must equal actor.input_size + sum(actor \
                 hidden layer sizes) = {} (the critic consumes latent_concat: the \
                 raw state concatenated with every actor hidden activation).",
                config.critic.input_size, expected_critic_input
            )));
        }

        Ok(())
    }

    /// Validates a replay-floor opt-in field per the shared tri-state
    /// sentinel contract.
    ///
    /// Accepts: `~-1.0` (sentinel for "opt-in not provided") OR any
    /// finite value in the **closed inclusive** interval
    /// `[0.0, upper_bound]` (`value == upper_bound` is accepted —
    /// `value > upper_bound` is rejected). Rejects: values in
    /// `(-1.0, 0.0)`, NaN, and ±Infinity.
    ///
    /// Used by both `scale_floor_replay` (v2.2.1) and
    /// `critic_floor_replay` (v3.0.0) — extracted in v3.0.0 to dedupe
    /// the validation rule across the two symmetric fields.
    fn validate_replay_floor(
        field_name: &str,
        value: f64,
        upper_bound: f64,
    ) -> Result<(), PcError> {
        let is_sentinel = crate::pc_actor_critic::config::is_replay_floor_sentinel(value);
        let is_valid_positive = value.is_finite() && value >= 0.0 && value <= upper_bound;
        if !is_sentinel && !is_valid_positive {
            return Err(PcError::ConfigValidation(format!(
                "{field_name} ({value}) must be either ~-1.0 (sentinel for \
                 'use scale_floor') or a finite non-negative value <= \
                 {upper_bound} (10× scale_ceil). NaN, ±Infinity, and values \
                 in (-1.0, 0.0) are rejected as ambiguous or likely \
                 configuration errors."
            )));
        }
        Ok(())
    }

    /// Relative-epsilon comparison for `f64` config fields.
    ///
    /// Tolerates small round-trip drift (JSON serialize/parse, repeated
    /// arithmetic) while still catching any semantically meaningful change.
    /// Returns true when `a == b` exactly, or when
    /// `|a - b| <= 4 * eps * max(|a|, |b|, 1.0)`.
    fn f64_approx_eq(a: f64, b: f64) -> bool {
        if a == b {
            return true;
        }
        if !a.is_finite() || !b.is_finite() {
            return false;
        }
        let scale = a.abs().max(b.abs()).max(1.0);
        (a - b).abs() <= 4.0 * f64::EPSILON * scale
    }

    /// Validates that a new config's network topology, structural parameters,
    /// and per-network parameters match the current agent.
    ///
    /// **Topology:** actor input/hidden/output sizes, critic input/hidden
    /// sizes, and hidden-layer activation functions (empirically these
    /// change network dynamics drastically — see CLAUDE.md training results
    /// for the tanh→relu depth collapse).
    /// **Structural:** output_activation, residual, rezero_init — these
    /// affect how existing weights are interpreted during forward pass.
    /// **Per-network:** actor lr_weights/alpha/tol/min_steps/max_steps/
    /// temperature/local_lambda/synchronous, critic lr — these live in
    /// `self.actor.config` and `self.critic.config` separately; mismatches
    /// would create divergence with `self.config.actor`/`self.config.critic`.
    /// Per-network params are immutable across [`apply_config`](Self::apply_config);
    /// reconstruct the agent with [`new`](Self::new) if they need to change.
    ///
    /// `f64` fields are compared with a relative epsilon
    /// (`f64_approx_eq`) to tolerate JSON round-trip
    /// drift; all other fields use exact equality.
    ///
    /// # Errors
    ///
    /// Returns [`PcError::ConfigValidation`] identifying which field mismatches.
    fn validate_topology_match(&self, config: &PcActorCriticConfig) -> Result<(), PcError> {
        // Exhaustive destructuring forces a compile error when a new field is
        // added to PcActorConfig or MlpCriticConfig, preventing silent drift
        // of this check. Do NOT replace these with `..` — the compile error is
        // the point.
        let PcActorConfig {
            input_size: cur_a_input,
            hidden_layers: cur_a_hidden,
            output_size: cur_a_output,
            output_activation: cur_a_out_act,
            alpha: cur_a_alpha,
            tol: cur_a_tol,
            min_steps: cur_a_min_steps,
            max_steps: cur_a_max_steps,
            lr_weights: cur_a_lr,
            synchronous: cur_a_sync,
            temperature: cur_a_temp,
            local_lambda: cur_a_lambda,
            residual: cur_a_residual,
            rezero_init: cur_a_rezero,
        } = &self.config.actor;
        let PcActorConfig {
            input_size: new_a_input,
            hidden_layers: new_a_hidden,
            output_size: new_a_output,
            output_activation: new_a_out_act,
            alpha: new_a_alpha,
            tol: new_a_tol,
            min_steps: new_a_min_steps,
            max_steps: new_a_max_steps,
            lr_weights: new_a_lr,
            synchronous: new_a_sync,
            temperature: new_a_temp,
            local_lambda: new_a_lambda,
            residual: new_a_residual,
            rezero_init: new_a_rezero,
        } = &config.actor;
        let MlpCriticConfig {
            input_size: cur_c_input,
            hidden_layers: cur_c_hidden,
            output_activation: cur_c_out_act,
            lr: cur_c_lr,
        } = &self.config.critic;
        let MlpCriticConfig {
            input_size: new_c_input,
            hidden_layers: new_c_hidden,
            output_activation: new_c_out_act,
            lr: new_c_lr,
        } = &config.critic;

        // Actor topology
        if cur_a_input != new_a_input {
            return Err(PcError::ConfigValidation(format!(
                "actor input_size mismatch: current {cur_a_input} vs new {new_a_input}"
            )));
        }
        if cur_a_hidden.len() != new_a_hidden.len() {
            return Err(PcError::ConfigValidation(format!(
                "actor hidden layer count mismatch: current {} vs new {}",
                cur_a_hidden.len(),
                new_a_hidden.len()
            )));
        }
        for (i, (cur, new)) in cur_a_hidden.iter().zip(new_a_hidden.iter()).enumerate() {
            if cur.size != new.size {
                return Err(PcError::ConfigValidation(format!(
                    "actor hidden layer {} size mismatch: current {} vs new {}",
                    i, cur.size, new.size
                )));
            }
            if cur.activation != new.activation {
                return Err(PcError::ConfigValidation(format!(
                    "actor hidden layer {} activation mismatch: current {:?} vs new {:?} — \
                     empirically, changing activation mid-training can collapse learned \
                     behavior; reconstruct agent instead",
                    i, cur.activation, new.activation
                )));
            }
        }
        if cur_a_output != new_a_output {
            return Err(PcError::ConfigValidation(format!(
                "actor output_size mismatch: current {cur_a_output} vs new {new_a_output}"
            )));
        }

        // Critic topology
        if cur_c_input != new_c_input {
            return Err(PcError::ConfigValidation(format!(
                "critic input_size mismatch: current {cur_c_input} vs new {new_c_input}"
            )));
        }
        if cur_c_hidden.len() != new_c_hidden.len() {
            return Err(PcError::ConfigValidation(format!(
                "critic hidden layer count mismatch: current {} vs new {}",
                cur_c_hidden.len(),
                new_c_hidden.len()
            )));
        }
        for (i, (cur, new)) in cur_c_hidden.iter().zip(new_c_hidden.iter()).enumerate() {
            if cur.size != new.size {
                return Err(PcError::ConfigValidation(format!(
                    "critic hidden layer {} size mismatch: current {} vs new {}",
                    i, cur.size, new.size
                )));
            }
            if cur.activation != new.activation {
                return Err(PcError::ConfigValidation(format!(
                    "critic hidden layer {} activation mismatch: current {:?} vs new {:?} — \
                     reconstruct agent instead",
                    i, cur.activation, new.activation
                )));
            }
        }

        // Actor structural (affect weight interpretation)
        if cur_a_out_act != new_a_out_act {
            return Err(PcError::ConfigValidation(format!(
                "actor output_activation mismatch: current {cur_a_out_act:?} vs new {new_a_out_act:?}"
            )));
        }
        if cur_a_residual != new_a_residual {
            return Err(PcError::ConfigValidation(format!(
                "actor residual mismatch: current {cur_a_residual} vs new {new_a_residual}"
            )));
        }
        if !Self::f64_approx_eq(*cur_a_rezero, *new_a_rezero) {
            return Err(PcError::ConfigValidation(format!(
                "actor rezero_init mismatch: current {cur_a_rezero} vs new {new_a_rezero}"
            )));
        }

        // Actor per-network params (must match self.actor.config exactly;
        // apply_config cannot mutate fields that live in the inner network).
        if !Self::f64_approx_eq(*cur_a_lr, *new_a_lr) {
            return Err(PcError::ConfigValidation(format!(
                "actor lr_weights mismatch: current {cur_a_lr} vs new {new_a_lr} — \
                 per-network params are immutable across apply_config(); reconstruct agent instead"
            )));
        }
        if !Self::f64_approx_eq(*cur_a_alpha, *new_a_alpha) {
            return Err(PcError::ConfigValidation(format!(
                "actor alpha mismatch: current {cur_a_alpha} vs new {new_a_alpha} — \
                 per-network params are immutable across apply_config()"
            )));
        }
        if !Self::f64_approx_eq(*cur_a_tol, *new_a_tol) {
            return Err(PcError::ConfigValidation(format!(
                "actor tol mismatch: current {cur_a_tol} vs new {new_a_tol}"
            )));
        }
        if cur_a_min_steps != new_a_min_steps {
            return Err(PcError::ConfigValidation(format!(
                "actor min_steps mismatch: current {cur_a_min_steps} vs new {new_a_min_steps}"
            )));
        }
        if cur_a_max_steps != new_a_max_steps {
            return Err(PcError::ConfigValidation(format!(
                "actor max_steps mismatch: current {cur_a_max_steps} vs new {new_a_max_steps}"
            )));
        }
        if !Self::f64_approx_eq(*cur_a_temp, *new_a_temp) {
            return Err(PcError::ConfigValidation(format!(
                "actor temperature mismatch: current {cur_a_temp} vs new {new_a_temp} — \
                 per-network params are immutable across apply_config()"
            )));
        }
        if !Self::f64_approx_eq(*cur_a_lambda, *new_a_lambda) {
            return Err(PcError::ConfigValidation(format!(
                "actor local_lambda mismatch: current {cur_a_lambda} vs new {new_a_lambda}"
            )));
        }
        if cur_a_sync != new_a_sync {
            return Err(PcError::ConfigValidation(format!(
                "actor synchronous mismatch: current {cur_a_sync} vs new {new_a_sync}"
            )));
        }

        // Critic per-network params
        if !Self::f64_approx_eq(*cur_c_lr, *new_c_lr) {
            return Err(PcError::ConfigValidation(format!(
                "critic lr mismatch: current {cur_c_lr} vs new {new_c_lr} — \
                 per-network params are immutable across apply_config(); reconstruct agent instead"
            )));
        }
        if cur_c_out_act != new_c_out_act {
            return Err(PcError::ConfigValidation(format!(
                "critic output_activation mismatch: current {cur_c_out_act:?} vs new {new_c_out_act:?}"
            )));
        }

        // Q-critic topology: reject topology changes for continuous SAC agents.
        // Rebuilding Q-critics from scratch would discard learned Q-weights and
        // requires an RNG; rejecting is the safe minimal choice, consistent with
        // how actor/critic topology changes are handled above.
        // Transitioning from Some → None or None → Some changes the mode entirely;
        // validate_config / SAC-mode checks in apply_config catch that independently.
        if let (Some(cur_q), Some(new_q)) = (&self.config.q_critic, &config.q_critic) {
            if cur_q.state_dim != new_q.state_dim {
                return Err(PcError::ConfigValidation(format!(
                    "apply_config cannot change q_critic topology for an existing SAC agent; \
                     reconstruct via new(). \
                     q_critic.state_dim mismatch: current {} vs new {}",
                    cur_q.state_dim, new_q.state_dim
                )));
            }
            if cur_q.action_dim != new_q.action_dim {
                return Err(PcError::ConfigValidation(format!(
                    "apply_config cannot change q_critic topology for an existing SAC agent; \
                     reconstruct via new(). \
                     q_critic.action_dim mismatch: current {} vs new {}",
                    cur_q.action_dim, new_q.action_dim
                )));
            }
            if cur_q.hidden_layers.len() != new_q.hidden_layers.len() {
                return Err(PcError::ConfigValidation(format!(
                    "apply_config cannot change q_critic topology for an existing SAC agent; \
                     reconstruct via new(). \
                     q_critic hidden layer count mismatch: current {} vs new {}",
                    cur_q.hidden_layers.len(),
                    new_q.hidden_layers.len()
                )));
            }
            for (i, (cur_hl, new_hl)) in cur_q
                .hidden_layers
                .iter()
                .zip(new_q.hidden_layers.iter())
                .enumerate()
            {
                if cur_hl.size != new_hl.size {
                    return Err(PcError::ConfigValidation(format!(
                        "apply_config cannot change q_critic topology for an existing SAC agent; \
                         reconstruct via new(). \
                         q_critic hidden layer {} size mismatch: current {} vs new {}",
                        i, cur_hl.size, new_hl.size
                    )));
                }
                if cur_hl.activation != new_hl.activation {
                    return Err(PcError::ConfigValidation(format!(
                        "apply_config cannot change q_critic topology for an existing SAC agent; \
                         reconstruct via new(). \
                         q_critic hidden layer {} activation mismatch: current {:?} vs new {:?}",
                        i, cur_hl.activation, new_hl.activation
                    )));
                }
            }
        }

        Ok(())
    }

    /// Applies a new configuration to the agent, preserving weights and topology.
    ///
    /// Reconfigures **agent-level** learning parameters (gamma, surprise, CL,
    /// TD/GAE mode). Per-network parameters (actor lr, temperature, alpha;
    /// critic lr) cannot be changed here — they live in `self.actor.config` /
    /// `self.critic.config` and are immutable across this call. Pass the same
    /// actor/critic sub-config the agent was constructed with; mismatches are
    /// rejected. If per-network params need to change, reconstruct the agent
    /// with [`new`](Self::new) and transfer weights via the serializer.
    ///
    /// `f64` fields are compared with a relative epsilon to tolerate JSON
    /// round-trip drift.
    ///
    /// All continuous learning state is reset to provide a clean baseline.
    ///
    /// # What changes
    ///
    /// Gamma, surprise thresholds, scale floor/ceil (M1), hysteresis (M2),
    /// consolidation decay (M3), EWC parameters (M4), TD(n) steps, GAE lambda,
    /// entropy coefficient, logits reversal, bidirectional coupling.
    ///
    /// # Replay buffer transitions
    ///
    /// * `0 → positive capacity`: a fresh empty buffer is allocated with
    ///   the new config's sizing and `positive_only` flag.
    /// * `positive → 0`: the existing buffer is deallocated; any
    ///   accumulated transitions are dropped.
    /// * `positive → positive`: if capacity fields and `positive_only`
    ///   are unchanged the existing buffer contents are preserved;
    ///   otherwise the buffer is reset to an empty state sized per the
    ///   new config (FIFO ordering is not transferable between
    ///   differently-sized buffers).
    /// * `replay_clamp_count` is always reset to 0 on `apply_config`
    ///   so telemetry reflects the new configuration's history.
    ///
    /// # What does NOT change
    ///
    /// Actor/critic weights and biases, network topology, actor lr/alpha/tol/
    /// min_steps/max_steps/temperature/local_lambda, critic lr, RNG state
    /// (continues generating fresh pseudo-random numbers from current position),
    /// backend.
    ///
    /// # Errors
    ///
    /// Returns `PcError::ConfigValidation` if the new config fails validation
    /// or if its topology does not match the current agent.
    pub fn apply_config(&mut self, config: PcActorCriticConfig) -> Result<(), PcError> {
        // 0. Defense-in-depth: verify config coherence invariant.
        // All per-network fields in self.config.actor/critic must match
        // self.actor.config / self.critic.config. If not, a prior mutation
        // bypassed the API contract. Do not make these fields pub without
        // adding a runtime check.
        debug_assert!(
            self.config.actor.lr_weights == self.actor.config.lr_weights
                && self.config.actor.temperature == self.actor.config.temperature
                && self.config.actor.alpha == self.actor.config.alpha
                && self.config.actor.tol == self.actor.config.tol
                && self.config.actor.min_steps == self.actor.config.min_steps
                && self.config.actor.max_steps == self.actor.config.max_steps
                && self.config.actor.local_lambda == self.actor.config.local_lambda
                && self.config.actor.synchronous == self.actor.config.synchronous
                && self.config.critic.lr == self.critic.config.lr
                && self.config.critic.output_activation == self.critic.config.output_activation,
            "BUG: self.config and self.actor/critic.config are out of sync"
        );

        // 1. Validate new config internally
        Self::validate_config(&config)?;

        // 2. Validate topology match
        self.validate_topology_match(&config)?;

        // 3. Recompute derived state
        let (actor_decay_factors, critic_decay_factors, layer_error_ema) =
            Self::compute_decay_factors(&config);
        let trace_len = Self::gae_trace_len(&config);

        // 4. Rebuild hysteresis state machines (DRY via build_hysteresis helper)
        let mut actor_hysteresis = Self::build_hysteresis(
            config.actor_hysteresis,
            config.actor_fast_window,
            config.actor_slow_window,
            config.actor_wake_fraction,
            config.actor_sleep_fraction,
        );
        let mut critic_hysteresis = Self::build_hysteresis(
            config.critic_hysteresis,
            config.critic_fast_window,
            config.critic_slow_window,
            config.critic_wake_fraction,
            config.critic_sleep_fraction,
        );

        // 5. Update min_initial_plastic for Fisher warmup
        let mfp = Self::min_fisher_phase(&config);
        if let Some(ref mut hyst) = actor_hysteresis {
            hyst.min_initial_plastic = std::cmp::max(hyst.min_initial_plastic, mfp);
        }
        if let Some(ref mut hyst) = critic_hysteresis {
            hyst.min_initial_plastic = std::cmp::max(hyst.min_initial_plastic, mfp);
        }

        // 6. Reallocate Fisher state (DRY via build_fisher_for_layers helper)
        let actor_fisher =
            Self::build_fisher_for_layers(&self.backend, &self.actor.layers, config.ewc_lambda);
        let critic_fisher =
            Self::build_fisher_for_layers(&self.backend, &self.critic.layers, config.ewc_lambda);

        // 6b. Reallocate Polyak + Frozen anchor slots on lambda transition.
        // Delegates to the single authoritative allocation site
        // (`allocate_anchor_slots`) so the lambda-based slot-presence
        // invariant matches every other constructor.
        let (polyak_target, frozen_champion) = Self::allocate_anchor_slots(&config, &self.actor);

        // 6d. Replay buffer slot transitions (Phase 2).
        //     0 → positive:  allocate fresh empty buffer.
        //     positive → 0:  deallocate.
        //     positive → positive: keep existing contents when capacity
        //       and filter are unchanged, otherwise reset to a fresh
        //       empty buffer sized per the new config. The reset is
        //       the only safe path — changing capacity mid-flight
        //       would leak FIFO ordering semantics between old and
        //       new sizes.
        //
        // NOTE: use `effective_positive_only` (same override as `new()`) so that
        // a continuous SAC agent rebuilt via apply_config never re-introduces the
        // positive_only=true filter that would discard all Pendulum-v1 transitions.
        let old_training_cap = self.config.replay_training_capacity;
        let new_training_cap = config.replay_training_capacity;
        let effective_po = Self::effective_positive_only(&config);
        let replay_buffer: Option<crate::pc_actor_critic::replay::ReplayBuffer> =
            if old_training_cap == 0 && new_training_cap > 0 {
                Some(crate::pc_actor_critic::replay::ReplayBuffer::new(
                    config.replay_training_capacity,
                    config.replay_recent_capacity,
                    effective_po,
                    config.action_space,
                ))
            } else if old_training_cap > 0 && new_training_cap == 0 {
                None
            } else if old_training_cap > 0 && new_training_cap > 0 {
                // Compare against the effective flag (same override) so a
                // continuous→continuous reconfigure with positive_only toggled
                // still triggers a fresh buffer rather than silently preserving
                // an old one with the wrong filter.
                let capacities_changed = old_training_cap != new_training_cap
                    || self.config.replay_recent_capacity != config.replay_recent_capacity
                    || Self::effective_positive_only(&self.config) != effective_po;
                if capacities_changed {
                    Some(crate::pc_actor_critic::replay::ReplayBuffer::new(
                        config.replay_training_capacity,
                        config.replay_recent_capacity,
                        effective_po,
                        config.action_space,
                    ))
                } else {
                    self.replay_buffer.take()
                }
            } else {
                None
            };

        // 7. Apply all fields atomically
        self.config = config;
        self.surprise_buffer = VecDeque::new();
        self.state_prev = None;
        self.action_prev = None;
        self.action_prev_continuous = None;
        self.infer_prev = None;
        self.valid_actions_prev = None;
        self.actor_hysteresis = actor_hysteresis;
        self.critic_hysteresis = critic_hysteresis;
        self.actor_plastic_step_counter = 0;
        self.critic_plastic_step_counter = 0;
        self.critic_frozen_steps = 0;
        self.actor_frozen_steps = 0;
        self.td_error_buffer = VecDeque::new();
        self.last_td_error = 0.0;
        self.actor_decay_factors = actor_decay_factors;
        self.critic_decay_factors = critic_decay_factors;
        self.layer_error_ema = layer_error_ema;
        self.actor_fisher = actor_fisher;
        self.critic_fisher = critic_fisher;
        self.actor_last_phase_reliable = false;
        self.critic_last_phase_reliable = false;
        self.actor_trace = vec![0.0; trace_len];
        self.polyak_target = polyak_target;
        self.frozen_champion = frozen_champion;
        self.replay_buffer = replay_buffer;
        self.replay_clamp_count = 0;
        self.rollback_hard_cooldown_steps = DEFAULT_ROLLBACK_HARD_COOLDOWN;
        self.steps_since_last_rollback_hard = u64::MAX;
        // SAC twin Q critics are not rebuilt on apply_config — they survive
        // config changes (T13 handles serialization/restore).
        // Leave q1/q2/q1_target/q2_target as-is.

        Ok(())
    }

    /// Creates a new PC Actor-Critic agent.
    ///
    /// # Arguments
    ///
    /// * `config` - Agent configuration with actor, critic, and learning parameters.
    /// * `seed` - Random seed for reproducibility.
    ///
    /// # Errors
    ///
    /// Returns `PcError::ConfigValidation` if any configuration field is invalid
    /// (gamma range, surprise buffer size, scale floor/ceil ordering, hysteresis
    /// fractions, consolidation decay bounds, EWC params, td_steps, gae_lambda).
    pub fn new(backend: L, config: PcActorCriticConfig, seed: u64) -> Result<Self, PcError> {
        Self::validate_config(&config)?;

        let (actor_decay_factors, critic_decay_factors, layer_error_ema) =
            Self::compute_decay_factors(&config);

        // Build hysteresis state machines (DRY via build_hysteresis helper)
        let mut actor_hysteresis = Self::build_hysteresis(
            config.actor_hysteresis,
            config.actor_fast_window,
            config.actor_slow_window,
            config.actor_wake_fraction,
            config.actor_sleep_fraction,
        );
        let mut critic_hysteresis = Self::build_hysteresis(
            config.critic_hysteresis,
            config.critic_fast_window,
            config.critic_slow_window,
            config.critic_wake_fraction,
            config.critic_sleep_fraction,
        );

        // Update min_initial_plastic for Fisher warmup
        let mfp = Self::min_fisher_phase(&config);
        if let Some(ref mut hyst) = actor_hysteresis {
            hyst.min_initial_plastic = std::cmp::max(hyst.min_initial_plastic, mfp);
        }
        if let Some(ref mut hyst) = critic_hysteresis {
            hyst.min_initial_plastic = std::cmp::max(hyst.min_initial_plastic, mfp);
        }

        use rand::SeedableRng;
        let mut rng = StdRng::seed_from_u64(seed);
        let actor = PcActor::<L>::new(backend.clone(), config.actor.clone(), &mut rng)?;
        let critic = MlpCritic::<L>::new(backend.clone(), config.critic.clone(), &mut rng)?;

        // Allocate Fisher state (DRY via build_fisher_for_layers helper)
        let actor_fisher =
            Self::build_fisher_for_layers(&backend, &actor.layers, config.ewc_lambda);
        let critic_fisher =
            Self::build_fisher_for_layers(&backend, &critic.layers, config.ewc_lambda);
        let new_trace_len = Self::gae_trace_len(&config);
        let (polyak_target, frozen_champion) = Self::allocate_anchor_slots(&config, &actor);
        let replay_buffer = if config.replay_training_capacity > 0 {
            // Use `effective_positive_only` (shared with apply_config) so both
            // construction paths apply the same SAC override: continuous SAC
            // forces false regardless of config.replay_positive_only.
            Some(crate::pc_actor_critic::replay::ReplayBuffer::new(
                config.replay_training_capacity,
                config.replay_recent_capacity,
                Self::effective_positive_only(&config),
                config.action_space,
            ))
        } else {
            None
        };

        // SAC twin Q critics + Polyak targets (v6.0.0).
        // Built only in continuous SAC mode (q_critic config present).
        // q1 and q2 get separate rng draws so they initialise differently.
        let (q1, q2, q1_target, q2_target) =
            Self::build_sac_critics(&backend, config.q_critic.clone(), &mut rng)?;

        // Cache log_alpha_init before config is moved into Self.
        let log_alpha_init = config.log_alpha_init;

        Ok(Self {
            actor,
            critic,
            config,
            rng,
            surprise_buffer: VecDeque::new(),
            backend,
            state_prev: None,
            action_prev: None,
            action_prev_continuous: None,
            infer_prev: None,
            valid_actions_prev: None,
            actor_hysteresis,
            critic_hysteresis,
            actor_plastic_step_counter: 0,
            critic_plastic_step_counter: 0,
            critic_frozen_steps: 0,
            actor_frozen_steps: 0,
            td_error_buffer: VecDeque::new(),
            last_td_error: 0.0,
            actor_decay_factors,
            critic_decay_factors,
            layer_error_ema,
            actor_fisher,
            critic_fisher,
            actor_last_phase_reliable: false,
            critic_last_phase_reliable: false,
            actor_trace: vec![0.0; new_trace_len],
            polyak_target,
            frozen_champion,
            rollback_hard_cooldown_steps: DEFAULT_ROLLBACK_HARD_COOLDOWN,
            steps_since_last_rollback_hard: u64::MAX,
            replay_buffer,
            replay_clamp_count: 0,
            log_alpha: log_alpha_init,
            q1,
            q2,
            q1_target,
            q2_target,
            sac_skipped_critic_updates: 0,
            sac_skipped_actor_updates: 0,
        })
    }
    /// Reconstructs an agent from pre-built components (used by serializer).
    ///
    /// # Arguments
    ///
    /// * `config` - Agent configuration.
    /// * `actor` - Pre-built PC actor with loaded weights.
    /// * `critic` - Pre-built MLP critic with loaded weights.
    /// * `rng` - Random number generator.
    pub fn from_parts(
        config: PcActorCriticConfig,
        actor: PcActor<L>,
        critic: MlpCritic<L>,
        rng: StdRng,
        backend: L,
    ) -> Self {
        let (actor_decay_factors, critic_decay_factors, layer_error_ema) =
            Self::compute_decay_factors(&config);
        let parts_trace_len = Self::gae_trace_len(&config);
        let (polyak_target, frozen_champion) = Self::allocate_anchor_slots(&config, &actor);
        Self {
            actor,
            critic,
            config,
            rng,
            surprise_buffer: VecDeque::new(),
            backend,
            state_prev: None,
            action_prev: None,
            action_prev_continuous: None,
            infer_prev: None,
            valid_actions_prev: None,
            actor_hysteresis: None,
            critic_hysteresis: None,
            actor_plastic_step_counter: 0,
            critic_plastic_step_counter: 0,
            critic_frozen_steps: 0,
            actor_frozen_steps: 0,
            td_error_buffer: VecDeque::new(),
            last_td_error: 0.0,
            actor_decay_factors,
            critic_decay_factors,
            layer_error_ema,
            actor_fisher: Vec::new(),
            critic_fisher: Vec::new(),
            actor_last_phase_reliable: false,
            critic_last_phase_reliable: false,
            actor_trace: vec![0.0; parts_trace_len],
            polyak_target,
            frozen_champion,
            rollback_hard_cooldown_steps: DEFAULT_ROLLBACK_HARD_COOLDOWN,
            steps_since_last_rollback_hard: u64::MAX,
            replay_buffer: None,
            replay_clamp_count: 0,
            log_alpha: 0.0,
            q1: None,
            q2: None,
            q1_target: None,
            q2_target: None,
            sac_skipped_critic_updates: 0,
            sac_skipped_actor_updates: 0,
        }
    }

    /// Extracts the continuous learning state for serialization.
    ///
    /// Converts all CL state (hysteresis, Fisher, counters) into
    /// CPU-side serializable types. Returns `None` if no CL features
    /// are active (all defaults).
    pub fn to_cl_state(&self) -> Option<crate::serializer::ClState> {
        use crate::serializer::{
            ClState, EwmaTrackerSerialized, FisherStateSerialized, HysteresisStateSerialized,
        };

        // Build the ClState unconditionally, then compare against the default.
        // This ensures any new CL field that gets a non-default value is
        // automatically detected — no manual OR-chain to extend.
        let serialize_ewma = |t: &EwmaTracker| EwmaTrackerSerialized {
            value: t.value,
            k: t.k,
            window: t.window,
        };

        let serialize_hysteresis = |h: &HysteresisState| HysteresisStateSerialized {
            fast: serialize_ewma(&h.fast),
            slow: serialize_ewma(&h.slow),
            state: h.state.clone(),
            wake_fraction: h.wake_fraction,
            sleep_fraction: h.sleep_fraction,
            min_initial_plastic: h.min_initial_plastic,
        };

        let serialize_fisher = |fs: &FisherState<L>, backend: &L| -> FisherStateSerialized {
            let mat_to_cpu = |m: &L::Matrix| -> crate::matrix::Matrix {
                let rows = backend.mat_rows(m);
                let cols = backend.mat_cols(m);
                let mut cpu = crate::matrix::Matrix::zeros(rows, cols);
                for r in 0..rows {
                    for c in 0..cols {
                        cpu.set(r, c, backend.mat_get(m, r, c));
                    }
                }
                cpu
            };
            FisherStateSerialized {
                f_total_weights: mat_to_cpu(&fs.f_total_weights),
                f_total_bias: backend.vec_to_vec(&fs.f_total_bias),
                f_ema_weights: mat_to_cpu(&fs.f_ema_weights),
                f_ema_bias: backend.vec_to_vec(&fs.f_ema_bias),
                theta_snapshot_weights: fs.theta_snapshot_weights.as_ref().map(mat_to_cpu),
                theta_snapshot_bias: fs
                    .theta_snapshot_bias
                    .as_ref()
                    .map(|v| backend.vec_to_vec(v)),
                theta_snapshot_rezero_alpha: fs.theta_snapshot_rezero_alpha,
                theta_snapshot_skip_proj: fs.theta_snapshot_skip_proj.as_ref().map(mat_to_cpu),
            }
        };

        let cl = ClState {
            actor_hysteresis: self.actor_hysteresis.as_ref().map(serialize_hysteresis),
            critic_hysteresis: self.critic_hysteresis.as_ref().map(serialize_hysteresis),
            actor_plastic_step_counter: self.actor_plastic_step_counter,
            critic_plastic_step_counter: self.critic_plastic_step_counter,
            critic_frozen_steps: self.critic_frozen_steps,
            actor_frozen_steps: self.actor_frozen_steps,
            actor_fisher: self
                .actor_fisher
                .iter()
                .map(|f| serialize_fisher(f, &self.backend))
                .collect(),
            critic_fisher: self
                .critic_fisher
                .iter()
                .map(|f| serialize_fisher(f, &self.backend))
                .collect(),
            actor_last_phase_reliable: self.actor_last_phase_reliable,
            critic_last_phase_reliable: self.critic_last_phase_reliable,
            layer_error_ema: self.layer_error_ema.clone(),
        };

        if cl == ClState::default() {
            None
        } else {
            Some(cl)
        }
    }

    /// Restores continuous learning state from a serialized `ClState`.
    ///
    /// Called after `from_parts()` during deserialization. If `cl_state`
    /// is `None` (legacy JSON), the agent keeps its clean defaults.
    pub fn restore_cl_state(&mut self, cl_state: crate::serializer::ClState) {
        use crate::serializer::{EwmaTrackerSerialized, HysteresisStateSerialized};

        let deserialize_ewma = |t: EwmaTrackerSerialized| -> EwmaTracker {
            EwmaTracker {
                value: t.value,
                k: t.k,
                window: t.window,
            }
        };

        let deserialize_hysteresis = |h: HysteresisStateSerialized| -> HysteresisState {
            HysteresisState {
                fast: deserialize_ewma(h.fast),
                slow: deserialize_ewma(h.slow),
                state: h.state,
                wake_fraction: h.wake_fraction,
                sleep_fraction: h.sleep_fraction,
                min_initial_plastic: h.min_initial_plastic,
            }
        };

        self.actor_hysteresis = cl_state.actor_hysteresis.map(deserialize_hysteresis);
        self.critic_hysteresis = cl_state.critic_hysteresis.map(deserialize_hysteresis);
        self.actor_plastic_step_counter = cl_state.actor_plastic_step_counter;
        self.critic_plastic_step_counter = cl_state.critic_plastic_step_counter;
        self.critic_frozen_steps = cl_state.critic_frozen_steps;
        self.actor_frozen_steps = cl_state.actor_frozen_steps;
        self.actor_last_phase_reliable = cl_state.actor_last_phase_reliable;
        self.critic_last_phase_reliable = cl_state.critic_last_phase_reliable;

        if !cl_state.layer_error_ema.is_empty() {
            self.layer_error_ema = cl_state.layer_error_ema;
        }

        // Restore Fisher state
        let deserialize_fisher_vec = |serialized: Vec<crate::serializer::FisherStateSerialized>,
                                      backend: &L|
         -> Vec<FisherState<L>> {
            serialized
                .into_iter()
                .map(|fs| {
                    let cpu_to_mat = |m: &crate::matrix::Matrix| -> L::Matrix {
                        let rows = m.rows;
                        let cols = m.cols;
                        let mut result = backend.zeros_mat(rows, cols);
                        for r in 0..rows {
                            for c in 0..cols {
                                backend.mat_set(&mut result, r, c, m.get(r, c));
                            }
                        }
                        result
                    };
                    let cpu_to_vec = |v: &[f64]| -> L::Vector { backend.vec_from_slice(v) };

                    FisherState {
                        f_total_weights: cpu_to_mat(&fs.f_total_weights),
                        f_total_bias: cpu_to_vec(&fs.f_total_bias),
                        f_ema_weights: cpu_to_mat(&fs.f_ema_weights),
                        f_ema_bias: cpu_to_vec(&fs.f_ema_bias),
                        theta_snapshot_weights: fs.theta_snapshot_weights.as_ref().map(cpu_to_mat),
                        theta_snapshot_bias: fs.theta_snapshot_bias.as_ref().map(|v| cpu_to_vec(v)),
                        theta_snapshot_rezero_alpha: fs.theta_snapshot_rezero_alpha,
                        theta_snapshot_skip_proj: fs
                            .theta_snapshot_skip_proj
                            .as_ref()
                            .map(cpu_to_mat),
                    }
                })
                .collect()
        };

        if !cl_state.actor_fisher.is_empty() {
            self.actor_fisher = deserialize_fisher_vec(cl_state.actor_fisher, &self.backend);
        }
        if !cl_state.critic_fisher.is_empty() {
            self.critic_fisher = deserialize_fisher_vec(cl_state.critic_fisher, &self.backend);
        }
    }

    /// Runs PC inference without selecting an action or modifying RNG state.
    ///
    /// Use this when you only need the inference result (e.g., for TD(0)
    /// next-state evaluation) without side effects.
    ///
    /// # Arguments
    ///
    /// * `input` - Board state vector.
    ///
    /// # Panics
    ///
    /// Panics if `input.len() != config.actor.input_size`.
    /// Shared post-delta bookkeeping: scale/decay, EWC/Fisher, weight update,
    /// M3b layer error EMA, surprise push, td_error push.
    ///
    /// Called by both GAE and standard learning paths after computing their
    /// respective deltas.
    ///
    /// # Arguments
    ///
    /// * `delta` - Policy gradient delta (already scaled by td_error or trace).
    /// * `infer` - Inference result from `act` at current state.
    /// * `input` - Current state.
    /// * `y_conv_vec` - Converged output logits as host Vec.
    /// * `valid_actions` - Valid actions at current state.
    /// * `action` - Action taken.
    /// * `td_error` - Temporal difference error.
    /// * `loss` - Critic loss to return.
    /// * `mode` - Learning mode. [`LearnMode::Replay`] skips the EWC Fisher
    ///   lifecycle and the td_error buffer push (MAGI R6 W1).
    ///
    /// # Returns
    ///
    /// Critic loss (pass-through).
    #[allow(clippy::too_many_arguments)]
    fn apply_actor_update_and_bookkeeping(
        &mut self,
        delta: &[f64],
        infer: &InferResult<L>,
        input: &[f64],
        y_conv_vec: &[f64],
        valid_actions: &[usize],
        _action: usize,
        td_error: f64,
        loss: f64,
        mode: LearnMode,
    ) -> f64 {
        let is_online = mode == LearnMode::Online;
        let s_scale = self.effective_actor_scale_for_mode(infer.surprise_score, mode);
        let actor_decay = self.effective_actor_decay();

        // KL distillation gradients: inject both Polyak and frozen signals
        // into delta before weight update. Both are additive.
        // Shared skip conditions: actor not frozen, >1 valid action.
        // Replay mode with `scale_floor_replay > 0.0` opts in via
        // `replay_bypasses_hysteresis`, lifting the FROZEN clamp on KL
        // distillation so Polyak/Frozen anchors can contribute.
        let replay_opt_in = !is_online && self.replay_bypasses_hysteresis();
        let skip_kl = valid_actions.len() <= 1 || (self.is_actor_frozen() && !replay_opt_in);

        // KL_polyak gradient
        let mut effective_delta: Vec<f64> = if !skip_kl
            && self.config.distillation_lambda_polyak > 0.0
            && self.polyak_target.is_some()
        {
            let g_kl_full = self.compute_kl_polyak_gradient(input, y_conv_vec, valid_actions);
            let lambda = self.config.distillation_lambda_polyak;
            delta
                .iter()
                .zip(g_kl_full.iter())
                .map(|(&d, &g)| d + lambda * g)
                .collect()
        } else {
            delta.to_vec()
        };

        // KL_frozen gradient: parallel to Polyak but targets the frozen champion.
        // The frozen champion is NEVER updated automatically.
        if !skip_kl
            && self.config.distillation_lambda_frozen > 0.0
            && self.frozen_champion.is_some()
        {
            let g_kl_frozen = self.compute_kl_frozen_gradient(input, y_conv_vec, valid_actions);
            let lambda_f = self.config.distillation_lambda_frozen;
            for (d, &g) in effective_delta.iter_mut().zip(g_kl_frozen.iter()) {
                *d += lambda_f * g;
            }
        }

        let delta = &effective_delta;

        // Fisher EMA accumulation and EWC correction (M4). Gated on Online:
        // off-policy replay batches must not contaminate the Fisher diagonal
        // estimate (MAGI R6 W1). Replay mode falls through to the plain
        // weight update with neither Fisher accumulation nor EWC correction.
        if is_online && self.config.ewc_lambda > 0.0 && !self.actor_fisher.is_empty() {
            // Step 2: Extract per-layer gradients for Fisher EMA (read-only).
            //
            // Continuous-only (v6.0.0): Fisher accumulates the actual gradient
            // direction (a valid Fisher proxy for Gaussian policies). The
            // discrete logits-reversal formulation was removed in v6.0.0.
            let fisher_delta = delta.to_vec();
            self.accumulate_actor_fisher_ema(&fisher_delta, infer, input, s_scale, &actor_decay);

            // EWC correction: capture pre-update weights, update, then correct
            let pre_weights: Vec<L::Matrix> = self
                .actor
                .layers
                .iter()
                .map(|l| l.weights.clone())
                .collect();
            let pre_biases: Vec<L::Vector> =
                self.actor.layers.iter().map(|l| l.bias.clone()).collect();

            self.actor
                .update_weights(delta, infer, input, s_scale, &actor_decay);

            // Apply EWC post-correction per layer
            self.apply_actor_ewc_correction(&pre_weights, &pre_biases, s_scale, &actor_decay);
        } else {
            self.actor
                .update_weights(delta, infer, input, s_scale, &actor_decay);
        }

        // Polyak target update: AFTER actor weights are updated.
        if let Some(ref mut polyak) = self.polyak_target {
            // Gate Polyak EMA on "actor weights actually changed this step"
            // (`s_scale > 0`). Semantically: Polyak tracks the live actor's
            // trajectory, so it should only advance when the actor advances.
            //
            // This gate captures TWO distinct no-update scenarios:
            //   (a) Hysteresis clamp: actor FROZEN → effective_actor_scale
            //       returns scale_floor. When scale_floor == 0.0 (default),
            //       s_scale == 0.0 → gate closes → Polyak preserved.
            //   (b) Organic zero: actor PLASTIC but surprise below surprise_low
            //       → surprise_scale returns scale_floor → s_scale == 0.0 →
            //       gate closes → Polyak preserved.
            //
            // Both are correct: the actor's weight update is `lr * s_scale *
            // delta`, which is zero when s_scale == 0.0 in either scenario.
            // Polyak tracks changes in actor weights, so no change → no
            // tracking needed.
            //
            // Edge case when consumer overrides scale_floor to a positive
            // value (e.g. scale_floor = 0.1): then s_scale >= 0.1 always,
            // gate is always open, Polyak tracks even under FROZEN — this
            // is the consistent behavior (actor is partially updating at
            // 0.1× rate, Polyak follows at its EMA lag).
            if s_scale > 0.0 {
                let _ = polyak.polyak_update_from(&self.actor, self.config.polyak_tau);
            }
        }

        // Update per-layer prediction error EMA for adaptive consolidation (M3b)
        if self.config.adaptive_consolidation && !self.layer_error_ema.is_empty() {
            let beta = self.config.consolidation_ema_beta;
            for (i, ema) in self.layer_error_ema.iter_mut().enumerate() {
                if i < infer.prediction_errors.len() {
                    let error_vec = &infer.prediction_errors[i];
                    let rms = {
                        let v = self.backend.vec_to_vec(error_vec);
                        let sum_sq: f64 = v.iter().map(|&x| x * x).sum();
                        (sum_sq / v.len().max(1) as f64).sqrt()
                    };
                    *ema = beta * *ema + (1.0 - beta) * rms;
                }
            }
        }

        if self.config.adaptive_surprise {
            self.push_surprise(infer.surprise_score);
        }

        // Online-only: last_td_error and the adaptive critic-scale buffer
        // both feed on-policy telemetry (hysteresis, surprise->LR mapping).
        // Off-policy replay batches must not overwrite or append to them
        // (MAGI R6 W1).
        if is_online {
            self.last_td_error = td_error;
            self.push_td_error(td_error.abs());
        }

        loss
    }

    /// Computes the learning rate scale factor based on surprise score.
    ///
    /// - surprise <= low → 0.1
    /// - surprise >= high → 2.0
    /// - Between → linear interpolation from 0.1 to 2.0
    ///
    /// If adaptive surprise is enabled and the buffer has >= 10 entries,
    /// thresholds are dynamically recomputed from the buffer statistics.
    pub fn surprise_scale(&self, surprise: f64) -> f64 {
        let (low, high) = if self.config.adaptive_surprise && self.surprise_buffer.len() >= 10 {
            let mean = self.surprise_buffer.iter().sum::<f64>() / self.surprise_buffer.len() as f64;
            let variance = self
                .surprise_buffer
                .iter()
                .map(|&s| (s - mean) * (s - mean))
                .sum::<f64>()
                / self.surprise_buffer.len() as f64;
            let std = variance.sqrt();
            let lo = (mean - 0.5 * std).max(0.0);
            let hi = mean + 1.5 * std;
            (lo, hi)
        } else {
            (self.config.surprise_low, self.config.surprise_high)
        };

        if surprise <= low {
            self.config.scale_floor
        } else if surprise >= high {
            self.config.scale_ceil
        } else {
            // Linear interpolation
            let t = (surprise - low) / (high - low);
            self.config.scale_floor + t * (self.config.scale_ceil - self.config.scale_floor)
        }
    }

    /// v4.0.0 — Continuous-mode training step.
    ///
    /// The canonical SAC training step for `ActionSpace::Continuous`:
    ///
    /// 1. Runs actor inference on the current `state` to obtain `μ(s)`.
    /// 2. Samples `a = tanh(μ + σ·ε)` with `ε ~ N(0, I)` via Box-Muller from
    ///    the agent's deterministic [`StdRng`] (so a fixed seed yields
    ///    a fixed action sequence). `σ` is the actor's LEARNED per-state standard
    ///    deviation (emitted by the actor's log_σ head); `policy_sigma` is ignored
    ///    in continuous SAC mode.
    /// 3. If a previous transition is stored from the prior call, runs
    ///    a TD(0) update via the internal continuous learning path with
    ///    `StepAction::Continuous` — exercising the Gaussian-policy
    ///    gradient `δ_j = td_error · (μ_j − a_j)/σ²`.
    /// 4. Records `(state, action, reward, next_state, done)` in the
    ///    replay buffer when configured.
    /// 5. Updates `state_prev / action_prev_continuous / infer_prev`
    ///    so the next call closes the TD bootstrap.
    /// 6. On terminal, clears all transient state to start a fresh
    ///    episode on the next call.
    ///
    /// # Arguments
    ///
    /// * `state` — current observation vector.
    /// * `reward` — reward received from the environment after the
    ///   previous action (ignored on the first call of an episode).
    /// * `done` — whether the current state is terminal.
    ///
    /// # Returns
    ///
    /// The tanh-squashed action vector `a = tanh(μ(s) + σ·ε) ∈ (−1, 1)` of
    /// length `config.actor.output_size`. The pre-squash value `a_raw =
    /// μ(s) + σ·ε` is stored internally so that the next call's Gaussian-policy
    /// gradient `(μ − a_raw)/σ²` remains correct.
    ///
    /// # Determinism
    ///
    /// Sampling uses the agent's internal `StdRng`. Same seed at
    /// construction → identical action sequence under identical inputs.
    ///
    /// # Errors
    ///
    /// Returns [`PcError::ConfigValidation`] if
    /// `config.action_space != Continuous`.
    ///
    /// # See also
    ///
    /// - [`step_continuous_raw_device`](Self::step_continuous_raw_device)
    ///   — same flow, returns `L::Vector` (forward-compat for GPU).
    /// - [`act_continuous`](Self::act_continuous) — inference-only with
    ///   Play/Training mode selector.
    pub fn step_continuous(
        &mut self,
        state: &[f64],
        reward: f64,
        done: bool,
    ) -> Result<Vec<f64>, PcError> {
        if self.config.action_space != ActionSpace::Continuous {
            return Err(PcError::ConfigValidation(format!(
                "step_continuous is only valid when action_space == Continuous; \
                 current action_space = {:?}. Use step_masked() for discrete \
                 action spaces.",
                self.config.action_space
            )));
        }

        // 1. Actor inference at current state. Snapshot y_conv as a host Vec
        //    so the borrow on `self.actor` ends before we need `&mut self.rng`.
        let current_infer = self.actor.infer(state);

        // 2. SAC off-policy update: push previous transition into the replay
        //    buffer, then trigger sac_learn_step (no-op until warmup reached).
        //    The on-policy V-critic (learn_continuous_inner) is NOT called here;
        //    SAC uses the twin Q-critics exclusively for value estimation.
        if let (Some(prev_state), Some(prev_action)) =
            (self.state_prev.take(), self.action_prev_continuous.take())
        {
            // Drop prev_infer — SAC doesn't use it for on-policy updates.
            let _ = self.infer_prev.take();

            let prev_state_vec = self.backend.vec_to_vec(&prev_state);

            // Push transition to replay buffer: action stored as pre-squash a_raw.
            let transition = crate::pc_actor_critic::replay::ReplayTransition {
                state: prev_state_vec,
                action: crate::pc_actor_critic::replay::Action::Continuous(prev_action),
                reward,
                next_state: state.to_vec(),
                done,
                valid_actions: None,
            };
            if let Some(ref mut buffer) = self.replay_buffer {
                let _ = buffer.push(transition);
            }

            // Off-policy SAC update (no-op until buffer >= batch_size).
            self.sac_learn_step();
        }

        // 3. Sample action using the SAC reparameterized dual-head.
        //    Actor output = [μ_raw | log_σ_raw] (length 2 * action_dim).
        //    a_raw = μ_raw + σ·ε  (pre-squash, stored for next-step replay).
        //    returned action = tanh(a_raw) ∈ (−1, 1).
        //
        //    Warmup override: while the replay buffer has fewer transitions
        //    than `learning_starts`, replace the policy action with a UNIFORM
        //    random action in the squashed space — `a_j ~ Uniform(−0.999, 0.999)`
        //    — and derive the pre-squash value via `a_raw_j = atanh(a_j)`.
        //    Sampling in the squashed space (then atanh) keeps coverage uniform
        //    in action space; sampling a_raw uniformly would cluster tanh(a_raw)
        //    at ±1.  Play mode is unaffected (warmup is Training-only).
        let y_conv = self.backend.vec_to_vec(&current_infer.y_conv);
        let action_dim = self
            .config
            .q_critic
            .as_ref()
            .map(|q| q.action_dim)
            .unwrap_or(y_conv.len());

        let buf_len = self
            .replay_buffer
            .as_ref()
            .map(|b| b.total_len())
            .unwrap_or(0);
        let (a_raw, squashed) =
            if self.config.learning_starts > 0 && buf_len < self.config.learning_starts {
                // Warmup path: uniform random in the squashed space.
                sample_uniform_squashed_action(action_dim, &mut self.rng)
            } else {
                // Normal path: sample from the policy network.
                let (mu, log_sigma) = split_mu_log_sigma(&y_conv, action_dim);
                sample_squashed_action(&mu, &log_sigma, &mut self.rng)
            };

        // 4. Stash (state, a_raw) for the next call's replay push.
        //    `action` is a_raw = μ_raw + σ·ε (pre-squash); the squashed
        //    version is returned to the caller / environment.
        self.state_prev = Some(self.backend.vec_from_slice(state));
        self.action_prev_continuous = Some(a_raw);
        self.infer_prev = Some(current_infer);
        // `valid_actions_prev` is discrete-only; clear to avoid stale state.
        self.valid_actions_prev = None;
        self.action_prev = None;

        // 5. Terminal: drop all transient state.
        if done {
            self.state_prev = None;
            self.action_prev = None;
            self.action_prev_continuous = None;
            self.infer_prev = None;
            self.valid_actions_prev = None;
            for v in &mut self.actor_trace {
                *v = 0.0;
            }
        }

        Ok(squashed)
    }

    /// v4.0.0 — same as [`step_continuous`](Self::step_continuous) but returns
    /// the device-native action vector. The returned value is the tanh-squashed
    /// action `tanh(μ_raw + σ·ε) ∈ (−1, 1)`, identical to `step_continuous`.
    /// Forward-compat hook for future `GpuLinAlg` backends. On `CpuLinAlg`
    /// (where `Vector = Vec<f64>`), this is bit-equivalent to `step_continuous`
    /// plus a `vec_from_slice` round-trip; future `GpuLinAlg` can override to
    /// be zero-copy device-side.
    ///
    /// **Precondition:** `config.action_space == ActionSpace::Continuous`.
    ///
    /// # Errors
    ///
    /// Returns [`PcError::ConfigValidation`] if
    /// `config.action_space != Continuous` — propagated from `step_continuous`.
    pub fn step_continuous_raw_device(
        &mut self,
        state: &[f64],
        reward: f64,
        done: bool,
    ) -> Result<L::Vector, PcError> {
        let action = self.step_continuous(state, reward, done)?;
        Ok(self.backend.vec_from_slice(&action))
    }

    /// v6.0.0 — Continuous-mode inference with `SelectionMode` control (SAC dual-head).
    ///
    /// Runs actor inference to obtain the raw actor output `[μ_raw | log_σ_raw]`
    /// (length `2 * action_dim`) for the current state, then produces an action
    /// according to the caller's mode:
    ///
    /// | `mode` | Action returned | Side-effect on RNG |
    /// |---|---|---|
    /// | `SelectionMode::Play` | Deterministic `tanh(μ_raw)` — no noise | None (RNG not advanced) |
    /// | `SelectionMode::Training` | `tanh(μ_raw + σ·ε)`, `ε ~ N(0, I)` reparameterized | One draw per action dimension |
    ///
    /// `σ = exp(log_σ_raw)` (learned per action dimension, from the actor dual-head).
    ///
    /// This is the inference-only counterpart of
    /// [`step_continuous`](Self::step_continuous): it does not perform a
    /// learning update or modify any stored state. Use it for evaluation
    /// rollouts or action collection inside a policy-gradient loop.
    ///
    /// **Determinism:** Under `Training` mode, the samples come from the agent's
    /// internal `StdRng`. The same seed at construction combined with the same
    /// sequence of calls yields identical actions.
    ///
    /// **Precondition:** `config.action_space == ActionSpace::Continuous`.
    /// `q_critic` is guaranteed `Some` for all continuous agents (enforced at
    /// construction by `validate_config`).
    ///
    /// # Arguments
    ///
    /// * `state` — current observation vector of length `actor.input_size`.
    /// * `mode` — `Play` for deterministic μ; `Training` for stochastic sample.
    ///
    /// # Returns
    ///
    /// A `(action, infer_result)` pair where `action` is a `Vec<f64>` of
    /// length `actor.output_size` and `infer_result` carries the full PC
    /// inference state (surprise score, activations, etc.).
    ///
    /// # Errors
    ///
    /// Returns [`PcError::ConfigValidation`] if
    /// `config.action_space != Continuous`.
    ///
    /// # See also
    ///
    /// - [`step_continuous`](Self::step_continuous) — learning step that
    ///   also samples an action and performs a SAC update.
    pub fn act_continuous(
        &mut self,
        state: &[f64],
        mode: crate::pc_actor::SelectionMode,
    ) -> Result<(Vec<f64>, InferResult<L>), PcError> {
        if self.config.action_space != ActionSpace::Continuous {
            return Err(PcError::ConfigValidation(format!(
                "act_continuous is only valid when action_space == Continuous; \
                 current action_space = {:?}. Use act() for discrete action spaces.",
                self.config.action_space
            )));
        }

        let infer = self.actor.infer(state);
        let y_conv = self.backend.vec_to_vec(&infer.y_conv);

        // Continuous mode is canonical SAC (v6.0.0): q_critic is guaranteed Some by
        // validate_config at construction. The actor output is [μ_raw | log_σ_raw].
        let action_dim = self
            .config
            .q_critic
            .as_ref()
            .expect("continuous agent always has q_critic (enforced at construction)")
            .action_dim;
        let (mu, log_sigma) = split_mu_log_sigma(&y_conv, action_dim);
        let action = match mode {
            crate::pc_actor::SelectionMode::Play => {
                // Deterministic: tanh(μ_raw), no RNG advance.
                deterministic_squashed_action(&mu)
            }
            crate::pc_actor::SelectionMode::Training => {
                // Reparameterized sample: tanh(μ_raw + exp(log_σ)·ε).
                let (_, a) = sample_squashed_action(&mu, &log_sigma, &mut self.rng);
                a
            }
        };

        Ok((action, infer))
    }

    /// Clears step-level internal state without affecting weights or learning state.
    ///
    /// After calling this method, the next `step()` or `step_masked()` call
    /// behaves as the first call of a new episode (skips learning).
    ///
    /// Does NOT modify: weights, surprise buffer, or any continuous learning state.
    pub fn reset_step(&mut self) {
        self.state_prev = None;
        self.action_prev = None;
        self.action_prev_continuous = None;
        self.infer_prev = None;
        self.valid_actions_prev = None;
        for v in &mut self.actor_trace {
            *v = 0.0;
        }
    }

    /// Pushes a surprise score into the adaptive buffer (circular).
    /// Non-finite values are silently dropped to prevent buffer corruption.
    fn push_surprise(&mut self, surprise: f64) {
        if !surprise.is_finite() {
            return;
        }
        if self.surprise_buffer.len() >= self.config.surprise_buffer_size {
            self.surprise_buffer.pop_front();
        }
        self.surprise_buffer.push_back(surprise);
    }

    /// Pushes a |TD error| into the critic adaptive buffer (circular).
    /// Non-finite values are silently dropped to prevent buffer corruption.
    fn push_td_error(&mut self, td_error: f64) {
        if !td_error.is_finite() {
            return;
        }
        if self.td_error_buffer.len() >= self.config.surprise_buffer_size {
            self.td_error_buffer.pop_front();
        }
        self.td_error_buffer.push_back(td_error);
    }

    /// Computes the learning rate scale factor for the critic based on |TD error|.
    ///
    /// Identical to [`surprise_scale()`](Self::surprise_scale) but reads from
    /// the `td_error_buffer` for adaptive threshold computation.
    pub fn critic_surprise_scale(&self, td_error: f64) -> f64 {
        let (low, high) = if self.config.adaptive_surprise && self.td_error_buffer.len() >= 10 {
            let mean = self.td_error_buffer.iter().sum::<f64>() / self.td_error_buffer.len() as f64;
            let variance = self
                .td_error_buffer
                .iter()
                .map(|&s| (s - mean) * (s - mean))
                .sum::<f64>()
                / self.td_error_buffer.len() as f64;
            let std = variance.sqrt();
            let lo = (mean - 0.5 * std).max(0.0);
            let hi = mean + 1.5 * std;
            (lo, hi)
        } else {
            (self.config.surprise_low, self.config.surprise_high)
        };

        if td_error <= low {
            self.config.scale_floor
        } else if td_error >= high {
            self.config.scale_ceil
        } else {
            let t = (td_error - low) / (high - low);
            self.config.scale_floor + t * (self.config.scale_ceil - self.config.scale_floor)
        }
    }

    /// Effective actor learning-rate scale for a given `LearnMode`.
    ///
    /// Sibling of
    /// [`effective_critic_scale_for_mode`](Self::effective_critic_scale_for_mode)
    /// — both methods implement the same FROZEN/Online vs FROZEN/Replay
    /// gate semantics, with the actor reading `scale_floor_replay` and
    /// the critic reading `critic_floor_replay`. Keep the two in lockstep
    /// when modifying gate behaviour to preserve actor-critic symmetry.
    ///
    /// - In `Online` mode: identical to `effective_actor_scale` — hysteresis
    ///   clamps to `scale_floor` when FROZEN.
    /// - In `Replay` mode: when `scale_floor_replay >= 0.0`, the FROZEN
    ///   clamp uses that custom floor instead of `scale_floor`. Only a
    ///   value strictly greater than zero (`> 0.0`) constitutes a real
    ///   opt-in: it activates the `skip_kl` bypass so Polyak/Frozen KL
    ///   anchors contribute. A value of exactly `0.0` is accepted for
    ///   documentary purposes (the consumer has explicitly acknowledged
    ///   the knob) but is functionally equivalent to the default sentinel
    ///   under default `scale_floor = 0.0` — no behavior change. The
    ///   default sentinel `-1.0` preserves v2.2.0 behavior (clamp to
    ///   `scale_floor`).
    pub(crate) fn effective_actor_scale_for_mode(&self, surprise: f64, mode: LearnMode) -> f64 {
        // v4.1.0: continuous mode bypasses surprise/td_error → LR modulation
        // (the variance-band throttle hard-locks continuous policy learning).
        // Use the base learning rate (scale 1.0). Discrete unchanged.
        if self.config.action_space == ActionSpace::Continuous {
            return 1.0;
        }
        // Runtime-mutation NaN/Inf escape guard. `config.scale_floor_replay`
        // is `pub`, so a consumer can bypass `validate_config` by writing
        // a non-finite value after construction. This debug_assert catches
        // that path in test/debug builds; release builds proceed safely
        // because downstream weight-update code has its own NaN guards
        // (see `apply_actor_update_and_bookkeeping`).
        debug_assert!(
            self.config.scale_floor_replay.is_finite(),
            "scale_floor_replay became non-finite post-construction: {}",
            self.config.scale_floor_replay
        );
        let is_sentinel = crate::pc_actor_critic::config::is_replay_floor_sentinel(
            self.config.scale_floor_replay,
        );
        let clamp_to = match mode {
            LearnMode::Online => self.config.scale_floor,
            LearnMode::Replay => {
                if is_sentinel {
                    self.config.scale_floor
                } else {
                    self.config.scale_floor_replay
                }
            }
        };
        match &self.actor_hysteresis {
            Some(h) if h.state == PlasticityState::Frozen => clamp_to,
            _ => self.surprise_scale(surprise),
        }
    }

    /// Whether `LearnMode::Replay` should bypass the hysteresis-driven
    /// `skip_kl` gate. True iff `scale_floor_replay > 0.0` — strict
    /// positive trigger to avoid the `0.0` anti-pattern where KL would be
    /// computed but then zeroed by `s_scale=0` (wasted compute, no
    /// behavior change vs default).
    #[inline]
    fn replay_bypasses_hysteresis(&self) -> bool {
        self.config.scale_floor_replay > 0.0
    }
    /// Computes per-hidden-layer decay factors for the actor.
    ///
    /// When `adaptive_consolidation` is true, uses sigmoid of per-layer
    /// error EMA (M3b). Otherwise uses precomputed fixed decay (M3a).
    pub(crate) fn effective_actor_decay(&self) -> Vec<f64> {
        if self.config.adaptive_consolidation {
            self.layer_error_ema
                .iter()
                .map(|&e| {
                    let x = -self.config.consolidation_sigmoid_k
                        * (e - self.config.consolidation_error_threshold);
                    let adaptive_decay = 1.0 / (1.0 + (-x).exp());
                    1.0 - adaptive_decay
                })
                .collect()
        } else {
            self.actor_decay_factors.clone()
        }
    }

    /// Returns `true` if the actor is in FROZEN hysteresis state.
    ///
    /// When hysteresis is disabled (`None`), returns `false` (always plastic).
    fn is_actor_frozen(&self) -> bool {
        matches!(
            &self.actor_hysteresis,
            Some(h) if h.state == PlasticityState::Frozen
        )
    }

    /// Computes the KL divergence gradient from the live actor toward a
    /// target actor, using log-softmax for numerical stability.
    ///
    /// Returns a full-action-space gradient vector where:
    /// - Valid action indices contain `g_kl[i] = π_live[i] * (log π_live[i] - log π_target[i] - KL)`.
    /// - Invalid action indices are zero.
    ///
    /// The gradient points in the direction that increases KL(π_live || π_target),
    /// so the caller adds `+lambda * g_kl` to the policy gradient delta (which
    /// is a descent direction in the minimization convention used here).
    fn compute_kl_gradient(
        &self,
        target: &PcActor<L>,
        input: &[f64],
        y_conv_vec: &[f64],
        valid_actions: &[usize],
    ) -> Vec<f64> {
        let n_actions = y_conv_vec.len();
        let temp = self.actor.config.temperature;

        // Live logits scaled by temperature
        let live_logits: Vec<f64> = valid_actions
            .iter()
            .map(|&i| y_conv_vec[i] / temp)
            .collect();

        // Forward pass through target to get target logits
        let target_infer = target.infer(input);
        let target_y_conv = self.backend.vec_to_vec(&target_infer.y_conv);
        let target_logits: Vec<f64> = valid_actions
            .iter()
            .map(|&i| target_y_conv[i] / temp)
            .collect();

        // log_softmax for live and target (numerically stable)
        let max_live = live_logits
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let max_target = target_logits
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        let lse_live = live_logits
            .iter()
            .map(|&x| (x - max_live).exp())
            .sum::<f64>()
            .ln()
            + max_live;
        let lse_target = target_logits
            .iter()
            .map(|&x| (x - max_target).exp())
            .sum::<f64>()
            .ln()
            + max_target;

        let log_pi_live: Vec<f64> = live_logits.iter().map(|&x| x - lse_live).collect();
        let log_pi_target: Vec<f64> = target_logits.iter().map(|&x| x - lse_target).collect();
        let pi_live: Vec<f64> = log_pi_live.iter().map(|&lp| lp.exp()).collect();

        // KL(π_live || π_target) = Σ_i π_live[i] * (log π_live[i] - log π_target[i])
        let kl_value: f64 = pi_live
            .iter()
            .zip(log_pi_live.iter())
            .zip(log_pi_target.iter())
            .map(|((&p, &lp), &lq)| p * (lp - lq))
            .sum();

        // g_kl[i] = π_live[i] * (log π_live[i] - log π_target[i] - KL)
        let g_kl: Vec<f64> = pi_live
            .iter()
            .zip(log_pi_live.iter())
            .zip(log_pi_target.iter())
            .map(|((&p, &lp), &lq)| p * (lp - lq - kl_value))
            .collect();

        // Scatter back to full action space
        let mut g_kl_full = vec![0.0; n_actions];
        for (idx, &a) in valid_actions.iter().enumerate() {
            g_kl_full[a] = g_kl[idx];
        }

        // NaN/Inf defense-in-depth (MAGI gate A — Caspar W1): if any
        // element is non-finite (e.g., from corrupted logits), fall back
        // to zero gradient rather than propagating NaN into weights.
        if g_kl_full.iter().any(|v| !v.is_finite()) {
            return vec![0.0; n_actions];
        }

        g_kl_full
    }

    /// Computes KL gradient from the live actor toward the Polyak target.
    ///
    /// Convenience wrapper around [`compute_kl_gradient`] for the Polyak slot.
    ///
    /// # Panics
    ///
    /// Panics if `polyak_target` is `None` (caller must check).
    fn compute_kl_polyak_gradient(
        &self,
        input: &[f64],
        y_conv_vec: &[f64],
        valid_actions: &[usize],
    ) -> Vec<f64> {
        let polyak = self
            .polyak_target
            .as_ref()
            .expect("polyak_target must be Some");
        self.compute_kl_gradient(polyak, input, y_conv_vec, valid_actions)
    }

    /// Computes KL gradient from the live actor toward the frozen champion.
    ///
    /// Convenience wrapper around [`compute_kl_gradient`] for the frozen slot.
    ///
    /// # Panics
    ///
    /// Panics if `frozen_champion` is `None` (caller must check).
    fn compute_kl_frozen_gradient(
        &self,
        input: &[f64],
        y_conv_vec: &[f64],
        valid_actions: &[usize],
    ) -> Vec<f64> {
        let frozen = self
            .frozen_champion
            .as_ref()
            .expect("frozen_champion must be Some");
        self.compute_kl_gradient(frozen, input, y_conv_vec, valid_actions)
    }

    /// Accumulates Fisher EMA for actor layers from extracted gradients.
    ///
    /// Extracts per-layer gradients using Approach 1 (activation derivative,
    /// hadamard, clip) and updates F_ema = beta * F_ema + (1-beta) * g_raw².
    fn accumulate_actor_fisher_ema(
        &mut self,
        output_delta: &[f64],
        infer: &InferResult<L>,
        _input: &[f64],
        _surprise_scale: f64,
        _decay_factors: &[f64],
    ) {
        let output_delta_vec = self.backend.vec_from_slice(output_delta);
        let n_hidden = self.actor.config.hidden_layers.len();
        let n_layers = self.actor.layers.len();

        // Output layer gradient extraction
        let output_output = &infer.y_conv;
        let deriv = self
            .backend
            .apply_derivative(output_output, self.actor.layers[n_layers - 1].activation);
        let mut grad = self.backend.vec_hadamard(&output_delta_vec, &deriv);
        self.backend.clip_vec(&mut grad, crate::matrix::GRAD_CLIP);

        // Update F_ema for output layer
        self.update_fisher_ema_layer(n_layers - 1, &grad, true);

        // Propagated delta: W^T * grad (read-only, using current weights)
        let wt = self
            .backend
            .mat_transpose(&self.actor.layers[n_layers - 1].weights);
        let mut bp_delta = self.backend.mat_vec_mul(&wt, &grad);

        // Hidden layers (from top to bottom)
        for i in (0..n_hidden).rev() {
            let layer_output = if self.actor.skip_alpha_index(i).is_some() {
                // Skip-eligible: use tanh_out
                infer.tanh_components[i].as_ref().unwrap()
            } else {
                &infer.hidden_states[i]
            };

            // Blend delta if using hybrid (same logic as update_weights_hybrid)
            let effective_delta = if (self.actor.config.local_lambda - 1.0).abs() < f64::EPSILON {
                bp_delta.clone()
            } else if self.actor.config.local_lambda.abs() < f64::EPSILON {
                let error_idx = n_hidden - 1 - i;
                infer.prediction_errors[error_idx].clone()
            } else {
                let error_idx = n_hidden - 1 - i;
                let pc_error = &infer.prediction_errors[error_idx];
                let bp_scaled = self
                    .backend
                    .vec_scale(&bp_delta, self.actor.config.local_lambda);
                let pc_scaled = self
                    .backend
                    .vec_scale(pc_error, 1.0 - self.actor.config.local_lambda);
                self.backend.vec_add(&bp_scaled, &pc_scaled)
            };

            // Scale by rezero_alpha if skip layer
            let scaled_delta = if let Some(alpha_idx) = self.actor.skip_alpha_index(i) {
                self.backend
                    .vec_scale(&effective_delta, self.actor.rezero_alpha[alpha_idx])
            } else {
                effective_delta.clone()
            };

            // Extract gradient
            let deriv_h = self
                .backend
                .apply_derivative(layer_output, self.actor.layers[i].activation);
            let mut grad_h = self.backend.vec_hadamard(&scaled_delta, &deriv_h);
            self.backend.clip_vec(&mut grad_h, crate::matrix::GRAD_CLIP);

            self.update_fisher_ema_layer(i, &grad_h, true);

            // Propagate delta read-only
            let wt_h = self.backend.mat_transpose(&self.actor.layers[i].weights);
            let propagated = self.backend.mat_vec_mul(&wt_h, &grad_h);

            if let Some(alpha_idx) = self.actor.skip_alpha_index(i) {
                // Skip path propagation (read-only)
                if let Some(ref proj) = self.actor.skip_projections[alpha_idx] {
                    let proj_t = self.backend.mat_transpose(proj);
                    let skip_delta = self.backend.mat_vec_mul(&proj_t, &effective_delta);
                    bp_delta = self.backend.vec_add(&propagated, &skip_delta);
                } else {
                    bp_delta = self.backend.vec_add(&propagated, &effective_delta);
                }
            } else {
                bp_delta = propagated;
            }
        }
    }

    /// Updates Fisher EMA for a single layer from its extracted gradient.
    ///
    /// F_ema = beta * F_ema + (1-beta) * g_raw²
    fn update_fisher_ema_layer(&mut self, layer_idx: usize, grad: &L::Vector, is_actor: bool) {
        let beta = self.config.fisher_ema_beta;
        let fisher = if is_actor {
            &mut self.actor_fisher[layer_idx]
        } else {
            &mut self.critic_fisher[layer_idx]
        };

        // Update bias F_ema
        let bias_len = self.backend.vec_len(&fisher.f_ema_bias);
        for i in 0..bias_len {
            let g = self.backend.vec_get(grad, i);
            let g_sq = g * g;
            let prev = self.backend.vec_get(&fisher.f_ema_bias, i);
            self.backend
                .vec_set(&mut fisher.f_ema_bias, i, beta * prev + (1.0 - beta) * g_sq);
        }

        // For weight F_ema, we need the outer product direction.
        // The gradient w.r.t. weights is outer(grad, input).
        // But we're tracking Fisher per-weight, so F_ema[r][c] = beta * F_ema[r][c] + (1-beta) * (grad[r] * input[c])².
        // However, this is expensive. The spec says "F_ema = beta * F_ema + (1-beta) * g_raw²"
        // where g_raw is the local gradient (not the weight gradient).
        // For Fisher information, we track per-parameter. The weight gradient for w[r][c] = grad[r] * input[c].
        // But the spec uses g_raw (the post-clip gradient vector) squared element-wise.
        // This is a diagonal approximation: F_ema for weights uses grad² broadcasted.
        // Actually re-reading spec: "g_raw = post-clip, pre-scaling gradient"
        // F_ema is per-parameter. For weights: dL/dW[r][c] = grad[r] * input[c].
        // To avoid storing full outer products, we can use the diagonal Fisher approximation:
        // F_ema_w[r][c] = beta * F_ema_w[r][c] + (1-beta) * grad[r]²
        // This is the standard diagonal Fisher for the row dimension.
        let rows = self.backend.mat_rows(&fisher.f_ema_weights);
        let cols = self.backend.mat_cols(&fisher.f_ema_weights);
        for r in 0..rows {
            let g = self.backend.vec_get(grad, r);
            let g_sq = g * g;
            for c in 0..cols {
                let prev = self.backend.mat_get(&fisher.f_ema_weights, r, c);
                self.backend.mat_set(
                    &mut fisher.f_ema_weights,
                    r,
                    c,
                    beta * prev + (1.0 - beta) * g_sq,
                );
            }
        }
    }

    /// Applies EWC post-update correction to actor layers.
    ///
    /// For each layer: W -= effective_lr * ewc_lambda * F_total * (W_pre - snapshot).
    /// Then applies WEIGHT_CLIP.
    fn apply_actor_ewc_correction(
        &mut self,
        pre_weights: &[L::Matrix],
        pre_biases: &[L::Vector],
        surprise_scale: f64,
        decay_factors: &[f64],
    ) {
        let n_hidden = self.actor.config.hidden_layers.len();
        let n_layers = self.actor.layers.len();

        for i in 0..n_layers {
            let fisher = &self.actor_fisher[i];
            let snapshot_w = match &fisher.theta_snapshot_weights {
                Some(s) => s,
                None => continue, // No snapshot yet
            };
            let snapshot_b = match &fisher.theta_snapshot_bias {
                Some(s) => s,
                None => continue,
            };

            // Compute effective_lr for this layer
            let layer_surprise = if i < n_hidden && !decay_factors.is_empty() {
                surprise_scale * decay_factors[i]
            } else {
                surprise_scale
            };
            let effective_lr = self.actor.config.lr_weights * layer_surprise;

            // EWC correction for weights: W -= effective_lr * ewc_lambda * F_total * (W_pre - snapshot)
            let rows = self.backend.mat_rows(&self.actor.layers[i].weights);
            let cols = self.backend.mat_cols(&self.actor.layers[i].weights);
            for r in 0..rows {
                for c in 0..cols {
                    let w_pre = self.backend.mat_get(&pre_weights[i], r, c);
                    let w_snap = self.backend.mat_get(snapshot_w, r, c);
                    let f_total = self.backend.mat_get(&fisher.f_total_weights, r, c);
                    let correction =
                        effective_lr * self.config.ewc_lambda * f_total * (w_pre - w_snap);
                    let w_cur = self.backend.mat_get(&self.actor.layers[i].weights, r, c);
                    let w_new = (w_cur - correction).clamp(-5.0, 5.0); // WEIGHT_CLIP
                    self.backend
                        .mat_set(&mut self.actor.layers[i].weights, r, c, w_new);
                }
            }

            // EWC correction for biases
            let bias_len = self.backend.vec_len(&self.actor.layers[i].bias);
            for j in 0..bias_len {
                let b_pre = self.backend.vec_get(&pre_biases[i], j);
                let b_snap = self.backend.vec_get(snapshot_b, j);
                let f_total = self.backend.vec_get(&fisher.f_total_bias, j);
                let correction = effective_lr * self.config.ewc_lambda * f_total * (b_pre - b_snap);
                let b_cur = self.backend.vec_get(&self.actor.layers[i].bias, j);
                let b_new = (b_cur - correction).clamp(-5.0, 5.0);
                self.backend
                    .vec_set(&mut self.actor.layers[i].bias, j, b_new);
            }
        }
    }

    /// Reset actor-only transient state that accumulates during
    /// learning (eligibility trace, plasticity counters, TD-error
    /// buffer, last TD error). Shared by
    /// [`rollback_soft`](Self::rollback_soft) and
    /// [`rollback_hard`](Self::rollback_hard) — neither of them should
    /// inherit the old live actor's learning bookkeeping after a
    /// weight rewrite.
    pub(crate) fn reset_actor_transient_state(&mut self) {
        self.actor_trace.fill(0.0);
        self.actor_plastic_step_counter = 0;
        self.actor_frozen_steps = 0;
        self.td_error_buffer.clear();
        self.last_td_error = 0.0;
    }

    /// Zero the Fisher EMA (short-horizon running estimate) for every
    /// actor layer. `f_total` and `theta_snapshot` are preserved —
    /// they encode long-horizon parameter importance and the quadratic
    /// penalty anchor, both of which must survive a rollback so EWC
    /// continues to penalise drift from the restored weights. No-op
    /// when EWC is disabled (`ewc_lambda == 0.0`).
    pub(crate) fn clear_actor_fisher_ema(&mut self) {
        if self.config.ewc_lambda <= 0.0 {
            return;
        }
        for fisher in self.actor_fisher.iter_mut() {
            let rows = self.backend.mat_rows(&fisher.f_ema_weights);
            let cols = self.backend.mat_cols(&fisher.f_ema_weights);
            fisher.f_ema_weights = self.backend.zeros_mat(rows, cols);
            let bias_len = self.backend.vec_len(&fisher.f_ema_bias);
            fisher.f_ema_bias = self.backend.zeros_vec(bias_len);
        }
    }

    // ── Replay buffer API (Phase 2 — commit 16) ───────────────────────────
    /// Transition the replay buffer from training-accumulation phase
    /// to stress-recording phase. Further `push` calls route into the
    /// recent compartment (FIFO).
    ///
    /// # Errors
    ///
    /// Returns [`PcError::ConfigValidation`] if no buffer is configured
    /// (i.e. `replay_training_capacity == 0` at construction and no
    /// subsequent `apply_config` has allocated one). The symmetric
    /// behaviour to [`clear_recent_memories`](Self::clear_recent_memories)
    /// surfaces the misconfiguration explicitly instead of silently
    /// succeeding — sealing a non-existent buffer is almost always a
    /// pipeline wiring bug that a consumer wants to observe.
    pub fn seal_replay_training_memories(&mut self) -> Result<(), PcError> {
        let buffer = self.replay_buffer.as_mut().ok_or_else(|| {
            PcError::ConfigValidation(
                "seal_replay_training_memories requires replay_training_capacity > 0 at construction"
                    .to_string(),
            )
        })?;
        buffer.seal_training_memories();
        Ok(())
    }

    /// Clear the recent-compartment (B) memories without touching
    /// training memories (A). `training_phase` is preserved.
    ///
    /// # Errors
    ///
    /// Returns [`PcError::ConfigValidation`] if no buffer is configured
    /// (i.e. `replay_training_capacity == 0` at construction and no
    /// subsequent `apply_config` has allocated one).
    pub fn clear_recent_memories(&mut self) -> Result<(), PcError> {
        let buffer = self.replay_buffer.as_mut().ok_or_else(|| {
            PcError::ConfigValidation(
                "clear_recent_memories requires replay_training_capacity > 0 at construction"
                    .to_string(),
            )
        })?;
        buffer.recent_memories.clear();
        Ok(())
    }

    /// Monotonic count of `replay_learn` iterations in which the
    /// internal TD-error clamp (`±MAX_REPLAY_TD_ERROR`) was binding.
    ///
    /// Exposed as observable telemetry for the self-recovery pipeline
    /// (MAGI R5 W5). The counter only advances when the clamp actually
    /// truncates the raw TD error; it does not count iterations that
    /// pass through the clamp unchanged.
    pub fn replay_clamp_count(&self) -> u64 {
        self.replay_clamp_count
    }

    /// Forward the LIVE Q-critic `q1` at `(state, action)` (test helper only).
    #[cfg(test)]
    pub(crate) fn q1_for_test(&self, state: &[f64], action: &[f64]) -> f64 {
        self.q1
            .as_ref()
            .expect("q1_for_test: q1 must be Some")
            .forward(state, action)
    }

    /// Actor inference on `state` -> `μ_raw` (first `action_dim` of `y_conv`) (test helper only).
    #[cfg(test)]
    pub(crate) fn actor_mu_raw_for_test(&self, state: &[f64]) -> Vec<f64> {
        let action_dim = self
            .config
            .q_critic
            .as_ref()
            .expect("actor_mu_raw_for_test: q_critic must be Some in SAC mode")
            .action_dim;
        let infer = self.actor.infer(state);
        let y_conv = self.backend.vec_to_vec(&infer.y_conv);
        split_mu_log_sigma(&y_conv, action_dim).0
    }

    /// Actor inference on `state` -> clamped `log_σ` (second `action_dim` of `y_conv`) (test helper only).
    #[cfg(test)]
    pub(crate) fn actor_log_sigma_for_test(&self, state: &[f64]) -> Vec<f64> {
        let action_dim = self
            .config
            .q_critic
            .as_ref()
            .expect("actor_log_sigma_for_test: q_critic must be Some in SAC mode")
            .action_dim;
        let infer = self.actor.infer(state);
        let y_conv = self.backend.vec_to_vec(&infer.y_conv);
        split_mu_log_sigma(&y_conv, action_dim).1
    }
}

#[cfg(test)]
mod sac_learning_guards {
    use super::*;
    use crate::activation::Activation;
    use crate::layer::LayerDef;

    fn default_config() -> PcActorCriticConfig {
        PcActorCriticConfig {
            actor: PcActorConfig {
                input_size: 9,
                hidden_layers: vec![LayerDef {
                    size: 18,
                    activation: Activation::Tanh,
                }],
                output_size: 9,
                output_activation: Activation::Tanh,
                alpha: 0.1,
                tol: 0.01,
                min_steps: 1,
                max_steps: 20,
                lr_weights: 0.01,
                synchronous: true,
                temperature: 1.0,
                local_lambda: 1.0,
                residual: false,
                rezero_init: 0.001,
            },
            critic: MlpCriticConfig {
                input_size: 27,
                hidden_layers: vec![LayerDef {
                    size: 36,
                    activation: Activation::Tanh,
                }],
                output_activation: Activation::Linear,
                lr: 0.005,
            },
            gamma: 0.95,
            surprise_low: 0.02,
            surprise_high: 0.15,
            adaptive_surprise: false,
            surprise_buffer_size: 100,
            entropy_coeff: 0.01,
            scale_floor: 0.1, // v2.0.0 compat: existing tests expect 0.1 floor
            scale_ceil: 2.0,
            actor_hysteresis: false,
            actor_fast_window: 20,
            actor_slow_window: 100,
            actor_wake_fraction: 0.5,
            actor_sleep_fraction: 0.3,
            critic_hysteresis: false,
            critic_fast_window: 20,
            critic_slow_window: 100,
            critic_wake_fraction: 0.5,
            critic_sleep_fraction: 0.3,
            actor_wakes_critic: true,
            actor_wakes_critic_threshold: 1000,
            critic_wakes_actor: true,
            critic_wakes_actor_threshold: 1000,
            consolidation_decay: 1.0,
            critic_consolidation_decay: 1.0,
            adaptive_consolidation: false,
            consolidation_ema_beta: 0.99,
            consolidation_sigmoid_k: 10.0,
            consolidation_error_threshold: 0.05,
            ewc_lambda: 0.0,
            fisher_decay: 0.9,
            fisher_ema_beta: 0.99,
            logits_reversal: false,
            td_steps: 0,
            gae_lambda: None,
            distillation_lambda_polyak: 0.0,
            polyak_tau: 0.005,
            distillation_lambda_frozen: 0.0,
            replay_training_capacity: 0,
            replay_recent_capacity: 0,
            replay_positive_only: true,
            replay_batch_size: 64,
            scale_floor_replay: -1.0,
            critic_floor_replay: -1.0,
            action_space: ActionSpace::Discrete,
            policy_sigma: 0.1,
            policy_entropy_coeff: 0.0,
            q_critic: None,
            target_entropy: None,
            log_alpha_init: 0.0,
            alpha_lr: 0.001,
            learning_starts: 0,
        }
    }

    fn continuous_sac_config() -> PcActorCriticConfig {
        let mut cfg = default_config();
        cfg.action_space = ActionSpace::Continuous;
        cfg.actor.output_size = 2; // 2 * action_dim(=1)
        cfg.actor.output_activation = crate::activation::Activation::Linear;
        cfg.policy_sigma = 0.3; // ignored by SAC but must be finite
        cfg.q_critic = Some(crate::q_critic::QCriticConfig {
            state_dim: cfg.actor.input_size, // 9
            action_dim: 1,
            hidden_layers: vec![LayerDef {
                size: 16,
                activation: Activation::Tanh,
            }],
            lr: 0.001,
        });
        cfg.replay_training_capacity = 1000;
        cfg.replay_batch_size = 8;
        cfg.polyak_tau = 0.005;
        cfg.gae_lambda = None;
        cfg.td_steps = 0;
        // Distillation unsupported in continuous mode — keep at 0.
        cfg.distillation_lambda_polyak = 0.0;
        cfg.distillation_lambda_frozen = 0.0;
        cfg
    }

    #[test]
    #[ignore = "slow learning guard (B2)"]
    fn test_sac_critic_learns_to_rank_actions() {
        // On a trivial 1-state task with reward = -(tanh(a_raw) - 0.7)^2 stored
        // in transitions, after many sac_critic_update calls the trained Q must
        // rank a=0.7 above a=-0.7.
        let mut agent =
            PcActorCritic::<CpuLinAlg>::new(CpuLinAlg::new(), continuous_sac_config(), 7).unwrap();
        let s = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
        let make = |a_raw: f64, r: f64| crate::pc_actor_critic::replay::ReplayTransition {
            state: s.clone(),
            action: crate::pc_actor_critic::replay::Action::Continuous(vec![a_raw]),
            reward: r,
            next_state: s.clone(),
            done: true,
            valid_actions: None,
        };
        // Build a batch covering a range of actions with reward = -(tanh(a_raw) - 0.7)^2.
        let batch: Vec<_> = (0..64)
            .map(|i| {
                let a_raw = -3.0 + (i as f64) * (6.0 / 63.0);
                let r = -((a_raw.tanh() - 0.7).powi(2));
                make(a_raw, r)
            })
            .collect();
        for _ in 0..300 {
            agent.sac_critic_update(&batch);
        }
        let q_good = agent.q1_for_test(&s, &[0.7]);
        let q_bad = agent.q1_for_test(&s, &[-0.7]);
        assert!(
            q_good > q_bad,
            "Q should rank good action above bad: {q_good} vs {q_bad}"
        );
    }

    #[test]
    #[ignore = "slow SAC learning guard (B4)"]
    fn test_pathwise_moves_mu_toward_saturated_optimum() {
        use crate::pc_actor_critic::replay::{Action, ReplayTransition};

        // Score-function contrast helper (test-local only):
        // advantage * (mu - a_raw) / σ².  At saturation (large |a_raw|)
        // the advantage is near-zero (reward flat in raw space) so this
        // does NOT reliably push μ toward the boundary.
        fn score_function_mu_delta(mu: f64, a_raw: f64, sigma: f64, advantage: f64) -> f64 {
            advantage * (mu - a_raw) / (sigma * sigma)
        }

        // Fixed probe state (state_dim=9) — the Q-critics see this.
        let s = vec![0.1_f64, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];

        let mut agent =
            PcActorCritic::<CpuLinAlg>::new(CpuLinAlg::new(), continuous_sac_config(), 17).unwrap();

        // ── Phase 1: prime the twin Q-critics to rank higher a_raw better ──
        // Reward = tanh(a_raw); a at boundary (+1) beats a at −1.
        // We pre-train with a range of (a_raw, reward=tanh(a_raw)) pairs
        // so Q(s, tanh(a_raw)) increases with a_raw.
        let q_batch: Vec<ReplayTransition> = (0..64)
            .map(|i| {
                let a_raw = -3.0 + (i as f64) * (6.0 / 63.0);
                let r = a_raw.tanh(); // reward = squashed action
                ReplayTransition {
                    state: s.clone(),
                    action: Action::Continuous(vec![a_raw]),
                    reward: r,
                    next_state: s.clone(),
                    done: true, // γ-masked: y = r
                    valid_actions: None,
                }
            })
            .collect();

        // Train critics enough to capture the monotone Q(s,·) shape.
        // With Fix 3 (critic batch-averaging), each call applies lr/batch_len
        // per transition.  Use more iterations so the effective critic signal
        // is comparable to the pre-fix baseline (property: Q_high > Q_low).
        for _ in 0..4000 {
            agent.sac_critic_update(&q_batch);
        }

        // Confirm critics rank high-a > low-a (sanity check for the phase below).
        let q_high = agent.q1_for_test(&s, &[0.8]);
        let q_low = agent.q1_for_test(&s, &[-0.8]);
        assert!(
            q_high > q_low,
            "critic pre-train sanity: Q(a=+0.8)={q_high} must > Q(a=−0.8)={q_low}"
        );

        // ── Phase 2: run actor updates and assert μ_raw climbs ──
        let mu_before = agent.actor_mu_raw_for_test(&s)[0];

        // Use single-transition batches so each call applies a full-lr step
        // (update_scaled(1/1) == update); property tests direction, not magnitude.
        // 32× smaller effective lr per 32-item batch at 200 iters would not move
        // μ_raw enough to satisfy the contrast threshold — single-sample is cleaner.
        let actor_single_batch: Vec<ReplayTransition> = (0..32)
            .map(|i| {
                let a_raw = -2.0 + (i as f64) * (4.0 / 31.0);
                ReplayTransition {
                    state: s.clone(),
                    action: Action::Continuous(vec![a_raw]),
                    reward: a_raw.tanh(),
                    next_state: s.clone(),
                    done: true,
                    valid_actions: None,
                }
            })
            .collect();

        // 200 iterations × 32 single-sample calls per inner loop =
        // 6400 full-lr actor gradient steps — matches the pre-fix baseline.
        for _ in 0..200 {
            for t in &actor_single_batch {
                agent.sac_actor_update(std::slice::from_ref(t));
            }
        }

        let mu_after = agent.actor_mu_raw_for_test(&s)[0];

        // Pathwise: μ_raw should have increased (Q-gradient pushes toward +∞ optimum).
        assert!(
            mu_after > mu_before,
            "B4: pathwise gradient must push μ_raw upward; before={mu_before:.4}, after={mu_after:.4}"
        );

        // Contrast: at a saturated point (large a_raw), the score-function
        // delta is small because the advantage is flat in raw space.
        let saturated_a_raw = 4.0_f64; // tanh(4) ≈ 0.9993 — deep in saturation
        let advantage_at_saturation = 0.02_f64; // realistic near-zero advantage
        let sf_delta =
            score_function_mu_delta(mu_after, saturated_a_raw, 0.3, advantage_at_saturation);
        // The pathwise moved μ_raw by more than 3× this score-function delta.
        let pathwise_movement = (mu_after - mu_before).abs();
        assert!(
            pathwise_movement > 3.0 * sf_delta.abs(),
            "B4 contrast: pathwise movement {pathwise_movement:.4} must exceed 3× score-fn delta {sf_delta:.4}"
        );
    }

    #[test]
    #[ignore = "slow SAC learning guard (B5)"]
    fn test_mu_raw_stays_bounded_under_sac() {
        const MU_RAW_BOUND: f64 = 50.0;
        const STEPS: usize = 500;

        // Five diverse probe states (state_dim=9).
        let probes: Vec<Vec<f64>> = vec![
            vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            vec![-0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, 0.0],
            vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.3, 0.6, 0.9, -0.3, -0.6, -0.9, 0.1, -0.1, 0.5],
        ];

        let mut agent =
            PcActorCritic::<CpuLinAlg>::new(CpuLinAlg::new(), continuous_sac_config(), 31).unwrap();

        let mut state = probes[0].clone();
        let next_states: Vec<Vec<f64>> =
            probes.iter().cycle().skip(1).take(STEPS).cloned().collect();

        // Drive the SAC loop with a boundary-optimum task (reward = action[0]).
        for (step, next_state) in next_states.iter().enumerate() {
            let done = (step + 1) % 50 == 0;
            let reward = {
                // Reward = deterministic action at current state (approx boundary).
                let mu = agent.actor_mu_raw_for_test(&state);
                mu[0].tanh() // reward at boundary optimum (+1)
            };
            let _ = agent.step_continuous(&state, reward, done);
            if !done {
                state = next_state.clone();
            } else {
                state = probes[0].clone();
            }
        }

        // Measure mean|μ_raw| over all probe states.
        let mean_abs_mu: f64 = probes
            .iter()
            .map(|p| agent.actor_mu_raw_for_test(p)[0].abs())
            .sum::<f64>()
            / probes.len() as f64;

        assert!(
            mean_abs_mu < MU_RAW_BOUND,
            "B5: μ_raw runaway detected under SAC — mean|μ_raw|={mean_abs_mu:.2} >= {MU_RAW_BOUND}; \
             this is a real integration finding"
        );
    }

    #[test]
    #[ignore = "slow SAC learning guard (B7)"]
    fn test_learned_sigma_collapses_as_policy_commits() {
        use crate::pc_actor_critic::replay::{Action, ReplayTransition};

        // Interior optimum: a* = 0.5 → Q = −(a − 0.5)².
        // μ_raw* = atanh(0.5) ≈ 0.549.
        const TARGET: f64 = 0.5;
        const SEED: u64 = 53;

        let s = vec![0.1_f64, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];

        // Build config: NO hidden layers (isolates μ and log_σ weight rows),
        // α = 1.0 frozen (log_alpha_init=0.0, alpha_lr=1e-9 ≈ frozen).
        // Without hidden layers, μ and log_σ weight rows are decoupled: the
        // output weight matrix W has independent rows for μ (row 0) and log_σ
        // (row 1), so delta[0] only updates W[0,:] and delta[1] only W[1,:].
        let mut cfg = continuous_sac_config();
        // Strip hidden layers: actor is a direct input (9) → output (2) linear.
        // With no hidden layers the actor latent_concat = raw state (9 dims).
        cfg.actor.hidden_layers = vec![];
        // The V-critic input_size must match the latent_concat width:
        // latent_concat = [state(9)] = 9 (no hidden activations to concatenate).
        cfg.critic.input_size = 9;
        // α frozen at 1.0: entropy baseline dominates over Q curvature for σ<0.94.
        cfg.log_alpha_init = 0.0; // α₀ = exp(0) = 1.0
        cfg.alpha_lr = 1e-9; // effectively frozen
                             // Smaller Q-critic for speed; the critic only needs to capture Q shape.
        cfg.q_critic = Some(crate::q_critic::QCriticConfig {
            state_dim: 9, // raw state only (no hidden activations in no-hidden actor)
            action_dim: 1,
            hidden_layers: vec![crate::layer::LayerDef {
                size: 16,
                activation: Activation::Tanh,
            }],
            lr: 0.001,
        });
        cfg.replay_training_capacity = 200;
        cfg.replay_batch_size = 16;

        let mut agent = PcActorCritic::<CpuLinAlg>::new(CpuLinAlg::new(), cfg, SEED).unwrap();

        // ── Phase 1: pre-train twin Q-critics on interior-optimum task ──
        //
        // Dense batch: a_squashed ∈ [−0.9, 0.9], reward = −(a − 0.5)².
        // done=true → γ-masked Bellman target = reward (no future Q needed).
        let q_batch: Vec<ReplayTransition> = (0..64_usize)
            .map(|i| {
                let a_squashed = -0.9 + (i as f64) * (1.8 / 63.0);
                let a_raw = a_squashed.atanh();
                let r = -((a_squashed - TARGET) * (a_squashed - TARGET));
                ReplayTransition {
                    state: s.clone(),
                    action: Action::Continuous(vec![a_raw]),
                    reward: r,
                    next_state: s.clone(),
                    done: true,
                    valid_actions: None,
                }
            })
            .collect();

        for _ in 0..800 {
            agent.sac_critic_update(&q_batch);
        }

        // Sanity: Q must score a* = 0.5 above the boundary.
        let q_opt = agent.q1_for_test(&s, &[TARGET]);
        let q_edge = agent.q1_for_test(&s, &[-0.9]);
        assert!(
            q_opt > q_edge,
            "B7 critic sanity: Q(a=0.5)={q_opt:.4} must > Q(a=−0.9)={q_edge:.4}"
        );

        // ── Phase 2: actor-only updates — σ narrows as policy commits ──
        //
        // The net expected descent on log_σ:
        //   E[delta[n+j]] = -alpha + 2*σ²*jac²(a*) = -1.0 + 2σ²*0.5625
        // For σ < 0.94: negative → log_σ descends → σ narrows.
        // No hidden layers → δlog_σ is independent of δμ (separate weight rows).
        let sigma_initial: f64 = agent
            .actor_log_sigma_for_test(&s)
            .iter()
            .map(|ls| ls.exp())
            .sum::<f64>();

        // Actor batch: spread of a_raw values at the probe state so the
        // actor update sees the Q-gradient shape over the interior-optimum task.
        let actor_batch: Vec<ReplayTransition> = (0..32_usize)
            .map(|i| {
                let a_raw = -2.0 + (i as f64) * (4.0 / 31.0);
                let r = -(a_raw.tanh() - TARGET).powi(2);
                ReplayTransition {
                    state: s.clone(),
                    action: Action::Continuous(vec![a_raw]),
                    reward: r,
                    next_state: s.clone(),
                    done: true,
                    valid_actions: None,
                }
            })
            .collect();

        for _ in 0..800 {
            agent.sac_actor_update(&actor_batch);
        }

        let sigma_final: f64 = agent
            .actor_log_sigma_for_test(&s)
            .iter()
            .map(|ls| ls.exp())
            .sum::<f64>();

        let mu_final = agent.actor_mu_raw_for_test(&s)[0];
        let mu_target = TARGET.atanh(); // ≈ 0.549

        assert!(
            sigma_final < sigma_initial,
            "B7: σ must narrow as policy commits to interior optimum \
             (no-hidden-layer actor, α=1.0 frozen); \
             σ_initial={sigma_initial:.4}, σ_final={sigma_final:.4} \
             (μ_final={mu_final:.4}, μ_target={mu_target:.4}); \
             seed={SEED}"
        );
    }

    #[test]
    #[ignore = "slow SAC learning guard (B8)"]
    fn test_auto_temperature_drives_entropy_toward_target() {
        // H_target = −action_dim = −1.0 (continuous_sac_config: action_dim=1).
        // The temperature update: grad = −α·(logp_mean + H_target)
        //   log_alpha -= alpha_lr * grad
        //
        // Case A: logp_mean = 0.0  → entropy = 0  > H_target = −1
        //         logp_mean + H_target = −1 < 0  → grad > 0  → log_alpha falls → α falls.
        // Case B: logp_mean = −5.0 → entropy = 5  (large exploration)
        //         logp_mean + H_target = −6 < 0  → grad > 0  → log_alpha falls → α falls.
        //
        // Wait — the correct direction:
        //   when entropy > H_target (policy too diffuse), α should FALL to sharpen.
        //   when entropy < H_target (policy too sharp),   α should RISE  to explore.
        //
        // Entropy = −logp.  entropy > H_target iff −logp > H_target iff logp < −H_target.
        // H_target = −1 → −H_target = 1.
        //
        // Case A: logp_mean = 0.0  → entropy = 0  < H_target=−1? No, 0 > −1.
        //         entropy > H_target → α should FALL.
        //   grad = −α·(0 + (−1)) = +α > 0 → log_alpha -= α·lr·(+α) → FALLS. Correct.
        //
        // Case B: logp_mean = −5.0 → entropy = 5.  5 > −1 → α should FALL.
        //   grad = −α·(−5 + (−1)) = +6α > 0 → log_alpha FALLS. Correct.
        //
        // Case C: logp_mean = +2.0 → entropy = −2 < −1 → α should RISE.
        //   grad = −α·(2 + (−1)) = −α < 0 → log_alpha RISES. Correct.

        let mut agent =
            PcActorCritic::<CpuLinAlg>::new(CpuLinAlg::new(), continuous_sac_config(), 79).unwrap();

        // ── Case A: entropy > H_target → α should FALL ──
        let alpha_initial = agent.alpha_for_test();
        // logp_mean = 0 → entropy = 0 > H_target=−1: α should fall.
        for _ in 0..200 {
            agent.sac_temperature_update(0.0);
        }
        let alpha_after_fall = agent.alpha_for_test();
        assert!(
            alpha_after_fall < alpha_initial,
            "B8 case A: when entropy > H_target, α must fall; \
             α_initial={alpha_initial:.6}, α_after_fall={alpha_after_fall:.6}"
        );

        // Reset log_alpha to 0.0 (α=1.0) for case B.
        // (Direct field access not available; rebuild the agent.)
        let mut agent2 =
            PcActorCritic::<CpuLinAlg>::new(CpuLinAlg::new(), continuous_sac_config(), 79).unwrap();
        let alpha_initial2 = agent2.alpha_for_test();

        // ── Case C: entropy < H_target → α should RISE ──
        // logp_mean = +2.0 → entropy = −2 < H_target=−1: α should rise.
        for _ in 0..200 {
            agent2.sac_temperature_update(2.0);
        }
        let alpha_after_rise = agent2.alpha_for_test();
        assert!(
            alpha_after_rise > alpha_initial2,
            "B8 case C: when entropy < H_target, α must rise; \
             α_initial={alpha_initial2:.6}, α_after_rise={alpha_after_rise:.6}"
        );
    }
}
