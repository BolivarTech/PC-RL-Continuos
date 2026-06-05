// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-03-25

//! JSON-based weight persistence for the PC-Actor-Critic agent.
//!
//! Provides save/load for complete agent state (weights, config, metadata)
//! and checkpoint support with auto-named files.
//!
//! Serialization always goes through CPU types (`CpuLinAlg`). Generic agents
//! convert to/from CPU weights via `to_weights()` / `from_weights()`.

use std::path::{Path, PathBuf};

use chrono::Utc;
use serde::{Deserialize, Serialize};

use crate::error::PcError;
use crate::layer::Layer;
use crate::linalg::LinAlg;
use crate::matrix::Matrix;
use crate::mlp_critic::MlpCritic;
use crate::pc_actor::PcActor;
use crate::pc_actor_critic::{PcActorCritic, PcActorCriticConfig, PlasticityState};

/// Metadata embedded in every save file.
///
/// Tracks version, creation timestamp, episode count, and optional
/// training metrics for provenance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentMetadata {
    /// Crate version string.
    pub version: String,
    /// UTC timestamp of when the file was created.
    pub created: String,
    /// Episode number at time of save.
    pub episode: usize,
    /// Optional training statistics snapshot.
    pub metrics: Option<TrainingMetrics>,
}

/// Training statistics snapshot for inclusion in save files.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMetrics {
    /// Fraction of games won.
    pub win_rate: f64,
    /// Fraction of games lost.
    pub loss_rate: f64,
    /// Fraction of games drawn.
    pub draw_rate: f64,
    /// Average surprise score over recent episodes.
    pub avg_surprise: f64,
    /// Current curriculum depth level.
    pub curriculum_depth: usize,
}

/// Serializable weight snapshot for the PC actor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PcActorWeights {
    /// Layer snapshots in order (hidden layers + output layer).
    pub layers: Vec<Layer>,
    /// ReZero scaling factors for residual skip connections.
    #[serde(default)]
    pub rezero_alpha: Vec<f64>,
    /// Projection matrices for heterogeneous skip connections.
    #[serde(default)]
    pub skip_projections: Vec<Option<crate::matrix::Matrix>>,
}

/// Serializable per-layer Fisher information state.
///
/// Stores accumulated Fisher (`f_total`), current-phase EMA (`f_ema`),
/// and optional weight snapshots as CPU-side `Matrix`/`Vec<f64>`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct FisherStateSerialized {
    /// Accumulated Fisher information for weights.
    pub f_total_weights: Matrix,
    /// Accumulated Fisher information for biases.
    pub f_total_bias: Vec<f64>,
    /// Current-phase running EMA of squared gradients for weights.
    pub f_ema_weights: Matrix,
    /// Current-phase running EMA of squared gradients for biases.
    pub f_ema_bias: Vec<f64>,
    /// Snapshot of weights at last PLASTIC→FROZEN transition.
    #[serde(default)]
    pub theta_snapshot_weights: Option<Matrix>,
    /// Snapshot of biases at last PLASTIC→FROZEN transition.
    #[serde(default)]
    pub theta_snapshot_bias: Option<Vec<f64>>,
    /// Snapshot of rezero alpha (for residual layers).
    #[serde(default)]
    pub theta_snapshot_rezero_alpha: Option<f64>,
    /// Snapshot of skip projection matrix (for heterogeneous residual layers).
    #[serde(default)]
    pub theta_snapshot_skip_proj: Option<Matrix>,
}

/// Serializable EWMA tracker state.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct EwmaTrackerSerialized {
    /// Current EWMA value.
    pub value: f64,
    /// Step counter.
    pub k: u64,
    /// Window size.
    pub window: usize,
}

/// Serializable hysteresis state machine.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HysteresisStateSerialized {
    /// Fast EWMA tracker.
    pub fast: EwmaTrackerSerialized,
    /// Slow EWMA tracker.
    pub slow: EwmaTrackerSerialized,
    /// Current plasticity state.
    pub state: PlasticityState,
    /// Wake fraction threshold.
    pub wake_fraction: f64,
    /// Sleep fraction threshold.
    pub sleep_fraction: f64,
    /// Minimum fast EWMA steps before sleep is allowed.
    pub min_initial_plastic: u64,
}

/// Top-level container for all continuous learning state.
///
/// Persisted in `SaveFile` as `Option<ClState>`. Legacy JSON files
/// without this field load as `None`, which means clean PLASTIC defaults.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ClState {
    /// Actor hysteresis state (None when disabled).
    #[serde(default)]
    pub actor_hysteresis: Option<HysteresisStateSerialized>,
    /// Critic hysteresis state (None when disabled).
    #[serde(default)]
    pub critic_hysteresis: Option<HysteresisStateSerialized>,
    /// Steps the actor has been in PLASTIC state.
    #[serde(default)]
    pub actor_plastic_step_counter: u64,
    /// Steps the critic has been in PLASTIC state.
    #[serde(default)]
    pub critic_plastic_step_counter: u64,
    /// Consecutive steps the critic has been FROZEN.
    #[serde(default)]
    pub critic_frozen_steps: u64,
    /// Consecutive steps the actor has been FROZEN.
    #[serde(default)]
    pub actor_frozen_steps: u64,
    /// Per-layer Fisher state for actor.
    #[serde(default)]
    pub actor_fisher: Vec<FisherStateSerialized>,
    /// Per-layer Fisher state for critic.
    #[serde(default)]
    pub critic_fisher: Vec<FisherStateSerialized>,
    /// Whether the last actor PLASTIC phase was reliable.
    #[serde(default)]
    pub actor_last_phase_reliable: bool,
    /// Whether the last critic PLASTIC phase was reliable.
    #[serde(default)]
    pub critic_last_phase_reliable: bool,
    /// Per-layer prediction error EMA for adaptive consolidation (M3b).
    #[serde(default)]
    pub layer_error_ema: Vec<f64>,
}

/// Complete save file containing agent state and metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SaveFile {
    /// File metadata (version, timestamp, episode).
    pub metadata: AgentMetadata,
    /// Agent configuration.
    pub config: PcActorCriticConfig,
    /// Actor network weights.
    pub actor_weights: PcActorWeights,
    /// Critic network weights.
    ///
    /// `#[serde(default)]` for forward-compatibility: the discrete V-critic
    /// (`MlpCritic`) is dead under SAC and slated for removal in a follow-up;
    /// tolerating an absent `critic_weights` field keeps future SAC-only save
    /// files loadable here without a schema break.
    #[serde(default)]
    pub critic_weights: crate::mlp_critic::MlpCriticWeights,
    /// Continuous learning state (None for legacy/v2.0.0 files).
    #[serde(default)]
    pub cl_state: Option<ClState>,
    /// Polyak-averaged target actor weights (None when lambda == 0 or legacy file).
    #[serde(default)]
    pub polyak_target_weights: Option<PcActorWeights>,
    /// Frozen champion actor weights (None when lambda == 0 or legacy file).
    #[serde(default)]
    pub frozen_champion_weights: Option<PcActorWeights>,
    /// Dual-compartment replay buffer state (None when
    /// `replay_training_capacity == 0` or legacy file).
    #[serde(default)]
    pub replay_buffer: Option<crate::pc_actor_critic::replay::ReplayBuffer>,
    /// SAC twin Q-critic 1 weights (v6.0.0). `None` for discrete agents and
    /// pre-v6 files (serde default). When `config.q_critic` is `Some` on load,
    /// ALL four Q-weight fields must be `Some`; a partial or missing set returns
    /// `Err(PcError::ConfigValidation)`.
    #[serde(default)]
    pub q1_weights: Option<crate::q_critic::QCriticWeights>,
    /// SAC twin Q-critic 2 weights (v6.0.0). See [`q1_weights`](Self::q1_weights).
    #[serde(default)]
    pub q2_weights: Option<crate::q_critic::QCriticWeights>,
    /// Polyak target of Q1 weights (v6.0.0). See [`q1_weights`](Self::q1_weights).
    #[serde(default)]
    pub q1_target_weights: Option<crate::q_critic::QCriticWeights>,
    /// Polyak target of Q2 weights (v6.0.0). See [`q1_weights`](Self::q1_weights).
    #[serde(default)]
    pub q2_target_weights: Option<crate::q_critic::QCriticWeights>,
    /// SAC log-temperature α (v6.0.0). `None` for discrete / pre-v6 files.
    /// When present, restored directly; absent for SAC mode falls back to
    /// `config.log_alpha_init`.
    #[serde(default)]
    pub log_alpha: Option<f64>,
    /// Monotonic count of replay_learn saturation events (legacy files
    /// default to 0).
    #[serde(default)]
    pub replay_clamp_count: u64,
    /// Number of learn steps elapsed since the last `rollback_hard()`.
    /// Legacy files default to `u64::MAX` (the "unlocked" bootstrap
    /// sentinel), which preserves the pre-W2-fix behaviour where a
    /// freshly-loaded agent can always invoke `rollback_hard()` at
    /// least once. New files persist the actual counter so a
    /// save-reload cycle cannot silently bypass the cooldown.
    #[serde(default = "default_steps_since_last_rollback_hard")]
    pub steps_since_last_rollback_hard: u64,
    /// User-configurable cooldown window (defaults to
    /// [`DEFAULT_ROLLBACK_HARD_COOLDOWN`](crate::pc_actor_critic::DEFAULT_ROLLBACK_HARD_COOLDOWN)
    /// when absent, so legacy files and any user override via
    /// `set_rollback_hard_cooldown` both round-trip cleanly).
    #[serde(default = "default_rollback_hard_cooldown_steps")]
    pub rollback_hard_cooldown_steps: u64,
}

fn default_steps_since_last_rollback_hard() -> u64 {
    u64::MAX
}

fn default_rollback_hard_cooldown_steps() -> u64 {
    crate::pc_actor_critic::DEFAULT_ROLLBACK_HARD_COOLDOWN
}

/// Saves the agent's full state to a JSON file.
///
/// Creates parent directories if they don't exist. Extracts weights
/// from both actor and critic via `to_weights()`, bundles with config
/// and metadata, and writes as pretty-printed JSON.
///
/// # Arguments
///
/// * `agent` - The agent to save (any `LinAlg` backend).
/// * `path` - File path for the JSON output.
/// * `episode` - Current episode number.
/// * `metrics` - Optional training metrics snapshot.
///
/// # Errors
///
/// Returns `PcError::Io` on file system errors, `PcError::Serialization`
/// on JSON encoding errors.
pub fn save_agent<L: LinAlg>(
    agent: &PcActorCritic<L>,
    path: &str,
    episode: usize,
    metrics: Option<TrainingMetrics>,
) -> Result<(), PcError> {
    let save_file = SaveFile {
        metadata: AgentMetadata {
            version: env!("CARGO_PKG_VERSION").to_string(),
            created: Utc::now().to_rfc3339(),
            episode,
            metrics,
        },
        config: agent.config.clone(),
        actor_weights: agent.actor.to_weights(),
        critic_weights: agent.critic.to_weights(),
        cl_state: agent.to_cl_state(),
        polyak_target_weights: agent.polyak_target.as_ref().map(|a| a.to_weights()),
        frozen_champion_weights: agent.frozen_champion.as_ref().map(|a| a.to_weights()),
        q1_weights: agent.q1.as_ref().map(|q| q.to_weights()),
        q2_weights: agent.q2.as_ref().map(|q| q.to_weights()),
        q1_target_weights: agent.q1_target.as_ref().map(|q| q.to_weights()),
        q2_target_weights: agent.q2_target.as_ref().map(|q| q.to_weights()),
        log_alpha: if agent.q1.is_some() {
            Some(agent.log_alpha)
        } else {
            None
        },
        replay_buffer: agent.replay_buffer.clone(),
        replay_clamp_count: agent.replay_clamp_count,
        steps_since_last_rollback_hard: agent.steps_since_last_rollback_hard,
        rollback_hard_cooldown_steps: agent.rollback_hard_cooldown_steps,
    };

    let json = serde_json::to_string_pretty(&save_file)?;

    // Create parent directories if needed
    let path = Path::new(path);
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }

    std::fs::write(path, json)?;
    Ok(())
}

/// Loads an agent from a JSON save file (CPU backend).
///
/// Reads the file, deserializes the `SaveFile`, validates that the
/// topology matches the config, then reconstructs the agent using
/// `CpuLinAlg` (the default backend).
///
/// # Arguments
///
/// * `path` - Path to the JSON save file.
///
/// # Errors
///
/// Returns `PcError::Io` if the file doesn't exist, `PcError::Serialization`
/// for invalid JSON, or `PcError::DimensionMismatch` if the saved weights
/// don't match the config topology.
pub fn load_agent(
    path: &str,
    backend: crate::linalg::cpu::CpuLinAlg,
) -> Result<(PcActorCritic, AgentMetadata), PcError> {
    load_agent_generic(path, backend)
}

/// Loads an agent from a JSON save file with a specific `LinAlg` backend.
///
/// Same as [`load_agent`] but reconstructs the agent using the specified
/// backend type `L`. Weights are deserialized as CPU types and then
/// converted via `PcActor::<L>::from_weights()` and
/// `MlpCritic::<L>::from_weights()`.
///
/// # Arguments
///
/// * `path` - Path to the JSON save file.
///
/// # Errors
///
/// Returns `PcError::Io` if the file doesn't exist, `PcError::Serialization`
/// for invalid JSON, or `PcError::DimensionMismatch` if the saved weights
/// don't match the config topology.
pub fn load_agent_generic<L: LinAlg>(
    path: &str,
    backend: L,
) -> Result<(PcActorCritic<L>, AgentMetadata), PcError> {
    let json = std::fs::read_to_string(path)?;
    let save_file: SaveFile = serde_json::from_str(&json)?;

    let actor = PcActor::<L>::from_weights(
        backend.clone(),
        save_file.config.actor.clone(),
        save_file.actor_weights,
    )?;
    let critic = MlpCritic::<L>::from_weights(
        backend.clone(),
        save_file.config.critic.clone(),
        save_file.critic_weights,
    )?;

    use rand::SeedableRng;
    let rng = rand::rngs::StdRng::from_entropy();

    let mut agent = PcActorCritic::from_parts(
        save_file.config.clone(),
        actor,
        critic,
        rng,
        backend.clone(),
    );

    // Restore Polyak target: saved weights > legacy clone > None
    if save_file.config.distillation_lambda_polyak > 0.0 {
        if let Some(polyak_weights) = save_file.polyak_target_weights {
            let polyak = PcActor::<L>::from_weights(
                backend.clone(),
                save_file.config.actor.clone(),
                polyak_weights,
            )?;
            agent.polyak_target = Some(polyak);
        }
        // else: from_parts already cloned actor (legacy compat)
    } else {
        agent.polyak_target = None;
    }

    // Restore frozen champion: saved weights > legacy clone > None
    if save_file.config.distillation_lambda_frozen > 0.0 {
        if let Some(frozen_weights) = save_file.frozen_champion_weights {
            let frozen = PcActor::<L>::from_weights(
                backend.clone(),
                save_file.config.actor.clone(),
                frozen_weights,
            )?;
            agent.frozen_champion = Some(frozen);
        }
        // else: from_parts already cloned actor (legacy compat)
    } else {
        agent.frozen_champion = None;
    }

    if let Some(cl_state) = save_file.cl_state {
        agent.restore_cl_state(cl_state);
    }

    // Restore SAC twin Q-critics and log_alpha (v6.0.0).
    // For SAC mode (q_critic Some in config): ALL four Q-weight fields must be
    // present. A partial or absent set (pre-v6 continuous save) is a hard error —
    // we reject it with ConfigValidation so the caller gets a clean Err, not a
    // panic or silently invalid agent.
    // For discrete mode (q_critic None): all four remain None; log_alpha stays 0.0.

    // Guard: a truly pre-v6 continuous save has action_space=Continuous but
    // no q_critic in config (the field did not exist before v6.0.0).  Without
    // this check the SAC-restore block below is skipped, the agent loads with
    // q1=None, and the first call to act_continuous panics.  Fail cleanly here
    // so the caller receives Err instead of a later panic.
    if save_file.config.action_space == crate::pc_actor_critic::ActionSpace::Continuous
        && save_file.config.q_critic.is_none()
    {
        return Err(PcError::ConfigValidation(
            "pre-v6 continuous save lacks q_critic config; cannot load as a v6 SAC agent. \
             Re-train from scratch with v6.0.0 to obtain a compatible checkpoint."
                .to_string(),
        ));
    }

    if save_file.config.q_critic.is_some() {
        let q_cfg = save_file.config.q_critic.clone().unwrap();
        let q1_w = save_file.q1_weights.ok_or_else(|| {
            PcError::ConfigValidation(
                "SAC save file missing q1_weights: pre-v6 continuous save cannot be loaded \
                 as a v6 SAC agent. Re-train from scratch with v6.0.0 to obtain a \
                 compatible checkpoint."
                    .to_string(),
            )
        })?;
        let q2_w = save_file.q2_weights.ok_or_else(|| {
            PcError::ConfigValidation(
                "SAC save file missing q2_weights: incomplete or corrupt checkpoint.".to_string(),
            )
        })?;
        let q1_target_w = save_file.q1_target_weights.ok_or_else(|| {
            PcError::ConfigValidation(
                "SAC save file missing q1_target_weights: incomplete or corrupt checkpoint."
                    .to_string(),
            )
        })?;
        let q2_target_w = save_file.q2_target_weights.ok_or_else(|| {
            PcError::ConfigValidation(
                "SAC save file missing q2_target_weights: incomplete or corrupt checkpoint."
                    .to_string(),
            )
        })?;

        agent.q1 = Some(crate::q_critic::QCritic::from_weights(
            backend.clone(),
            q_cfg.clone(),
            q1_w,
        )?);
        agent.q2 = Some(crate::q_critic::QCritic::from_weights(
            backend.clone(),
            q_cfg.clone(),
            q2_w,
        )?);
        agent.q1_target = Some(crate::q_critic::QCritic::from_weights(
            backend.clone(),
            q_cfg.clone(),
            q1_target_w,
        )?);
        agent.q2_target = Some(crate::q_critic::QCritic::from_weights(
            backend.clone(), // use clone; `backend` is used above
            q_cfg,
            q2_target_w,
        )?);
        agent.log_alpha = save_file
            .log_alpha
            .unwrap_or(save_file.config.log_alpha_init);
    }

    // Restore replay buffer:
    //   * If the SaveFile carries a `Some(buf)`, use it directly.
    //   * Else if the effective config's `replay_training_capacity > 0`, allocate
    //     a fresh empty buffer (legacy save-file compat — Phase 1 files lack the
    //     `replay_buffer` key).
    //   * Else, no buffer.
    agent.replay_buffer = if let Some(buf) = save_file.replay_buffer {
        Some(buf)
    } else if save_file.config.replay_training_capacity > 0 {
        Some(crate::pc_actor_critic::replay::ReplayBuffer::new(
            save_file.config.replay_training_capacity,
            save_file.config.replay_recent_capacity,
            save_file.config.replay_positive_only,
            save_file.config.action_space,
        ))
    } else {
        None
    };

    // Restore replay telemetry and rollback cooldown state so dashboards
    // and the cooldown gate survive save/load cycles. Legacy files that
    // pre-date these fields deserialize with sensible defaults via
    // `#[serde(default = ...)]`: 0 for the clamp counter, `u64::MAX`
    // (unlocked bootstrap) for the elapsed counter, and
    // `DEFAULT_ROLLBACK_HARD_COOLDOWN` for the cooldown window.
    agent.replay_clamp_count = save_file.replay_clamp_count;
    agent.steps_since_last_rollback_hard = save_file.steps_since_last_rollback_hard;
    agent.rollback_hard_cooldown_steps = save_file.rollback_hard_cooldown_steps;

    Ok((agent, save_file.metadata))
}

/// Generates a checkpoint filename with no colons (filesystem-safe).
///
/// Format: `checkpoint_ep{N}_{YYYYMMDD_HHMMSS}.json`
///
/// # Arguments
///
/// * `episode` - Episode number to embed in the filename.
///
/// # Examples
///
/// ```
/// use pc_rl_continuos::serializer::checkpoint_filename;
///
/// let name = checkpoint_filename(100);
/// assert!(name.starts_with("checkpoint_ep100_"));
/// assert!(name.ends_with(".json"));
/// assert!(!name.contains(':'));
/// ```
pub fn checkpoint_filename(episode: usize) -> String {
    let now = Utc::now().format("%Y%m%d_%H%M%S");
    format!("checkpoint_ep{episode}_{now}.json")
}

/// Saves a checkpoint to a directory with an auto-generated filename.
///
/// # Arguments
///
/// * `agent` - The agent to checkpoint (any `LinAlg` backend).
/// * `dir` - Directory where the checkpoint file will be created.
/// * `episode` - Current episode number.
/// * `metrics` - Optional training metrics snapshot.
///
/// # Returns
///
/// The full path to the created checkpoint file.
///
/// # Errors
///
/// Returns `PcError` on I/O or serialization failures.
pub fn save_checkpoint<L: LinAlg>(
    agent: &PcActorCritic<L>,
    dir: &str,
    episode: usize,
    metrics: Option<TrainingMetrics>,
) -> Result<PathBuf, PcError> {
    let filename = checkpoint_filename(episode);
    let path = Path::new(dir).join(filename);
    let path_str = path.to_string_lossy().to_string();
    save_agent(agent, &path_str, episode, metrics)?;
    Ok(path)
}
