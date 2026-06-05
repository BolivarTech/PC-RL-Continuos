// Author: Julian Bolivar
// Version: 2.0.0
// Date: 2026-05-26

//! Predictive Coding Actor-Critic framework — continuous (SAC) only.
//!
//! Canonical Soft Actor-Critic (SAC) reinforcement learning library where the
//! actor uses iterative top-down/bottom-up predictive coding inference loops
//! and emits the parameters of a tanh-squashed Gaussian policy
//! (`[μ_raw | log_σ_raw]`). A twin Q critic with Polyak-averaged targets,
//! automatic temperature tuning, and a replay buffer drive learning.
//!
//! # Key Components
//!
//! - [`PcActor`] — Predictive coding policy network. Outputs
//!   `2 * action_dim` values split into `μ_raw` and `log_σ_raw` heads.
//! - [`QCritic`] — Twin Q(s, a) critic with backprop-to-input gradient.
//! - [`PcActorCritic`] — Integrated SAC agent: `act_continuous`,
//!   `step_continuous`, automatic temperature, replay, Polyak target updates.
//! - [`serializer`] — JSON weight persistence with checkpointing support.

pub mod activation;
pub mod error;
pub mod layer;
pub mod linalg;
pub mod matrix;
pub(crate) mod mlp_critic;
pub mod pc_actor;
pub mod pc_actor_critic;
pub mod q_critic;
pub mod serializer;

pub use activation::Activation;
pub use error::PcError;
pub use layer::{Layer, LayerDef};
pub use linalg::cpu::CpuLinAlg;
pub use linalg::golub_kahan::{GolubKahanSvd, SvdError};
pub use linalg::LinAlg;
pub use matrix::{rms_error, Matrix, GRAD_CLIP, WEIGHT_CLIP};
pub use pc_actor::{InferResult, PcActor, PcActorConfig, SelectionMode};
pub use pc_actor_critic::{ActivationCache, PcActorCritic, PcActorCriticConfig, TrajectoryStep};
pub use q_critic::{QCritic, QCriticConfig, QCriticWeights};
pub use serializer::{
    checkpoint_filename, load_agent, load_agent_generic, save_agent, save_checkpoint,
    AgentMetadata, PcActorWeights, SaveFile, TrainingMetrics,
};

/// Type alias: CPU-backed layer.
pub type LayerCpu = Layer<CpuLinAlg>;
/// Type alias: CPU-backed PC actor.
pub type PcActorCpu = PcActor<CpuLinAlg>;
/// Type alias: CPU-backed Q-critic.
pub type QCriticCpu = QCritic<CpuLinAlg>;
/// Type alias: CPU-backed PC actor-critic agent.
pub type PcActorCriticCpu = PcActorCritic<CpuLinAlg>;
