// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-05-24

//! Action-value critic `Q(s, a)` for canonical SAC (v6.0.0). Input is
//! `state ⊕ action`, output a scalar action-value (Linear). Distinct from
//! `MlpCritic` (the discrete V-critic). `action_gradient` (∇_a Q) is added in T3.
//!
//! # Examples
//!
//! ```
//! use pc_rl_core::activation::Activation;
//! use pc_rl_core::layer::LayerDef;
//! use pc_rl_core::linalg::cpu::CpuLinAlg;
//! use pc_rl_core::q_critic::{QCritic, QCriticConfig};
//! use rand::{rngs::StdRng, SeedableRng};
//!
//! let mut rng = StdRng::seed_from_u64(42);
//! let cfg = QCriticConfig {
//!     state_dim: 3,
//!     action_dim: 2,
//!     hidden_layers: vec![LayerDef { size: 16, activation: Activation::Tanh }],
//!     lr: 0.001,
//! };
//! let q: QCritic = QCritic::new(CpuLinAlg::new(), cfg, &mut rng).unwrap();
//! let value = q.forward(&[0.1, 0.2, 0.3], &[0.5, -0.5]);
//! assert!(value.is_finite());
//! ```

use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::activation::Activation;
use crate::error::PcError;
use crate::layer::{layer_from_cpu, layer_to_cpu, Layer, LayerDef};
use crate::linalg::cpu::CpuLinAlg;
use crate::linalg::LinAlg;

/// Default Q-critic learning rate.
fn default_q_lr() -> f64 {
    0.001
}

/// Configuration for the Q-critic network.
///
/// The network input is the concatenation of `state` (length `state_dim`) and
/// `action` (length `action_dim`), and the output is a single scalar
/// action-value with a [`Activation::Linear`] output activation.
///
/// # Examples
///
/// ```
/// use pc_rl_core::activation::Activation;
/// use pc_rl_core::layer::LayerDef;
/// use pc_rl_core::q_critic::QCriticConfig;
///
/// let cfg = QCriticConfig {
///     state_dim: 3,
///     action_dim: 2,
///     hidden_layers: vec![LayerDef { size: 16, activation: Activation::Tanh }],
///     lr: 0.001,
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QCriticConfig {
    /// Dimensionality of the state observation vector.
    pub state_dim: usize,
    /// Dimensionality of the action vector.
    pub action_dim: usize,
    /// Hidden layer definitions (sizes and activations).
    pub hidden_layers: Vec<LayerDef>,
    /// Learning rate for weight updates. Default: 0.001.
    #[serde(default = "default_q_lr")]
    pub lr: f64,
}

/// Serializable snapshot of Q-critic weights.
///
/// Used for persistence and restoration without requiring an RNG.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct QCriticWeights {
    /// Layer weight snapshots in order (hidden layers + output layer).
    pub layers: Vec<Layer>,
}

/// Action-value critic `Q(s, a)` for canonical SAC.
///
/// The network accepts the concatenation of `state ⊕ action` as input and
/// outputs a single unbounded scalar estimate of `Q(s, a)`. The output layer
/// always uses [`Activation::Linear`].
///
/// Generic over a [`LinAlg`] backend `L`. Defaults to [`CpuLinAlg`].
///
/// # Examples
///
/// ```
/// use pc_rl_core::activation::Activation;
/// use pc_rl_core::layer::LayerDef;
/// use pc_rl_core::linalg::cpu::CpuLinAlg;
/// use pc_rl_core::q_critic::{QCritic, QCriticConfig};
/// use rand::{rngs::StdRng, SeedableRng};
///
/// let mut rng = StdRng::seed_from_u64(42);
/// let cfg = QCriticConfig {
///     state_dim: 3,
///     action_dim: 2,
///     hidden_layers: vec![LayerDef { size: 16, activation: Activation::Tanh }],
///     lr: 0.001,
/// };
/// let mut q: QCritic = QCritic::new(CpuLinAlg::new(), cfg, &mut rng).unwrap();
/// let value = q.forward(&[0.1, 0.2, 0.3], &[0.5, -0.5]);
/// assert!(value.is_finite());
/// let loss = q.update(&[0.1, 0.2, 0.3], &[0.5, -0.5], 1.0);
/// assert!(loss >= 0.0);
/// ```
#[derive(Debug)]
pub struct QCritic<L: LinAlg = CpuLinAlg> {
    /// Dense layers: hidden layers followed by the scalar output layer.
    pub(crate) layers: Vec<Layer<L>>,
    /// Configuration used to build this Q-critic.
    pub config: QCriticConfig,
    /// Backend used for linear algebra operations.
    pub(crate) backend: L,
}

impl<L: LinAlg> QCritic<L> {
    /// Builds the layer chain from the given configuration.
    ///
    /// The input size is `state_dim + action_dim`; the output layer always
    /// has exactly 1 neuron with [`Activation::Linear`].
    ///
    /// # Arguments
    ///
    /// * `backend` - Linear algebra backend.
    /// * `config` - Q-critic topology and hyperparameters.
    /// * `rng` - Random number generator for Xavier weight initialization.
    ///
    /// # Errors
    ///
    /// Returns [`PcError::ConfigValidation`] if `action_dim == 0` or
    /// `state_dim + action_dim == 0`.
    pub fn new(backend: L, config: QCriticConfig, rng: &mut impl Rng) -> Result<Self, PcError> {
        if config.action_dim == 0 {
            return Err(PcError::ConfigValidation(
                "QCritic action_dim must be > 0".into(),
            ));
        }
        let input_size = config.state_dim + config.action_dim;
        if input_size == 0 {
            return Err(PcError::ConfigValidation(
                "QCritic state_dim + action_dim must be > 0".into(),
            ));
        }

        let mut layers = Vec::with_capacity(config.hidden_layers.len() + 1);
        let mut prev = input_size;
        for def in &config.hidden_layers {
            layers.push(Layer::<L>::new(
                prev,
                def.size,
                def.activation,
                &backend,
                rng,
            ));
            prev = def.size;
        }
        // Output layer: 1 neuron, Linear (Q is unbounded)
        layers.push(Layer::<L>::new(prev, 1, Activation::Linear, &backend, rng));

        Ok(Self {
            layers,
            config,
            backend,
        })
    }

    /// Concatenates `state` and `action` into a single input vector.
    fn concat(&self, state: &[f64], action: &[f64]) -> Vec<f64> {
        let mut v = Vec::with_capacity(state.len() + action.len());
        v.extend_from_slice(state);
        v.extend_from_slice(action);
        v
    }

    /// Computes the scalar action-value estimate Q(s, a).
    ///
    /// Sequentially forwards through all layers and returns the single
    /// output neuron's activation.
    ///
    /// # Arguments
    ///
    /// * `state` - State observation vector of length `config.state_dim`.
    /// * `action` - Action vector of length `config.action_dim`.
    pub fn forward(&self, state: &[f64], action: &[f64]) -> f64 {
        let input = self.concat(state, action);
        let mut cur = self.backend.vec_from_slice(&input);
        for layer in &self.layers {
            cur = layer.forward(&cur);
        }
        self.backend.vec_get(&cur, 0)
    }

    /// Forward pass that also returns per-layer inputs and outputs.
    ///
    /// Used internally by [`QCritic::update`] and (in T3) by
    /// `action_gradient` to compute ∇_a Q.
    ///
    /// # Returns
    ///
    /// `(q_value, layer_inputs, layer_outputs)`.
    pub fn forward_with_io(
        &self,
        state: &[f64],
        action: &[f64],
    ) -> (f64, Vec<L::Vector>, Vec<L::Vector>) {
        let input = self.concat(state, action);
        let mut inputs = Vec::with_capacity(self.layers.len());
        let mut outputs = Vec::with_capacity(self.layers.len());
        let mut cur = self.backend.vec_from_slice(&input);
        for layer in &self.layers {
            inputs.push(cur.clone());
            cur = layer.forward(&cur);
            outputs.push(cur.clone());
        }
        (self.backend.vec_get(&cur, 0), inputs, outputs)
    }

    /// Performs one MSE-loss update and returns the loss.
    ///
    /// 1. Forward pass storing each layer's input and output.
    /// 2. Loss = (target − predicted)².
    /// 3. Output gradient: delta = [−2 × (target − predicted)].
    /// 4. Backprop through layers in reverse via [`Layer::backward`].
    /// 5. Returns loss.
    ///
    /// # Arguments
    ///
    /// * `state` - State observation vector.
    /// * `action` - Action vector.
    /// * `target` - Target Q-value (e.g., soft Bellman target).
    pub fn update(&mut self, state: &[f64], action: &[f64], target: f64) -> f64 {
        let (predicted, inputs, outputs) = self.forward_with_io(state, action);
        let error = target - predicted;
        let loss = error * error;

        let mut delta = self.backend.vec_from_slice(&[-2.0 * error]);
        for i in (0..self.layers.len()).rev() {
            delta = self.layers[i].backward(&inputs[i], &outputs[i], &delta, self.config.lr, 1.0);
        }
        loss
    }

    /// Extracts a serializable snapshot of current weights.
    ///
    /// Converts generic layers to CPU layers via `layer_to_cpu` for
    /// backend-agnostic serialization.
    pub fn to_weights(&self) -> QCriticWeights {
        let layers = self
            .layers
            .iter()
            .map(|l| layer_to_cpu(l, &self.backend))
            .collect();
        QCriticWeights { layers }
    }

    /// Restores a Q-critic from saved weights without requiring an RNG.
    ///
    /// Validates that all weight matrix dimensions and bias lengths match
    /// the expected topology, then rebuilds via `layer_from_cpu`.
    ///
    /// # Arguments
    ///
    /// * `backend` - Backend to allocate the restored layers on.
    /// * `config` - Must match the topology used when weights were saved.
    /// * `weights` - Previously saved weight snapshot.
    ///
    /// # Errors
    ///
    /// Returns [`PcError::DimensionMismatch`] if any weight matrix or bias
    /// vector has dimensions inconsistent with the config topology.
    pub fn from_weights(
        backend: L,
        config: QCriticConfig,
        weights: QCriticWeights,
    ) -> Result<Self, PcError> {
        let n_hidden = config.hidden_layers.len();
        let expected_layers = n_hidden + 1;

        if weights.layers.len() != expected_layers {
            return Err(PcError::DimensionMismatch {
                expected: expected_layers,
                got: weights.layers.len(),
                context: "Q-critic layer count",
            });
        }

        let input_size = config.state_dim + config.action_dim;
        let mut prev_size = input_size;
        for (i, cpu_layer) in weights.layers.iter().enumerate() {
            let (expected_rows, expected_cols) = if i < n_hidden {
                (config.hidden_layers[i].size, prev_size)
            } else {
                (1, prev_size) // output layer: 1 neuron
            };

            if cpu_layer.weights.rows != expected_rows {
                return Err(PcError::DimensionMismatch {
                    expected: expected_rows,
                    got: cpu_layer.weights.rows,
                    context: "Q-critic layer weight rows",
                });
            }
            if cpu_layer.weights.cols != expected_cols {
                return Err(PcError::DimensionMismatch {
                    expected: expected_cols,
                    got: cpu_layer.weights.cols,
                    context: "Q-critic layer weight cols",
                });
            }
            if cpu_layer.bias.len() != expected_rows {
                return Err(PcError::DimensionMismatch {
                    expected: expected_rows,
                    got: cpu_layer.bias.len(),
                    context: "Q-critic layer bias length",
                });
            }

            if i < n_hidden {
                prev_size = config.hidden_layers[i].size;
            }
        }

        let layers = weights
            .layers
            .iter()
            .map(|cpu_layer| layer_from_cpu(cpu_layer, &backend))
            .collect();

        Ok(Self {
            layers,
            config,
            backend,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activation::Activation;
    use crate::layer::LayerDef;
    use crate::linalg::cpu::CpuLinAlg;
    use rand::{rngs::StdRng, SeedableRng};

    #[test]
    fn test_action_gradient_matches_finite_difference() {
        let mut rng = StdRng::seed_from_u64(7);
        let cfg = QCriticConfig { state_dim: 2, action_dim: 2,
            hidden_layers: vec![LayerDef { size: 12, activation: Activation::Tanh }], lr: 0.0 };
        let q: QCritic = QCritic::new(CpuLinAlg::new(), cfg, &mut rng).unwrap();
        let s = [0.3, -0.4]; let a = [0.2, -0.1];
        let grad = q.action_gradient(&s, &a);
        assert_eq!(grad.len(), 2);
        let h = 1e-6;
        for j in 0..2 {
            let mut ap = a; let mut am = a;
            ap[j] += h; am[j] -= h;
            let num = (q.forward(&s, &ap) - q.forward(&s, &am)) / (2.0 * h);
            assert!((grad[j] - num).abs() < 1e-4,
                "∇_a Q[{j}] = {} vs finite-diff {num}", grad[j]);
        }
    }

    #[test]
    fn test_qcritic_forward_finite_scalar() {
        let mut rng = StdRng::seed_from_u64(42);
        let cfg = QCriticConfig {
            state_dim: 3,
            action_dim: 1,
            hidden_layers: vec![LayerDef {
                size: 16,
                activation: Activation::Tanh,
            }],
            lr: 0.001,
        };
        let q: QCritic = QCritic::new(CpuLinAlg::new(), cfg, &mut rng).unwrap();
        let v = q.forward(&[0.1, 0.2, 0.3], &[0.5]);
        assert!(v.is_finite());
    }

    #[test]
    fn test_qcritic_update_loss_decreases() {
        let mut rng = StdRng::seed_from_u64(42);
        let cfg = QCriticConfig {
            state_dim: 3,
            action_dim: 1,
            hidden_layers: vec![LayerDef {
                size: 16,
                activation: Activation::Tanh,
            }],
            lr: 0.01,
        };
        let mut q: QCritic = QCritic::new(CpuLinAlg::new(), cfg, &mut rng).unwrap();
        let s = [0.1, 0.2, 0.3];
        let a = [0.5];
        let target = 1.0;
        let l0 = q.update(&s, &a, target);
        let mut lf = l0;
        for _ in 0..29 {
            lf = q.update(&s, &a, target);
        }
        assert!(lf < l0, "Q loss should decrease: {l0} -> {lf}");
    }

    #[test]
    fn test_qcritic_zero_action_dim_rejected() {
        let mut rng = StdRng::seed_from_u64(42);
        let cfg = QCriticConfig {
            state_dim: 3,
            action_dim: 0,
            hidden_layers: vec![],
            lr: 0.001,
        };
        assert!(QCritic::<CpuLinAlg>::new(CpuLinAlg::new(), cfg, &mut rng).is_err());
    }

    #[test]
    fn test_qcritic_weights_roundtrip() {
        let mut rng = StdRng::seed_from_u64(42);
        let cfg = QCriticConfig {
            state_dim: 3,
            action_dim: 1,
            hidden_layers: vec![LayerDef {
                size: 16,
                activation: Activation::Tanh,
            }],
            lr: 0.001,
        };
        let q: QCritic = QCritic::new(CpuLinAlg::new(), cfg.clone(), &mut rng).unwrap();
        let before = q.forward(&[0.1, 0.2, 0.3], &[0.5]);
        let w = q.to_weights();
        let q2: QCritic = QCritic::from_weights(CpuLinAlg::new(), cfg, w).unwrap();
        let after = q2.forward(&[0.1, 0.2, 0.3], &[0.5]);
        assert!((before - after).abs() < 1e-12);
    }
}
