// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-03-25

//! Standard MLP critic (value function) for the PC Actor-Critic agent.
//!
//! Receives the concatenation of board state and actor latent representation,
//! outputs a scalar value estimate. Learns via MSE loss backpropagation.

use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::activation::Activation;
use crate::error::PcError;
use crate::layer::{layer_from_cpu, layer_to_cpu, Layer, LayerDef};
use crate::linalg::cpu::CpuLinAlg;
use crate::linalg::LinAlg;

/// Default critic learning rate.
fn default_critic_lr() -> f64 {
    0.005
}

/// Configuration for the MLP critic network.
///
/// # Examples
///
/// ```text
/// // Internal (pub(crate)) discrete V-critic — illustrative only; dead under
/// // SAC and slated for removal in a follow-up (see CHANGELOG 1.0.0).
/// MlpCriticConfig {
///     input_size: 27,
///     hidden_layers: vec![LayerDef { size: 36, activation: Activation::Tanh }],
///     output_activation: Activation::Linear,
///     lr: 0.005,
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MlpCriticConfig {
    /// Dimensionality of the input vector (board state + latent concat).
    pub input_size: usize,
    /// Hidden layer definitions (sizes and activations).
    pub hidden_layers: Vec<LayerDef>,
    /// Activation for the single-neuron output layer.
    pub output_activation: Activation,
    /// Learning rate for weight updates. Default: 0.005.
    #[serde(default = "default_critic_lr")]
    pub lr: f64,
}

/// Serializable snapshot of critic weights.
///
/// Used by the serializer module to persist and restore the critic
/// without requiring an RNG.
#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct MlpCriticWeights {
    /// Layer weight snapshots in order (hidden layers + output layer).
    pub layers: Vec<Layer>,
}

/// Standard MLP value function (critic).
///
/// Estimates V(s) given the concatenation of board state and actor latent
/// activations. Trained via MSE loss backpropagation through dense layers.
///
/// Generic over a [`LinAlg`] backend `L`. Defaults to [`CpuLinAlg`] for
/// backward compatibility.
///
/// # Examples
///
/// ```text
/// // Internal (pub(crate)) discrete V-critic — illustrative only; dead under
/// // SAC and slated for removal in a follow-up (see CHANGELOG 1.0.0).
/// let config = MlpCriticConfig {
///     input_size: 27,
///     hidden_layers: vec![LayerDef { size: 36, activation: Activation::Tanh }],
///     output_activation: Activation::Linear,
///     lr: 0.005,
/// };
/// let critic = MlpCritic::new(CpuLinAlg::new(), config, &mut rng).unwrap();
/// let value = critic.forward(&vec![0.0; 27]); // V(s)
/// ```
#[derive(Debug)]
pub struct MlpCritic<L: LinAlg = CpuLinAlg> {
    /// Dense layers: hidden layers followed by the output layer (1 neuron).
    pub(crate) layers: Vec<Layer<L>>,
    /// Configuration used to build this critic.
    pub config: MlpCriticConfig,
    /// Backend used for linear algebra operations.
    pub(crate) backend: L,
}

impl<L: LinAlg> MlpCritic<L> {
    /// Builds the layer chain from the given configuration.
    ///
    /// The output layer always has exactly 1 neuron with the configured
    /// `output_activation`.
    ///
    /// # Arguments
    ///
    /// * `config` - Critic topology and hyperparameters.
    /// * `rng` - Random number generator for Xavier weight initialization.
    /// # Errors
    ///
    /// Returns `PcError::ConfigValidation` if `input_size` is zero.
    pub fn new(backend: L, config: MlpCriticConfig, rng: &mut impl Rng) -> Result<Self, PcError> {
        if config.input_size == 0 {
            return Err(PcError::ConfigValidation(
                "critic input_size must be > 0".into(),
            ));
        }

        let mut layers: Vec<Layer<L>> = Vec::with_capacity(config.hidden_layers.len() + 1);
        let mut prev_size = config.input_size;

        for def in &config.hidden_layers {
            layers.push(Layer::<L>::new(
                prev_size,
                def.size,
                def.activation,
                &backend,
                rng,
            ));
            prev_size = def.size;
        }

        // Output layer: 1 neuron
        layers.push(Layer::<L>::new(
            prev_size,
            1,
            config.output_activation,
            &backend,
            rng,
        ));

        Ok(Self {
            layers,
            config,
            backend,
        })
    }

    /// Computes the scalar value estimate V(s).
    ///
    /// Sequentially forwards through all layers and returns the single
    /// output neuron's activation.
    ///
    /// # Panics
    ///
    /// Panics if `input.len() != config.input_size`.
    pub fn forward(&self, input: &[f64]) -> f64 {
        assert_eq!(
            input.len(),
            self.config.input_size,
            "MlpCritic::forward: expected input size {}, got {}",
            self.config.input_size,
            input.len()
        );
        let mut current = self.backend.vec_from_slice(input);
        for layer in &self.layers {
            current = layer.forward(&current);
        }
        self.backend.vec_get(&current, 0)
    }

    /// Computes V(s) and returns both the value and hidden layer activations.
    ///
    /// Identical to [`forward`](Self::forward) but also captures intermediate
    /// hidden layer activations for use in CCA neuron alignment during crossover.
    ///
    /// # Returns
    ///
    /// `(value, hidden_states)` where `hidden_states[i]` is the activation
    /// vector of hidden layer `i` (excludes the output layer).
    ///
    /// # Panics
    ///
    /// Panics if `input.len() != config.input_size`.
    pub fn forward_with_hidden(&self, input: &[f64]) -> (f64, Vec<L::Vector>) {
        assert_eq!(
            input.len(),
            self.config.input_size,
            "MlpCritic::forward_with_hidden: expected input size {}, got {}",
            self.config.input_size,
            input.len()
        );
        let num_hidden = self.config.hidden_layers.len();
        let mut hidden_states = Vec::with_capacity(num_hidden);
        let mut current = self.backend.vec_from_slice(input);
        for (i, layer) in self.layers.iter().enumerate() {
            current = layer.forward(&current);
            if i < num_hidden {
                hidden_states.push(current.clone());
            }
        }
        (self.backend.vec_get(&current, 0), hidden_states)
    }

    /// Performs one MSE-loss update and returns the loss.
    ///
    /// 1. Forward pass storing each layer's input and output.
    /// 2. Loss = (target - predicted)^2.
    /// 3. Output gradient: delta = [-2.0 * (target - predicted)].
    /// 4. Backprop through layers in reverse via `layer.backward(...)`.
    /// 5. Returns loss.
    ///
    /// # Arguments
    ///
    /// * `input` - Concatenated board state + latent activations.
    /// * `target` - Target value (e.g., discounted return).
    pub fn update(&mut self, input: &[f64], target: f64) -> f64 {
        // Forward pass, storing intermediate inputs and outputs
        let mut inputs: Vec<L::Vector> = Vec::with_capacity(self.layers.len());
        let mut outputs: Vec<L::Vector> = Vec::with_capacity(self.layers.len());

        let mut current = self.backend.vec_from_slice(input);
        for layer in &self.layers {
            inputs.push(current.clone());
            current = layer.forward(&current);
            outputs.push(current.clone());
        }

        let predicted = self.backend.vec_get(&current, 0);
        let error = target - predicted;
        let loss = error * error;

        // Output gradient: d(loss)/d(predicted) = -2*(target - predicted)
        let mut delta = self.backend.vec_from_slice(&[-2.0 * error]);

        // Backprop through layers in reverse
        for i in (0..self.layers.len()).rev() {
            delta = self.layers[i].backward(
                &inputs[i],
                &outputs[i],
                &delta,
                self.config.lr,
                1.0, // surprise_scale = 1.0 for critic
            );
        }

        loss
    }

    /// Performs a full training step with per-layer consolidation decay.
    ///
    /// Like [`MlpCritic::update`], but applies `surprise_scale * decay_factors[i]` to
    /// each hidden layer's backward pass. The output layer always receives
    /// the raw `surprise_scale`. Empty `decay_factors` means no per-layer
    /// decay (all layers get `surprise_scale`).
    ///
    /// # Arguments
    ///
    /// * `input` - Raw input vector (state features).
    /// * `target` - Target value for the critic.
    /// * `surprise_scale` - Global learning rate scale from surprise.
    /// * `decay_factors` - Per-hidden-layer consolidation decay factors.
    pub fn update_with_decay(
        &mut self,
        input: &[f64],
        target: f64,
        surprise_scale: f64,
        decay_factors: &[f64],
    ) -> f64 {
        let mut inputs: Vec<L::Vector> = Vec::with_capacity(self.layers.len());
        let mut outputs: Vec<L::Vector> = Vec::with_capacity(self.layers.len());

        let mut current = self.backend.vec_from_slice(input);
        for layer in &self.layers {
            inputs.push(current.clone());
            current = layer.forward(&current);
            outputs.push(current.clone());
        }

        let predicted = self.backend.vec_get(&current, 0);
        let error = target - predicted;
        let loss = error * error;

        let mut delta = self.backend.vec_from_slice(&[-2.0 * error]);
        let n_layers = self.layers.len();
        let n_hidden = if n_layers > 0 { n_layers - 1 } else { 0 };

        for i in (0..n_layers).rev() {
            let layer_scale = if i < n_hidden && !decay_factors.is_empty() {
                surprise_scale * decay_factors[i]
            } else {
                surprise_scale
            };
            delta = self.layers[i].backward(
                &inputs[i],
                &outputs[i],
                &delta,
                self.config.lr,
                layer_scale,
            );
        }

        loss
    }

    /// Extracts a serializable snapshot of current weights.
    ///
    /// Converts generic layers to CPU layers via `layer_to_cpu` for
    /// backend-agnostic serialization.
    pub fn to_weights(&self) -> MlpCriticWeights {
        let layers = self
            .layers
            .iter()
            .map(|l| layer_to_cpu(l, &self.backend))
            .collect();
        MlpCriticWeights { layers }
    }

    /// Restores a critic from saved weights without requiring an RNG.
    ///
    /// Converts CPU layers to generic layers element-by-element for
    /// backend-agnostic restoration. Validates that all weight matrix
    /// dimensions and bias lengths match the expected topology.
    ///
    /// # Arguments
    ///
    /// * `config` - Must match the topology used when weights were saved.
    /// * `weights` - Previously saved weight snapshot.
    ///
    /// # Errors
    ///
    /// Returns `PcError::DimensionMismatch` if any weight matrix or bias
    /// vector has dimensions inconsistent with the config topology.
    pub fn from_weights(
        backend: L,
        config: MlpCriticConfig,
        weights: MlpCriticWeights,
    ) -> Result<Self, PcError> {
        let n_hidden = config.hidden_layers.len();
        let expected_layers = n_hidden + 1;

        if weights.layers.len() != expected_layers {
            return Err(PcError::DimensionMismatch {
                expected: expected_layers,
                got: weights.layers.len(),
                context: "critic layer count",
            });
        }

        let mut prev_size = config.input_size;
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
                    context: "critic layer weight rows",
                });
            }
            if cpu_layer.weights.cols != expected_cols {
                return Err(PcError::DimensionMismatch {
                    expected: expected_cols,
                    got: cpu_layer.weights.cols,
                    context: "critic layer weight cols",
                });
            }
            if cpu_layer.bias.len() != expected_rows {
                return Err(PcError::DimensionMismatch {
                    expected: expected_rows,
                    got: cpu_layer.bias.len(),
                    context: "critic layer bias length",
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
    use crate::activation::Activation;
    use crate::layer::LayerDef;

    use rand::rngs::StdRng;
    use rand::SeedableRng;

    use super::*;

    fn make_rng() -> StdRng {
        StdRng::seed_from_u64(42)
    }

    fn default_config() -> MlpCriticConfig {
        MlpCriticConfig {
            input_size: 27,
            hidden_layers: vec![LayerDef {
                size: 36,
                activation: Activation::Tanh,
            }],
            output_activation: Activation::Linear,
            lr: 0.005,
        }
    }

    // ── forward tests ──────────────────────────────────────────────

    #[test]
    fn test_forward_returns_finite_scalar() {
        let mut rng = make_rng();
        let critic: MlpCritic =
            MlpCritic::new(CpuLinAlg::new(), default_config(), &mut rng).unwrap();
        let input = vec![0.0; 27];
        let v = critic.forward(&input);
        assert!(v.is_finite(), "forward output {v} is not finite");
    }

    #[test]
    fn test_forward_different_inputs_give_different_outputs() {
        let mut rng = make_rng();
        let critic: MlpCritic =
            MlpCritic::new(CpuLinAlg::new(), default_config(), &mut rng).unwrap();
        let a = critic.forward(&vec![0.0; 27]);
        let mut input_b = vec![0.0; 27];
        input_b[0] = 1.0;
        input_b[5] = -1.0;
        let b = critic.forward(&input_b);
        assert!(
            (a - b).abs() > 1e-12,
            "Different inputs should give different outputs: {a} vs {b}"
        );
    }

    #[test]
    fn test_forward_deep_topology_returns_finite() {
        let mut rng = make_rng();
        let config = MlpCriticConfig {
            input_size: 27,
            hidden_layers: vec![
                LayerDef {
                    size: 36,
                    activation: Activation::Tanh,
                },
                LayerDef {
                    size: 18,
                    activation: Activation::Tanh,
                },
            ],
            output_activation: Activation::Linear,
            lr: 0.005,
        };
        let critic: MlpCritic = MlpCritic::new(CpuLinAlg::new(), config, &mut rng).unwrap();
        let v = critic.forward(&vec![0.5; 27]);
        assert!(v.is_finite(), "Deep topology output {v} is not finite");
    }

    #[test]
    fn test_forward_extreme_input_still_finite() {
        let mut rng = make_rng();
        let critic: MlpCritic =
            MlpCritic::new(CpuLinAlg::new(), default_config(), &mut rng).unwrap();
        let input: Vec<f64> = (0..27)
            .map(|i| if i % 2 == 0 { 1e6 } else { -1e6 })
            .collect();
        let v = critic.forward(&input);
        assert!(v.is_finite(), "Extreme input output {v} is not finite");
    }

    #[test]
    #[should_panic]
    fn test_forward_panics_wrong_input_size() {
        let mut rng = make_rng();
        let critic: MlpCritic =
            MlpCritic::new(CpuLinAlg::new(), default_config(), &mut rng).unwrap();
        let _ = critic.forward(&[0.0; 10]); // wrong size
    }

    // ── update tests ───────────────────────────────────────────────

    #[test]
    fn test_update_loss_decreases_over_30_iterations() {
        let mut rng = make_rng();
        let mut critic: MlpCritic =
            MlpCritic::new(CpuLinAlg::new(), default_config(), &mut rng).unwrap();
        let input = vec![0.1; 27];
        let target = 0.5;
        let initial_loss = critic.update(&input, target);
        let mut final_loss = initial_loss;
        for _ in 0..29 {
            final_loss = critic.update(&input, target);
        }
        assert!(
            final_loss < initial_loss,
            "Loss should decrease: initial={initial_loss}, final={final_loss}"
        );
    }

    #[test]
    fn test_update_returns_finite_nonneg_loss() {
        let mut rng = make_rng();
        let mut critic: MlpCritic =
            MlpCritic::new(CpuLinAlg::new(), default_config(), &mut rng).unwrap();
        let loss = critic.update(&vec![0.0; 27], 1.0);
        assert!(loss.is_finite(), "Loss {loss} is not finite");
        assert!(loss >= 0.0, "Loss {loss} is negative");
    }

    #[test]
    fn test_update_changes_weights() {
        let mut rng = make_rng();
        let mut critic: MlpCritic =
            MlpCritic::new(CpuLinAlg::new(), default_config(), &mut rng).unwrap();
        let w_before = critic.layers[0].weights.get(0, 0);
        let _ = critic.update(&vec![0.1; 27], 1.0);
        let w_after = critic.layers[0].weights.get(0, 0);
        assert!(
            (w_before - w_after).abs() > 1e-15,
            "Weights should change after update"
        );
    }

    #[test]
    fn test_update_clips_weights() {
        let mut rng = make_rng();
        let mut critic: MlpCritic =
            MlpCritic::new(CpuLinAlg::new(), default_config(), &mut rng).unwrap();
        // Extreme update to force clipping
        for _ in 0..100 {
            let _ = critic.update(&vec![10.0; 27], 1e6);
        }
        for layer in &critic.layers {
            for r in 0..layer.weights.rows {
                for c in 0..layer.weights.cols {
                    let w = layer.weights.get(r, c);
                    assert!(
                        w.abs() <= crate::matrix::WEIGHT_CLIP + 1e-12,
                        "Weight {w} exceeds WEIGHT_CLIP"
                    );
                }
            }
        }
    }

    // ── serde test ─────────────────────────────────────────────────

    #[test]
    fn test_serde_roundtrip_preserves_weights() {
        let mut rng = make_rng();
        let critic: MlpCritic =
            MlpCritic::new(CpuLinAlg::new(), default_config(), &mut rng).unwrap();
        let input = vec![0.3; 27];
        let original_output = critic.forward(&input);

        let weights = critic.to_weights();
        let restored: MlpCritic =
            MlpCritic::from_weights(CpuLinAlg::new(), default_config(), weights).unwrap();
        let restored_output = restored.forward(&input);

        assert!(
            (original_output - restored_output).abs() < 1e-12,
            "Serde roundtrip changed output: {original_output} vs {restored_output}"
        );
    }

    // ── Phase 6 Cycle 6.1: MlpCritic crossover same topology ───

    // ── Phase 6 Cycle 6.2: MlpCritic crossover dimension mismatch ──

    // ── Fix #5: Empty hidden_layers guard ────────────────────────

    // ── Fix #2: forward_with_hidden ────────────────────────────

    #[test]
    fn test_forward_with_hidden_returns_value_and_states() {
        let mut rng = make_rng();
        let critic: MlpCritic =
            MlpCritic::new(CpuLinAlg::new(), default_config(), &mut rng).unwrap();
        let input = vec![0.3; 27];
        let (value, hidden_states) = critic.forward_with_hidden(&input);

        assert!(value.is_finite(), "value not finite: {value}");
        // 1 hidden layer → 1 entry in hidden_states
        assert_eq!(hidden_states.len(), 1);
        // Hidden layer has 36 neurons
        assert_eq!(hidden_states[0].len(), 36);
    }

    #[test]
    fn test_forward_with_hidden_matches_forward() {
        let mut rng = make_rng();
        let critic: MlpCritic =
            MlpCritic::new(CpuLinAlg::new(), default_config(), &mut rng).unwrap();
        let input = vec![0.3; 27];
        let value_plain = critic.forward(&input);
        let (value_hidden, _) = critic.forward_with_hidden(&input);

        assert!(
            (value_plain - value_hidden).abs() < 1e-12,
            "forward and forward_with_hidden should return same value: {value_plain} vs {value_hidden}"
        );
    }

    #[test]
    fn test_forward_with_hidden_two_layers() {
        let config = MlpCriticConfig {
            input_size: 27,
            hidden_layers: vec![
                LayerDef {
                    size: 36,
                    activation: Activation::Tanh,
                },
                LayerDef {
                    size: 24,
                    activation: Activation::Tanh,
                },
            ],
            output_activation: Activation::Linear,
            lr: 0.005,
        };
        let mut rng = make_rng();
        let critic: MlpCritic = MlpCritic::new(CpuLinAlg::new(), config, &mut rng).unwrap();
        let input = vec![0.3; 27];
        let (value, hidden_states) = critic.forward_with_hidden(&input);

        assert!(value.is_finite());
        assert_eq!(hidden_states.len(), 2);
        assert_eq!(hidden_states[0].len(), 36);
        assert_eq!(hidden_states[1].len(), 24);
    }

    // ── Fix #5: Empty hidden_layers guard ────────────────────────

    // ── from_weights dimension validation tests ──────────────────────

    /// Helper: build valid MlpCriticWeights from a config.
    fn valid_weights_for(config: &MlpCriticConfig) -> MlpCriticWeights {
        let mut rng = make_rng();
        let critic =
            MlpCritic::<CpuLinAlg>::new(CpuLinAlg::new(), config.clone(), &mut rng).unwrap();
        critic.to_weights()
    }

    #[test]
    fn test_from_weights_valid_returns_ok() {
        let config = default_config();
        let weights = valid_weights_for(&config);
        let result = MlpCritic::<CpuLinAlg>::from_weights(CpuLinAlg::new(), config, weights);
        assert!(result.is_ok());
    }

    #[test]
    fn test_from_weights_wrong_weight_rows_returns_err() {
        let config = default_config(); // input=27, hidden=[36], output=1
        let mut weights = valid_weights_for(&config);
        // Layer 0 should be 36x27; corrupt rows to 20x27
        weights.layers[0].weights = crate::matrix::Matrix::zeros(20, 27);
        weights.layers[0].bias = vec![0.0; 20];
        let result = MlpCritic::<CpuLinAlg>::from_weights(CpuLinAlg::new(), config, weights);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, PcError::DimensionMismatch { .. }),
            "Expected DimensionMismatch, got: {err}"
        );
    }

    #[test]
    fn test_from_weights_wrong_weight_cols_returns_err() {
        let config = default_config(); // layer 0 should be 36x27
        let mut weights = valid_weights_for(&config);
        weights.layers[0].weights = crate::matrix::Matrix::zeros(36, 10); // wrong cols
        let result = MlpCritic::<CpuLinAlg>::from_weights(CpuLinAlg::new(), config, weights);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, PcError::DimensionMismatch { .. }),
            "Expected DimensionMismatch, got: {err}"
        );
    }

    #[test]
    fn test_from_weights_wrong_bias_length_returns_err() {
        let config = default_config(); // layer 0 bias should be len 36
        let mut weights = valid_weights_for(&config);
        weights.layers[0].bias = vec![0.0; 5];
        let result = MlpCritic::<CpuLinAlg>::from_weights(CpuLinAlg::new(), config, weights);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, PcError::DimensionMismatch { .. }),
            "Expected DimensionMismatch, got: {err}"
        );
    }

    #[test]
    fn test_from_weights_wrong_output_layer_dims_returns_err() {
        let config = default_config(); // output layer should be 1x36
        let mut weights = valid_weights_for(&config);
        let last = weights.layers.len() - 1;
        weights.layers[last].weights = crate::matrix::Matrix::zeros(1, 10); // wrong cols
        let result = MlpCritic::<CpuLinAlg>::from_weights(CpuLinAlg::new(), config, weights);
        assert!(result.is_err());
    }
}
