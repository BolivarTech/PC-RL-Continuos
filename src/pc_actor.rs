// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-03-25

//! Predictive Coding Actor Network.
//!
//! Implements an actor that uses iterative top-down/bottom-up predictive coding
//! inference loops instead of standard feedforward passes. The prediction error
//! (surprise score) drives learning rate modulation in the actor-critic agent.

use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::activation::Activation;
use crate::error::PcError;
use crate::layer::{Layer, LayerDef};
use crate::linalg::cpu::CpuLinAlg;
use crate::linalg::LinAlg;

/// Configuration for the predictive coding actor network.
///
/// # Examples
///
/// ```
/// use pc_rl_continuos::activation::Activation;
/// use pc_rl_continuos::layer::LayerDef;
/// use pc_rl_continuos::pc_actor::PcActorConfig;
///
/// let config = PcActorConfig {
///     input_size: 9,
///     hidden_layers: vec![LayerDef { size: 18, activation: Activation::Tanh }],
///     output_size: 9,
///     output_activation: Activation::Tanh,
///     alpha: 0.1,
///     tol: 0.01,
///     min_steps: 1,
///     max_steps: 20,
///     lr_weights: 0.01,
///     synchronous: true,
///     temperature: 1.0,
///     local_lambda: 1.0,
///     residual: false,
///     rezero_init: 0.001,
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PcActorConfig {
    /// Number of input features (e.g. 9 for tic-tac-toe board).
    pub input_size: usize,
    /// Hidden layer topology definitions.
    pub hidden_layers: Vec<LayerDef>,
    /// Number of output actions.
    pub output_size: usize,
    /// Activation function for the output layer.
    pub output_activation: Activation,
    /// Inference learning rate for PC loop state updates (`h += alpha * error`).
    /// Set to 0.0 to disable PC inference (network behaves as standard MLP).
    /// Active regardless of `residual` setting. Default: 0.1.
    #[serde(default = "default_alpha")]
    pub alpha: f64,
    /// Convergence threshold for RMS prediction error.
    /// PC loop exits early when surprise < tol (after at least `min_steps`).
    /// Active regardless of `residual` setting. Default: 0.01.
    #[serde(default = "default_tol")]
    pub tol: f64,
    /// Minimum PC inference steps before convergence check is allowed.
    /// Active regardless of `residual` setting. Default: 1.
    #[serde(default = "default_min_steps")]
    pub min_steps: usize,
    /// Maximum PC inference steps per action.
    /// Active regardless of `residual` setting. Default: 20.
    #[serde(default = "default_max_steps")]
    pub max_steps: usize,
    /// Base learning rate for weight updates. Default: 0.01.
    #[serde(default = "default_lr_weights")]
    pub lr_weights: f64,
    /// If true, use synchronous snapshot mode; otherwise in-place. Default: true.
    #[serde(default = "default_synchronous")]
    pub synchronous: bool,
    /// Softmax temperature for action selection. Default: 1.0.
    #[serde(default = "default_temperature")]
    pub temperature: f64,
    /// Blend factor for hidden layer weight updates, range `[0.0, 1.0]`.
    ///
    /// Controls how hidden layers combine two gradient signals:
    /// `delta = lambda * backprop_grad + (1 - lambda) * pc_prediction_error`
    ///
    /// - `1.0` — Pure backprop: reward signal propagated from output (default).
    /// - `0.0` — Pure local PC: prediction errors from inference loop
    ///   used as gradients (Millidge et al. 2022). No vanishing gradient
    ///   but no reward signal reaches hidden layers.
    /// - `0.0 < lambda < 1.0` — Hybrid: reward-aware backprop regularized
    ///   by local PC consistency errors.
    ///
    /// The output layer always uses standard backprop regardless of this value.
    #[serde(default = "default_local_lambda")]
    pub local_lambda: f64,
    /// Enable residual skip connections between same-dimension hidden layers.
    /// When false, `rezero_init` is ignored. When true, all hidden layers
    /// must have the same size, and skip connections with learnable ReZero
    /// scaling are added between consecutive hidden layers (not the first,
    /// since input_size typically differs from hidden_size).
    #[serde(default)]
    pub residual: bool,
    /// Initial value for ReZero scaling factors on residual connections.
    /// Only used when `residual = true`. Controls initial contribution of
    /// the nonlinear component: `h[i] = rezero_init * tanh(...) + h[i-1]`.
    ///
    /// - `0.001` — Near-identity start (ReZero: network learns depth gradually)
    /// - `1.0` — Standard ResNet residual (full contribution from start)
    ///
    /// Ignored when `residual = false`.
    #[serde(default = "default_rezero_init")]
    pub rezero_init: f64,
}

/// Default PC inference learning rate.
fn default_alpha() -> f64 {
    0.1
}

/// Default convergence tolerance for PC loop.
fn default_tol() -> f64 {
    0.01
}

/// Default minimum PC inference steps.
fn default_min_steps() -> usize {
    1
}

/// Default maximum PC inference steps.
fn default_max_steps() -> usize {
    20
}

/// Default base learning rate for weight updates.
fn default_lr_weights() -> f64 {
    0.01
}

/// Default synchronous mode (snapshot).
fn default_synchronous() -> bool {
    true
}

/// Default softmax temperature.
fn default_temperature() -> f64 {
    1.0
}

/// Default local_lambda: 1.0 (pure backprop).
fn default_local_lambda() -> f64 {
    1.0
}

/// Default rezero_init: 0.001 (near-identity at start).
fn default_rezero_init() -> f64 {
    0.001
}

/// Result of the predictive coding inference loop.
///
/// Contains converged output logits, hidden state representations,
/// and diagnostic information about the inference process.
///
/// Generic over a [`LinAlg`] backend `L`. Defaults to [`CpuLinAlg`].
#[derive(Debug, Clone)]
pub struct InferResult<L: LinAlg = CpuLinAlg> {
    /// Converged output logits.
    pub y_conv: L::Vector,
    /// All hidden states concatenated (fed to critic).
    pub latent_concat: L::Vector,
    /// Per-layer hidden state activations.
    pub hidden_states: Vec<L::Vector>,
    /// Per-layer prediction errors from the last PC inference step.
    /// Ordered from top hidden layer to bottom (reverse layer order).
    pub prediction_errors: Vec<L::Vector>,
    /// RMS prediction error across layers.
    pub surprise_score: f64,
    /// Number of inference steps performed.
    pub steps_used: usize,
    /// Whether the inference loop converged within tolerance.
    pub converged: bool,
    /// Per-layer tanh components for residual layers.
    /// `None` for non-skip layers, `Some(tanh_out)` for skip-eligible layers.
    /// Needed for correct backward pass (derivative on tanh_out, not full h\[i\]).
    pub tanh_components: Vec<Option<L::Vector>>,
}

/// Action selection mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectionMode {
    /// Stochastic sampling from softmax distribution.
    Training,
    /// Deterministic argmax selection.
    Play,
}

/// Predictive coding actor network.
///
/// Uses iterative top-down/bottom-up inference loops to produce
/// stable hidden representations and output logits.
///
/// Generic over a [`LinAlg`] backend `L`. Defaults to [`CpuLinAlg`].
///
/// # Examples
///
/// ```
/// use pc_rl_continuos::activation::Activation;
/// use pc_rl_continuos::layer::LayerDef;
/// use pc_rl_continuos::linalg::cpu::CpuLinAlg;
/// use pc_rl_continuos::pc_actor::{PcActor, PcActorConfig, SelectionMode};
/// use rand::SeedableRng;
/// use rand::rngs::StdRng;
///
/// let config = PcActorConfig {
///     input_size: 9,
///     hidden_layers: vec![LayerDef { size: 18, activation: Activation::Tanh }],
///     output_size: 9,
///     output_activation: Activation::Tanh,
///     alpha: 0.1, tol: 0.01, min_steps: 1, max_steps: 20,
///     lr_weights: 0.01, synchronous: true, temperature: 1.0,
///     local_lambda: 1.0,
///     residual: false,
///     rezero_init: 0.001,
/// };
/// let mut rng = StdRng::seed_from_u64(42);
/// let actor: PcActor = PcActor::new(CpuLinAlg::new(), config, &mut rng).unwrap();
/// let result = actor.infer(&[0.0; 9]);
/// assert_eq!(result.y_conv.len(), 9);
/// ```
#[derive(Debug, Clone)]
pub struct PcActor<L: LinAlg = CpuLinAlg> {
    /// Network layers: hidden_layers.len() + 1 (output layer).
    pub(crate) layers: Vec<Layer<L>>,
    /// Actor configuration.
    pub config: PcActorConfig,
    /// ReZero scaling factors for skip connections. One per skip layer (all i >= 1 when residual=true).
    pub(crate) rezero_alpha: Vec<f64>,
    /// Projection matrices for skip connections between layers of different sizes.
    /// One entry per skip layer: `None` for identity (same size), `Some(Matrix)` for projection.
    pub(crate) skip_projections: Vec<Option<L::Matrix>>,
    /// Backend used for linear algebra operations.
    pub(crate) backend: L,
}

impl<L: LinAlg> PcActor<L> {
    /// Creates a new PC actor with Xavier-initialized layers.
    ///
    /// # Arguments
    ///
    /// * `config` - Actor configuration specifying topology and hyperparameters.
    /// * `rng` - Random number generator for weight initialization.
    ///
    /// # Errors
    ///
    /// Returns `PcError::ConfigValidation` if `input_size`, `output_size`,
    /// or `temperature` are invalid.
    pub fn new(backend: L, config: PcActorConfig, rng: &mut impl Rng) -> Result<Self, PcError> {
        if config.input_size == 0 {
            return Err(PcError::ConfigValidation("input_size must be > 0".into()));
        }
        if config.output_size == 0 {
            return Err(PcError::ConfigValidation("output_size must be > 0".into()));
        }
        if config.temperature <= 0.0 {
            return Err(PcError::ConfigValidation(format!(
                "temperature must be positive, got {}",
                config.temperature
            )));
        }
        if !(0.0..=1.0).contains(&config.local_lambda) {
            return Err(PcError::ConfigValidation(format!(
                "local_lambda must be in [0.0, 1.0], got {}",
                config.local_lambda
            )));
        }
        if config.rezero_init < 0.0 {
            return Err(PcError::ConfigValidation(format!(
                "rezero_init must be >= 0, got {}",
                config.rezero_init
            )));
        }
        let mut layers: Vec<Layer<L>> = Vec::new();
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

        // Output layer
        layers.push(Layer::<L>::new(
            prev_size,
            config.output_size,
            config.output_activation,
            &backend,
            rng,
        ));

        // Compute rezero_alpha and skip_projections: one per skip layer (all i >= 1)
        let (rezero_alpha, skip_projections) = if config.residual {
            let mut alphas = Vec::new();
            let mut projs = Vec::new();
            for i in 1..config.hidden_layers.len() {
                alphas.push(config.rezero_init);
                if config.hidden_layers[i].size != config.hidden_layers[i - 1].size {
                    projs.push(Some(backend.xavier_mat(
                        config.hidden_layers[i].size,
                        config.hidden_layers[i - 1].size,
                        rng,
                    )));
                } else {
                    projs.push(None);
                }
            }
            (alphas, projs)
        } else {
            (Vec::new(), Vec::new())
        };

        Ok(Self {
            layers,
            config,
            rezero_alpha,
            skip_projections,
            backend,
        })
    }

    /// Returns the total size of the latent concatenation (sum of hidden layer sizes).
    pub fn latent_size(&self) -> usize {
        self.config.hidden_layers.iter().map(|def| def.size).sum()
    }

    /// Runs the predictive coding inference loop on the given input.
    ///
    /// This method is `&self` — it never modifies weights.
    ///
    /// # Arguments
    ///
    /// * `input` - Input vector of length `input_size`.
    ///
    /// # Panics
    ///
    /// Panics if `input.len() != config.input_size`.
    /// Returns whether hidden layer `i` has a skip connection (identity or projection).
    fn is_skip_layer(&self, i: usize) -> bool {
        self.config.residual && i >= 1
    }

    /// Returns the rezero_alpha/skip_projections index for hidden layer `i`.
    pub(crate) fn skip_alpha_index(&self, i: usize) -> Option<usize> {
        if !self.is_skip_layer(i) {
            return None;
        }
        Some(i - 1)
    }

    pub fn infer(&self, input: &[f64]) -> InferResult<L> {
        assert_eq!(
            input.len(),
            self.config.input_size,
            "input size mismatch: got {}, expected {}",
            input.len(),
            self.config.input_size
        );

        let input_vec = self.backend.vec_from_slice(input);
        let n_hidden = self.config.hidden_layers.len();

        // Forward pass to initialize hidden states and output
        let mut hidden_states: Vec<L::Vector> = Vec::with_capacity(n_hidden);
        let mut tanh_components: Vec<Option<L::Vector>> = Vec::with_capacity(n_hidden);
        let mut prev = input_vec.clone();
        for (i, layer) in self.layers[..n_hidden].iter().enumerate() {
            let tanh_out = layer.forward(&prev);
            if let Some(alpha_idx) = self.skip_alpha_index(i) {
                let alpha = self.rezero_alpha[alpha_idx];
                let scaled = self.backend.vec_scale(&tanh_out, alpha);
                let skip_path = if let Some(ref proj) = self.skip_projections[alpha_idx] {
                    self.backend.mat_vec_mul(proj, &prev)
                } else {
                    prev.clone()
                };
                prev = self.backend.vec_add(&skip_path, &scaled);
                tanh_components.push(Some(tanh_out));
            } else {
                prev = tanh_out;
                tanh_components.push(None);
            }
            hidden_states.push(prev.clone());
        }
        // Output from last hidden (or input if no hidden)
        let last_input = if n_hidden > 0 {
            &hidden_states[n_hidden - 1]
        } else {
            &input_vec
        };
        let mut y = self.layers[n_hidden].forward(last_input);

        // PC inference loop
        let mut steps_used = 0;
        let mut converged = false;
        let mut surprise_score = 0.0;
        let mut last_errors: Vec<L::Vector> = Vec::new();

        for step in 0..self.config.max_steps {
            steps_used = step + 1;

            // Synchronous mode freezes states before updating (snapshot);
            // in-place mode reads live states that include prior updates.
            // Both modes need an owned copy of target[i] since we write
            // hidden_states[i] within the loop body.
            let snap_h: Vec<L::Vector>;
            let snap_tc: Vec<Option<L::Vector>>;
            let use_snapshot = self.config.synchronous;
            if use_snapshot {
                snap_h = hidden_states.clone();
                snap_tc = tanh_components.clone();
            } else {
                snap_h = Vec::new();
                snap_tc = Vec::new();
            }

            let mut error_vecs: Vec<L::Vector> = Vec::new();

            for i in (0..n_hidden).rev() {
                // state_above: sync reads frozen snapshot, in-place reads live
                let state_above = if i == n_hidden - 1 {
                    &y
                } else if use_snapshot {
                    snap_tc[i + 1].as_ref().unwrap_or(&snap_h[i + 1])
                } else {
                    tanh_components[i + 1]
                        .as_ref()
                        .unwrap_or(&hidden_states[i + 1])
                };

                // target: always read pre-update value (clone to own it)
                let target = if use_snapshot {
                    snap_tc[i].as_ref().unwrap_or(&snap_h[i]).clone()
                } else {
                    tanh_components[i]
                        .as_ref()
                        .unwrap_or(&hidden_states[i])
                        .clone()
                };

                let prediction = self.layers[i + 1]
                    .transpose_forward(state_above, self.config.hidden_layers[i].activation);

                let error = self.backend.vec_sub(&prediction, &target);
                error_vecs.push(error.clone());

                let updated_target = self
                    .backend
                    .vec_add(&target, &self.backend.vec_scale(&error, self.config.alpha));
                if let Some(alpha_idx) = self.skip_alpha_index(i) {
                    tanh_components[i] = Some(updated_target.clone());
                    let alpha = self.rezero_alpha[alpha_idx];
                    let prev_h = if i > 0 {
                        &hidden_states[i - 1]
                    } else {
                        &input_vec
                    };
                    let skip_path = if let Some(ref proj) = self.skip_projections[alpha_idx] {
                        self.backend.mat_vec_mul(proj, prev_h)
                    } else {
                        prev_h.clone()
                    };
                    hidden_states[i] = self
                        .backend
                        .vec_add(&skip_path, &self.backend.vec_scale(&updated_target, alpha));
                } else {
                    hidden_states[i] = updated_target;
                }
            }

            let top_hidden = if n_hidden > 0 {
                &hidden_states[n_hidden - 1]
            } else {
                &input_vec
            };
            y = self.layers[n_hidden].forward(top_hidden);

            let refs: Vec<&L::Vector> = error_vecs.iter().collect();
            surprise_score = self.backend.rms_error(&refs);
            last_errors = error_vecs;

            // Convergence check (alpha must be > 0 for meaningful convergence)
            if self.config.alpha > 0.0
                && step + 1 >= self.config.min_steps
                && surprise_score < self.config.tol
            {
                converged = true;
                break;
            }
        }

        // Build latent_concat (uses vec_to_vec for GPU compatibility)
        let mut latent_raw: Vec<f64> = Vec::new();
        for h in &hidden_states {
            latent_raw.extend_from_slice(&self.backend.vec_to_vec(h));
        }
        let latent_concat = self.backend.vec_from_slice(&latent_raw);

        InferResult {
            y_conv: y,
            latent_concat,
            hidden_states,
            prediction_errors: last_errors,
            surprise_score,
            steps_used,
            converged,
            tanh_components,
        }
    }

    /// Updates network weights using a blend of backprop and local PC error.
    ///
    /// The `local_lambda` config controls the blend: 1.0 = pure backprop,
    /// 0.0 = pure local PC learning (Millidge et al. 2022), intermediate = hybrid.
    ///
    /// # Arguments
    ///
    /// * `output_delta` - Error signal at the output layer.
    /// * `infer_result` - Result from the most recent inference.
    /// * `input` - Original input that was fed to `infer`.
    /// * `surprise_scale` - Multiplier on learning rate based on surprise.
    ///
    /// # Panics
    ///
    /// Panics if `input.len() != config.input_size`.
    pub fn update_weights(
        &mut self,
        output_delta: &[f64],
        infer_result: &InferResult<L>,
        input: &[f64],
        surprise_scale: f64,
        decay_factors: &[f64],
    ) {
        assert_eq!(
            input.len(),
            self.config.input_size,
            "input size mismatch: got {}, expected {}",
            input.len(),
            self.config.input_size
        );

        self.update_weights_hybrid(
            output_delta,
            infer_result,
            input,
            surprise_scale,
            self.config.local_lambda,
            decay_factors,
        );
    }

    /// Hybrid weight update blending backprop and local PC error signals.
    ///
    /// For hidden layers, the effective delta is:
    /// `delta = lambda * backprop_delta + (1 - lambda) * pc_error`
    ///
    /// * `lambda = 1.0` → pure backprop (standard mode).
    /// * `lambda = 0.0` → pure local PC learning (Millidge et al. 2022).
    /// * `0 < lambda < 1` → hybrid blend.
    ///
    /// The output layer always uses standard backprop from `output_delta`.
    ///
    /// Per-layer consolidation decay is applied via `decay_factors[i]`:
    /// `layer_surprise = surprise_scale * decay_factors[i]`.
    /// Empty `decay_factors` means no per-layer decay (all layers get `surprise_scale`).
    fn update_weights_hybrid(
        &mut self,
        output_delta: &[f64],
        infer_result: &InferResult<L>,
        input: &[f64],
        surprise_scale: f64,
        lambda: f64,
        decay_factors: &[f64],
    ) {
        let input_vec = self.backend.vec_from_slice(input);
        let output_delta_vec = self.backend.vec_from_slice(output_delta);
        let n_hidden = self.config.hidden_layers.len();
        let n_layers = self.layers.len();

        // Output layer: always raw surprise_scale (no decay)
        let output_input = if n_hidden > 0 {
            &infer_result.hidden_states[n_hidden - 1]
        } else {
            &input_vec
        };
        let output_output = &infer_result.y_conv;
        let mut bp_delta = self.layers[n_layers - 1].backward(
            output_input,
            output_output,
            &output_delta_vec,
            self.config.lr_weights,
            surprise_scale,
        );

        // Hidden layers (from top to bottom)
        for i in (0..n_hidden).rev() {
            let layer_input = if i > 0 {
                &infer_result.hidden_states[i - 1]
            } else {
                &input_vec
            };

            // Per-layer surprise with consolidation decay
            let layer_surprise = if decay_factors.is_empty() {
                surprise_scale
            } else {
                surprise_scale * decay_factors[i]
            };

            // Blend backprop delta with local PC error
            let effective_delta = if (lambda - 1.0).abs() < f64::EPSILON {
                bp_delta.clone()
            } else if lambda.abs() < f64::EPSILON {
                let error_idx = n_hidden - 1 - i;
                infer_result.prediction_errors[error_idx].clone()
            } else {
                let error_idx = n_hidden - 1 - i;
                let pc_error = &infer_result.prediction_errors[error_idx];
                let bp_scaled = self.backend.vec_scale(&bp_delta, lambda);
                let pc_scaled = self.backend.vec_scale(pc_error, 1.0 - lambda);
                self.backend.vec_add(&bp_scaled, &pc_scaled)
            };

            if let Some(alpha_idx) = self.skip_alpha_index(i) {
                // Skip-eligible layer: use tanh_out for derivative, scale by alpha,
                // add identity path to propagated gradient, update alpha.
                let tanh_out = infer_result.tanh_components[i].as_ref().unwrap();
                let effective_lr = self.config.lr_weights * layer_surprise;

                // Scale delta by rezero_alpha for the nonlinear path
                let scaled_delta = self
                    .backend
                    .vec_scale(&effective_delta, self.rezero_alpha[alpha_idx]);

                // Backward through the layer using tanh_out (not hidden_states[i])
                let propagated = self.layers[i].backward(
                    layer_input,
                    tanh_out,
                    &scaled_delta,
                    self.config.lr_weights,
                    layer_surprise,
                );

                // Update rezero_alpha: dL/d(alpha) = delta · tanh_out
                let grad_alpha: f64 = self.backend.vec_dot(&effective_delta, tanh_out);
                self.rezero_alpha[alpha_idx] -= effective_lr * grad_alpha;

                // Propagated delta = nonlinear path + skip path (identity or projection)
                if let Some(ref mut proj) = self.skip_projections[alpha_idx] {
                    // Projection path: W_proj^T × delta
                    let proj_t = self.backend.mat_transpose(proj);
                    let skip_delta = self.backend.mat_vec_mul(&proj_t, &effective_delta);
                    // Update projection: W_proj -= lr × outer(delta, layer_input)
                    let dw_proj = self.backend.outer_product(&effective_delta, layer_input);
                    self.backend.mat_scale_add(proj, &dw_proj, -effective_lr);
                    bp_delta = self.backend.vec_add(&propagated, &skip_delta);
                } else {
                    // Identity path: + delta
                    bp_delta = self.backend.vec_add(&propagated, &effective_delta);
                }
            } else {
                // Standard layer: use hidden_states[i] as output
                let layer_output = &infer_result.hidden_states[i];
                bp_delta = self.layers[i].backward(
                    layer_input,
                    layer_output,
                    &effective_delta,
                    self.config.lr_weights,
                    layer_surprise,
                );
            }
        }
    }

    /// In-place soft (Polyak) update toward another actor.
    ///
    /// Updates each layer's weights and biases via:
    ///     `self.weights <- tau * other.weights + (1 - tau) * self.weights`
    ///     `self.bias    <- tau * other.bias    + (1 - tau) * self.bias`
    ///
    /// Includes residual ReZero scaling factors and skip projection weights
    /// when present.
    ///
    /// # Arguments
    ///
    /// * `other` - Source actor whose weights provide the target.
    /// * `tau` - Mixing rate in `[0.0, 1.0]`. Special cases: `0.0` is no-op,
    ///   `1.0` is equivalent to `copy_weights_from`.
    ///
    /// # Errors
    ///
    /// Returns `PcError::DimensionMismatch` if topologies differ.
    pub fn polyak_update_from(&mut self, other: &PcActor<L>, tau: f64) -> Result<(), PcError> {
        Self::validate_topology_match(self, other)?;

        if tau == 0.0 {
            return Ok(());
        }

        let one_minus_tau = 1.0 - tau;

        // Update layer weights and biases
        for (self_layer, other_layer) in self.layers.iter_mut().zip(other.layers.iter()) {
            let rows = self.backend.mat_rows(&self_layer.weights);
            let cols = self.backend.mat_cols(&self_layer.weights);
            for r in 0..rows {
                for c in 0..cols {
                    let s = self.backend.mat_get(&self_layer.weights, r, c);
                    let o = self.backend.mat_get(&other_layer.weights, r, c);
                    self.backend.mat_set(
                        &mut self_layer.weights,
                        r,
                        c,
                        tau * o + one_minus_tau * s,
                    );
                }
            }
            let len = self.backend.vec_len(&self_layer.bias);
            for i in 0..len {
                let s = self.backend.vec_get(&self_layer.bias, i);
                let o = self.backend.vec_get(&other_layer.bias, i);
                self.backend
                    .vec_set(&mut self_layer.bias, i, tau * o + one_minus_tau * s);
            }
        }

        // Update ReZero scaling factors
        for (sa, oa) in self.rezero_alpha.iter_mut().zip(other.rezero_alpha.iter()) {
            *sa = tau * oa + one_minus_tau * (*sa);
        }

        // Update skip projection matrices
        for (sp, op) in self
            .skip_projections
            .iter_mut()
            .zip(other.skip_projections.iter())
        {
            if let (Some(sm), Some(om)) = (sp, op) {
                let rows = self.backend.mat_rows(sm);
                let cols = self.backend.mat_cols(sm);
                for r in 0..rows {
                    for c in 0..cols {
                        let s = self.backend.mat_get(sm, r, c);
                        let o = self.backend.mat_get(om, r, c);
                        self.backend.mat_set(sm, r, c, tau * o + one_minus_tau * s);
                    }
                }
            }
        }

        Ok(())
    }

    /// In-place hard copy of weights and biases from another actor.
    ///
    /// Copies all weights, biases, ReZero scaling factors, and skip
    /// projection matrices directly without arithmetic interpolation.
    ///
    /// # Arguments
    ///
    /// * `other` - Source actor whose weights are copied.
    ///
    /// # Errors
    ///
    /// Returns `PcError::DimensionMismatch` if topologies differ.
    pub fn copy_weights_from(&mut self, other: &PcActor<L>) -> Result<(), PcError> {
        Self::validate_topology_match(self, other)?;

        // Copy layer weights and biases
        for (self_layer, other_layer) in self.layers.iter_mut().zip(other.layers.iter()) {
            let rows = self.backend.mat_rows(&self_layer.weights);
            let cols = self.backend.mat_cols(&self_layer.weights);
            for r in 0..rows {
                for c in 0..cols {
                    let val = self.backend.mat_get(&other_layer.weights, r, c);
                    self.backend.mat_set(&mut self_layer.weights, r, c, val);
                }
            }
            let len = self.backend.vec_len(&self_layer.bias);
            for i in 0..len {
                let val = self.backend.vec_get(&other_layer.bias, i);
                self.backend.vec_set(&mut self_layer.bias, i, val);
            }
        }

        // Copy ReZero scaling factors
        for (sa, oa) in self.rezero_alpha.iter_mut().zip(other.rezero_alpha.iter()) {
            *sa = *oa;
        }

        // Copy skip projection matrices
        for (sp, op) in self
            .skip_projections
            .iter_mut()
            .zip(other.skip_projections.iter())
        {
            if let (Some(sm), Some(om)) = (sp, op) {
                let rows = self.backend.mat_rows(sm);
                let cols = self.backend.mat_cols(sm);
                for r in 0..rows {
                    for c in 0..cols {
                        let val = self.backend.mat_get(om, r, c);
                        self.backend.mat_set(sm, r, c, val);
                    }
                }
            }
        }

        Ok(())
    }

    /// Validates that two actors have matching topologies.
    ///
    /// Checks layer count, per-layer weight/bias dimensions, residual
    /// component counts, per-slot skip projection presence agreement,
    /// and skip projection matrix dimensions.
    ///
    /// # Errors
    ///
    /// Returns `PcError::DimensionMismatch` if any topology element differs.
    fn validate_topology_match(a: &PcActor<L>, b: &PcActor<L>) -> Result<(), PcError> {
        if a.layers.len() != b.layers.len() {
            return Err(PcError::DimensionMismatch {
                expected: a.layers.len(),
                got: b.layers.len(),
                context: "polyak actor layer count",
            });
        }
        for (i, (la, lb)) in a.layers.iter().zip(b.layers.iter()).enumerate() {
            let ra = a.backend.mat_rows(&la.weights);
            let rb = b.backend.mat_rows(&lb.weights);
            if ra != rb {
                return Err(PcError::DimensionMismatch {
                    expected: ra,
                    got: rb,
                    context: if i < a.layers.len() - 1 {
                        "polyak actor hidden layer rows"
                    } else {
                        "polyak actor output layer rows"
                    },
                });
            }
            let ca = a.backend.mat_cols(&la.weights);
            let cb = b.backend.mat_cols(&lb.weights);
            if ca != cb {
                return Err(PcError::DimensionMismatch {
                    expected: ca,
                    got: cb,
                    context: if i < a.layers.len() - 1 {
                        "polyak actor hidden layer cols"
                    } else {
                        "polyak actor output layer cols"
                    },
                });
            }
        }
        if a.rezero_alpha.len() != b.rezero_alpha.len() {
            return Err(PcError::DimensionMismatch {
                expected: a.rezero_alpha.len(),
                got: b.rezero_alpha.len(),
                context: "polyak actor rezero_alpha count",
            });
        }
        if a.skip_projections.len() != b.skip_projections.len() {
            return Err(PcError::DimensionMismatch {
                expected: a.skip_projections.len(),
                got: b.skip_projections.len(),
                context: "polyak actor skip_projections count",
            });
        }
        // Per-slot skip projection presence agreement and dimension check
        for (sa, sb) in a.skip_projections.iter().zip(b.skip_projections.iter()) {
            match (sa, sb) {
                (Some(ma), Some(mb)) => {
                    let ra = a.backend.mat_rows(ma);
                    let rb = b.backend.mat_rows(mb);
                    if ra != rb {
                        return Err(PcError::DimensionMismatch {
                            expected: ra,
                            got: rb,
                            context: "polyak actor skip_projection rows",
                        });
                    }
                    let ca = a.backend.mat_cols(ma);
                    let cb = b.backend.mat_cols(mb);
                    if ca != cb {
                        return Err(PcError::DimensionMismatch {
                            expected: ca,
                            got: cb,
                            context: "polyak actor skip_projection cols",
                        });
                    }
                }
                (None, None) => {}
                (Some(_), None) => {
                    return Err(PcError::DimensionMismatch {
                        expected: 1,
                        got: 0,
                        context: "polyak actor skip_projection presence at slot",
                    });
                }
                (None, Some(_)) => {
                    return Err(PcError::DimensionMismatch {
                        expected: 0,
                        got: 1,
                        context: "polyak actor skip_projection presence at slot",
                    });
                }
            }
        }
        Ok(())
    }

    /// Extracts a serializable snapshot of current weights.
    ///
    /// Converts generic layers and skip projections to CPU-backed types.
    pub fn to_weights(&self) -> crate::serializer::PcActorWeights {
        let cpu_layers: Vec<Layer<CpuLinAlg>> = self
            .layers
            .iter()
            .map(|layer| {
                let rows = self.backend.mat_rows(&layer.weights);
                let cols = self.backend.mat_cols(&layer.weights);
                let mut cpu_weights = crate::matrix::Matrix::zeros(rows, cols);
                for r in 0..rows {
                    for c in 0..cols {
                        cpu_weights.set(r, c, self.backend.mat_get(&layer.weights, r, c));
                    }
                }
                let bias_data = self.backend.vec_to_vec(&layer.bias);
                Layer {
                    weights: cpu_weights,
                    bias: bias_data,
                    activation: layer.activation,
                    backend: CpuLinAlg::new(),
                }
            })
            .collect();
        let cpu_projs: Vec<Option<crate::matrix::Matrix>> = self
            .skip_projections
            .iter()
            .map(|opt| {
                opt.as_ref().map(|m| {
                    let rows = self.backend.mat_rows(m);
                    let cols = self.backend.mat_cols(m);
                    let mut cpu_m = crate::matrix::Matrix::zeros(rows, cols);
                    for r in 0..rows {
                        for c in 0..cols {
                            cpu_m.set(r, c, self.backend.mat_get(m, r, c));
                        }
                    }
                    cpu_m
                })
            })
            .collect();
        crate::serializer::PcActorWeights {
            layers: cpu_layers,
            rezero_alpha: self.rezero_alpha.clone(),
            skip_projections: cpu_projs,
        }
    }

    /// Restores an actor from saved weights without requiring an RNG.
    ///
    /// Converts CPU-backed weight snapshots to the target backend `L`.
    /// Validates that all weight matrix dimensions and bias lengths match
    /// the expected topology from `config`.
    ///
    /// # Errors
    ///
    /// Returns `PcError::DimensionMismatch` if any weight matrix or bias
    /// vector has dimensions inconsistent with the config topology.
    pub fn from_weights(
        backend: L,
        config: PcActorConfig,
        weights: crate::serializer::PcActorWeights,
    ) -> Result<Self, PcError> {
        let n_hidden = config.hidden_layers.len();
        let expected_layers = n_hidden + 1;

        if weights.layers.len() != expected_layers {
            return Err(PcError::DimensionMismatch {
                expected: expected_layers,
                got: weights.layers.len(),
                context: "actor layer count",
            });
        }

        // Validate each layer's dimensions
        let mut prev_size = config.input_size;
        for (i, cpu_layer) in weights.layers.iter().enumerate() {
            let (expected_rows, expected_cols) = if i < n_hidden {
                (config.hidden_layers[i].size, prev_size)
            } else {
                (config.output_size, prev_size)
            };

            if cpu_layer.weights.rows != expected_rows {
                return Err(PcError::DimensionMismatch {
                    expected: expected_rows,
                    got: cpu_layer.weights.rows,
                    context: "actor layer weight rows",
                });
            }
            if cpu_layer.weights.cols != expected_cols {
                return Err(PcError::DimensionMismatch {
                    expected: expected_cols,
                    got: cpu_layer.weights.cols,
                    context: "actor layer weight cols",
                });
            }
            if cpu_layer.bias.len() != expected_rows {
                return Err(PcError::DimensionMismatch {
                    expected: expected_rows,
                    got: cpu_layer.bias.len(),
                    context: "actor layer bias length",
                });
            }

            if i < n_hidden {
                prev_size = config.hidden_layers[i].size;
            }
        }

        // Validate residual components
        if config.residual {
            let expected_residual = n_hidden.saturating_sub(1);
            if weights.rezero_alpha.len() != expected_residual {
                return Err(PcError::DimensionMismatch {
                    expected: expected_residual,
                    got: weights.rezero_alpha.len(),
                    context: "actor rezero_alpha count",
                });
            }
            if weights.skip_projections.len() != expected_residual {
                return Err(PcError::DimensionMismatch {
                    expected: expected_residual,
                    got: weights.skip_projections.len(),
                    context: "actor skip_projections count",
                });
            }
            // Validate skip projection dimensions (rows/cols)
            for (i, proj_opt) in weights.skip_projections.iter().enumerate() {
                if let Some(ref proj) = proj_opt {
                    let expected_rows = config.hidden_layers[i + 1].size;
                    let expected_cols = config.hidden_layers[i].size;
                    if proj.rows != expected_rows || proj.cols != expected_cols {
                        return Err(PcError::DimensionMismatch {
                            expected: expected_rows * expected_cols,
                            got: proj.rows * proj.cols,
                            context: "actor skip_projection dimensions",
                        });
                    }
                }
            }
        }

        // Convert layers
        let layers: Vec<Layer<L>> = weights
            .layers
            .into_iter()
            .map(|cpu_layer| {
                let rows = cpu_layer.weights.rows;
                let cols = cpu_layer.weights.cols;
                let mut mat = backend.zeros_mat(rows, cols);
                for r in 0..rows {
                    for c in 0..cols {
                        backend.mat_set(&mut mat, r, c, cpu_layer.weights.get(r, c));
                    }
                }
                let bias = backend.vec_from_slice(&cpu_layer.bias);
                Layer {
                    weights: mat,
                    bias,
                    activation: cpu_layer.activation,
                    backend: backend.clone(),
                }
            })
            .collect();
        let skip_projections: Vec<Option<L::Matrix>> = weights
            .skip_projections
            .into_iter()
            .map(|opt| {
                opt.map(|cpu_m| {
                    let rows = cpu_m.rows;
                    let cols = cpu_m.cols;
                    let mut mat = backend.zeros_mat(rows, cols);
                    for r in 0..rows {
                        for c in 0..cols {
                            backend.mat_set(&mut mat, r, c, cpu_m.get(r, c));
                        }
                    }
                    mat
                })
            })
            .collect();
        Ok(Self {
            layers,
            config,
            rezero_alpha: weights.rezero_alpha,
            skip_projections,
            backend,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activation::Activation;
    use crate::layer::LayerDef;
    use crate::matrix::WEIGHT_CLIP;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn make_rng() -> StdRng {
        StdRng::seed_from_u64(42)
    }

    fn default_config() -> PcActorConfig {
        PcActorConfig {
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
        }
    }

    fn two_hidden_config() -> PcActorConfig {
        PcActorConfig {
            hidden_layers: vec![
                LayerDef {
                    size: 18,
                    activation: Activation::Tanh,
                },
                LayerDef {
                    size: 12,
                    activation: Activation::Tanh,
                },
            ],
            ..default_config()
        }
    }

    // ── Inference Tests ──────────────────────────────────────────────

    #[test]
    fn test_infer_converges_on_zero_board() {
        let mut rng = make_rng();
        let actor: PcActor = PcActor::new(CpuLinAlg::new(), default_config(), &mut rng).unwrap();
        let result = actor.infer(&[0.0; 9]);
        // Should complete without panic; all finite
        for &v in &result.y_conv {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_infer_steps_used_at_least_min_steps() {
        let mut rng = make_rng();
        let config = PcActorConfig {
            min_steps: 3,
            ..default_config()
        };
        let actor: PcActor = PcActor::new(CpuLinAlg::new(), config, &mut rng).unwrap();
        let result = actor.infer(&[0.0; 9]);
        assert!(result.steps_used >= 3);
    }

    #[test]
    fn test_infer_alpha_zero_does_not_converge() {
        let mut rng = make_rng();
        let config = PcActorConfig {
            alpha: 0.0,
            ..default_config()
        };
        let actor: PcActor = PcActor::new(CpuLinAlg::new(), config, &mut rng).unwrap();
        let result = actor.infer(&[0.0; 9]);
        assert!(!result.converged);
        assert_eq!(result.steps_used, 20);
    }

    #[test]
    fn test_infer_does_not_modify_weights() {
        let mut rng = make_rng();
        let actor: PcActor = PcActor::new(CpuLinAlg::new(), default_config(), &mut rng).unwrap();
        let weights_before: Vec<Vec<f64>> = actor
            .layers
            .iter()
            .map(|l| l.weights.data.clone())
            .collect();
        let _ = actor.infer(&[0.0; 9]);
        for (i, layer) in actor.layers.iter().enumerate() {
            assert_eq!(layer.weights.data, weights_before[i]);
        }
    }

    #[test]
    fn test_infer_latent_size_single_hidden() {
        let mut rng = make_rng();
        let actor: PcActor = PcActor::new(CpuLinAlg::new(), default_config(), &mut rng).unwrap();
        let result = actor.infer(&[0.0; 9]);
        assert_eq!(result.latent_concat.len(), 18);
    }

    #[test]
    fn test_infer_latent_size_two_hidden() {
        let mut rng = make_rng();
        let actor: PcActor = PcActor::new(CpuLinAlg::new(), two_hidden_config(), &mut rng).unwrap();
        let result = actor.infer(&[0.0; 9]);
        assert_eq!(result.latent_concat.len(), 30);
    }

    #[test]
    fn test_infer_latent_size_matches_latent_size_method() {
        let mut rng = make_rng();
        let actor: PcActor = PcActor::new(CpuLinAlg::new(), two_hidden_config(), &mut rng).unwrap();
        let result = actor.infer(&[0.0; 9]);
        assert_eq!(result.latent_concat.len(), actor.latent_size());
    }

    #[test]
    fn test_infer_y_conv_length_equals_output_size() {
        let mut rng = make_rng();
        let actor: PcActor = PcActor::new(CpuLinAlg::new(), default_config(), &mut rng).unwrap();
        let result = actor.infer(&[0.0; 9]);
        assert_eq!(result.y_conv.len(), 9);
    }

    #[test]
    fn test_infer_hidden_states_count_matches_hidden_layers() {
        let mut rng = make_rng();
        let actor: PcActor = PcActor::new(CpuLinAlg::new(), two_hidden_config(), &mut rng).unwrap();
        let result = actor.infer(&[0.0; 9]);
        assert_eq!(result.hidden_states.len(), 2);
    }

    #[test]
    fn test_infer_all_outputs_finite() {
        let mut rng = make_rng();
        let actor: PcActor = PcActor::new(CpuLinAlg::new(), default_config(), &mut rng).unwrap();
        let result = actor.infer(&[1.0, -1.0, 0.5, -0.5, 0.0, 1.0, -1.0, 0.5, -0.5]);
        for &v in &result.y_conv {
            assert!(v.is_finite());
        }
        for &v in &result.latent_concat {
            assert!(v.is_finite());
        }
        assert!(result.surprise_score.is_finite());
    }

    #[test]
    fn test_infer_surprise_score_nonnegative() {
        let mut rng = make_rng();
        let actor: PcActor = PcActor::new(CpuLinAlg::new(), default_config(), &mut rng).unwrap();
        let result = actor.infer(&[0.0; 9]);
        assert!(result.surprise_score >= 0.0);
    }

    #[test]
    fn test_infer_synchronous_and_inplace_both_converge() {
        let mut rng = make_rng();
        let sync_actor: PcActor =
            PcActor::new(CpuLinAlg::new(), default_config(), &mut rng).unwrap();
        let mut rng2 = make_rng();
        let inplace_config = PcActorConfig {
            synchronous: false,
            ..default_config()
        };
        let inplace_actor: PcActor =
            PcActor::new(CpuLinAlg::new(), inplace_config, &mut rng2).unwrap();
        let sync_result = sync_actor.infer(&[0.0; 9]);
        let inplace_result = inplace_actor.infer(&[0.0; 9]);
        // Both should complete without panic; at least one should converge or use all steps
        assert!(sync_result.steps_used > 0);
        assert!(inplace_result.steps_used > 0);
    }

    #[test]
    fn test_infer_synchronous_produces_different_result_than_inplace() {
        let mut rng = make_rng();
        let config = PcActorConfig {
            hidden_layers: vec![
                LayerDef {
                    size: 18,
                    activation: Activation::Tanh,
                },
                LayerDef {
                    size: 12,
                    activation: Activation::Tanh,
                },
            ],
            alpha: 0.3,
            tol: 1e-15,
            min_steps: 1,
            max_steps: 3,
            ..default_config()
        };
        let sync_actor: PcActor = PcActor::new(CpuLinAlg::new(), config.clone(), &mut rng).unwrap();
        let mut rng2 = make_rng();
        let inplace_config = PcActorConfig {
            synchronous: false,
            ..config
        };
        let inplace_actor: PcActor =
            PcActor::new(CpuLinAlg::new(), inplace_config, &mut rng2).unwrap();
        let input = [1.0, -1.0, 0.5, -0.5, 0.0, 1.0, -1.0, 0.5, -0.5];
        let sync_result = sync_actor.infer(&input);
        let inplace_result = inplace_actor.infer(&input);
        // Different update orders should produce different hidden representations
        let differs = sync_result
            .latent_concat
            .iter()
            .zip(inplace_result.latent_concat.iter())
            .any(|(a, b)| (a - b).abs() > 1e-12);
        assert!(
            differs,
            "Synchronous and in-place should produce different results"
        );
    }

    #[test]
    #[should_panic(expected = "input size")]
    fn test_infer_panics_wrong_input_length() {
        let mut rng = make_rng();
        let actor: PcActor = PcActor::new(CpuLinAlg::new(), default_config(), &mut rng).unwrap();
        let _ = actor.infer(&[0.0; 5]);
    }

    // ── Action Selection Tests ───────────────────────────────────────

    // ── Weight Update Tests ──────────────────────────────────────────

    #[test]
    fn test_update_weights_changes_first_layer() {
        let mut rng = make_rng();
        let mut actor: PcActor =
            PcActor::new(CpuLinAlg::new(), default_config(), &mut rng).unwrap();
        let input = vec![1.0, -1.0, 0.5, -0.5, 0.0, 1.0, -1.0, 0.5, -0.5];
        let infer_result = actor.infer(&input);
        let weights_before = actor.layers[0].weights.data.clone();
        let delta = vec![0.1; 9];
        actor.update_weights(&delta, &infer_result, &input, 1.0, &[]);
        assert_ne!(actor.layers[0].weights.data, weights_before);
    }

    #[test]
    fn test_update_weights_clips_all_layers() {
        let mut rng = make_rng();
        let mut actor: PcActor =
            PcActor::new(CpuLinAlg::new(), default_config(), &mut rng).unwrap();
        let input = vec![1.0; 9];
        let infer_result = actor.infer(&input);
        let delta = vec![1e6; 9];
        actor.update_weights(&delta, &infer_result, &input, 1.0, &[]);
        for layer in &actor.layers {
            for &w in &layer.weights.data {
                assert!(
                    w.abs() <= WEIGHT_CLIP + 1e-12,
                    "Weight {w} exceeds WEIGHT_CLIP"
                );
            }
        }
    }

    #[test]
    fn test_update_weights_two_hidden_changes_both_layers() {
        let mut rng = make_rng();
        let mut actor: PcActor =
            PcActor::new(CpuLinAlg::new(), two_hidden_config(), &mut rng).unwrap();
        let input = vec![0.5; 9];
        let infer_result = actor.infer(&input);
        let w0_before = actor.layers[0].weights.data.clone();
        let w1_before = actor.layers[1].weights.data.clone();
        let delta = vec![0.1; 9];
        actor.update_weights(&delta, &infer_result, &input, 1.0, &[]);
        assert_ne!(
            actor.layers[0].weights.data, w0_before,
            "Layer 0 should change"
        );
        assert_ne!(
            actor.layers[1].weights.data, w1_before,
            "Layer 1 should change"
        );
    }

    #[test]
    #[should_panic(expected = "input size")]
    fn test_update_weights_panics_wrong_x_size() {
        let mut rng = make_rng();
        let mut actor: PcActor =
            PcActor::new(CpuLinAlg::new(), default_config(), &mut rng).unwrap();
        let input = vec![0.0; 9];
        let infer_result = actor.infer(&input);
        let delta = vec![0.1; 9];
        actor.update_weights(&delta, &infer_result, &[0.0; 5], 1.0, &[]);
    }

    // ── Zero Hidden Layers Test ─────────────────────────────────

    #[test]
    fn test_infer_zero_hidden_layers_produces_finite_output() {
        let mut rng = make_rng();
        let config = PcActorConfig {
            hidden_layers: vec![],
            ..default_config()
        };
        let actor: PcActor = PcActor::new(CpuLinAlg::new(), config, &mut rng).unwrap();
        let result = actor.infer(&[0.5; 9]);
        assert_eq!(result.y_conv.len(), 9);
        assert!(result.y_conv.iter().all(|v| v.is_finite()));
        assert!(result.latent_concat.is_empty());
        assert!(result.hidden_states.is_empty());
    }

    // ── Config Validation Tests ─────────────────────────────────

    #[test]
    fn test_new_zero_input_size_returns_error() {
        let mut rng = make_rng();
        let config = PcActorConfig {
            input_size: 0,
            ..default_config()
        };
        let result: Result<PcActor, _> = PcActor::new(CpuLinAlg::new(), config, &mut rng);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, crate::error::PcError::ConfigValidation(_)));
    }

    #[test]
    fn test_new_zero_output_size_returns_error() {
        let mut rng = make_rng();
        let config = PcActorConfig {
            output_size: 0,
            ..default_config()
        };
        let result: Result<PcActor, _> = PcActor::new(CpuLinAlg::new(), config, &mut rng);
        assert!(result.is_err());
    }

    #[test]
    fn test_new_zero_temperature_returns_error() {
        let mut rng = make_rng();
        let config = PcActorConfig {
            temperature: 0.0,
            ..default_config()
        };
        let result: Result<PcActor, _> = PcActor::new(CpuLinAlg::new(), config, &mut rng);
        assert!(result.is_err());
    }

    #[test]
    fn test_new_negative_temperature_returns_error() {
        let mut rng = make_rng();
        let config = PcActorConfig {
            temperature: -1.0,
            ..default_config()
        };
        let result: Result<PcActor, _> = PcActor::new(CpuLinAlg::new(), config, &mut rng);
        assert!(result.is_err());
    }

    // ── Residual / ReZero Config Tests ────────────────────────

    #[test]
    fn test_default_config_residual_false() {
        let config = default_config();
        assert!(!config.residual);
    }

    #[test]
    fn test_default_config_rezero_init() {
        let config = default_config();
        assert!((config.rezero_init - 0.001).abs() < 1e-12);
    }

    #[test]
    fn test_new_negative_rezero_init_returns_error() {
        let mut rng = make_rng();
        let config = PcActorConfig {
            residual: true,
            rezero_init: -0.1,
            ..default_config()
        };
        let result: Result<PcActor, _> = PcActor::new(CpuLinAlg::new(), config, &mut rng);
        assert!(result.is_err());
    }

    #[test]
    fn test_residual_mixed_sizes_accepted() {
        let mut rng = make_rng();
        let config = PcActorConfig {
            residual: true,
            hidden_layers: vec![
                LayerDef {
                    size: 27,
                    activation: Activation::Tanh,
                },
                LayerDef {
                    size: 18,
                    activation: Activation::Tanh,
                },
            ],
            ..default_config()
        };
        let result: Result<PcActor, _> = PcActor::new(CpuLinAlg::new(), config, &mut rng);
        assert!(result.is_ok());
    }

    #[test]
    fn test_residual_mixed_sizes_all_skip() {
        // [27, 27, 18]: ALL layers i>=1 get skip — identity for 27→27, projection for 27→18
        let mut rng = make_rng();
        let config = PcActorConfig {
            residual: true,
            hidden_layers: vec![
                LayerDef {
                    size: 27,
                    activation: Activation::Tanh,
                },
                LayerDef {
                    size: 27,
                    activation: Activation::Tanh,
                },
                LayerDef {
                    size: 18,
                    activation: Activation::Tanh,
                },
            ],
            ..default_config()
        };
        let actor: PcActor = PcActor::new(CpuLinAlg::new(), config, &mut rng).unwrap();
        // 2 skips: layer 1 (identity) + layer 2 (projection)
        assert_eq!(actor.rezero_alpha.len(), 2);
    }

    #[test]
    fn test_residual_heterogeneous_has_projection() {
        // [27, 18]: different sizes → projection matrix created
        let mut rng = make_rng();
        let config = PcActorConfig {
            residual: true,
            hidden_layers: vec![
                LayerDef {
                    size: 27,
                    activation: Activation::Tanh,
                },
                LayerDef {
                    size: 18,
                    activation: Activation::Tanh,
                },
            ],
            ..default_config()
        };
        let actor: PcActor = PcActor::new(CpuLinAlg::new(), config, &mut rng).unwrap();
        assert_eq!(actor.rezero_alpha.len(), 1);
        assert_eq!(actor.skip_projections.len(), 1);
        assert!(actor.skip_projections[0].is_some());
        let proj = actor.skip_projections[0].as_ref().unwrap();
        assert_eq!(proj.rows, 18); // output dim
        assert_eq!(proj.cols, 27); // input dim
    }

    #[test]
    fn test_residual_homogeneous_no_projection() {
        // [27, 27]: same sizes → no projection needed
        let mut rng = make_rng();
        let actor: PcActor =
            PcActor::new(CpuLinAlg::new(), residual_two_hidden_config(), &mut rng).unwrap();
        assert_eq!(actor.skip_projections.len(), 1);
        assert!(actor.skip_projections[0].is_none());
    }

    #[test]
    fn test_residual_mixed_sizes_infer_finite() {
        let mut rng = make_rng();
        let config = PcActorConfig {
            residual: true,
            hidden_layers: vec![
                LayerDef {
                    size: 27,
                    activation: Activation::Tanh,
                },
                LayerDef {
                    size: 27,
                    activation: Activation::Tanh,
                },
                LayerDef {
                    size: 18,
                    activation: Activation::Tanh,
                },
            ],
            ..default_config()
        };
        let actor: PcActor = PcActor::new(CpuLinAlg::new(), config, &mut rng).unwrap();
        let result = actor.infer(&[0.5; 9]);
        for &v in &result.y_conv {
            assert!(v.is_finite());
        }
        assert_eq!(result.hidden_states.len(), 3);
        assert_eq!(result.latent_concat.len(), 27 + 27 + 18);
    }

    #[test]
    fn test_residual_same_size_hidden_layers_accepted() {
        let mut rng = make_rng();
        let config = PcActorConfig {
            residual: true,
            hidden_layers: vec![
                LayerDef {
                    size: 27,
                    activation: Activation::Tanh,
                },
                LayerDef {
                    size: 27,
                    activation: Activation::Tanh,
                },
            ],
            ..default_config()
        };
        let result: Result<PcActor, _> = PcActor::new(CpuLinAlg::new(), config, &mut rng);
        assert!(result.is_ok());
    }

    fn residual_two_hidden_config() -> PcActorConfig {
        PcActorConfig {
            residual: true,
            hidden_layers: vec![
                LayerDef {
                    size: 27,
                    activation: Activation::Tanh,
                },
                LayerDef {
                    size: 27,
                    activation: Activation::Tanh,
                },
            ],
            ..default_config()
        }
    }

    #[test]
    fn test_non_residual_actor_empty_rezero_alpha() {
        let mut rng = make_rng();
        let actor: PcActor = PcActor::new(CpuLinAlg::new(), default_config(), &mut rng).unwrap();
        assert!(actor.rezero_alpha.is_empty());
    }

    #[test]
    fn test_residual_two_hidden_one_rezero_alpha() {
        let mut rng = make_rng();
        let actor: PcActor =
            PcActor::new(CpuLinAlg::new(), residual_two_hidden_config(), &mut rng).unwrap();
        assert_eq!(actor.rezero_alpha.len(), 1);
    }

    #[test]
    fn test_residual_three_hidden_two_rezero_alpha() {
        let mut rng = make_rng();
        let config = PcActorConfig {
            residual: true,
            hidden_layers: vec![
                LayerDef {
                    size: 27,
                    activation: Activation::Tanh,
                },
                LayerDef {
                    size: 27,
                    activation: Activation::Tanh,
                },
                LayerDef {
                    size: 27,
                    activation: Activation::Tanh,
                },
            ],
            ..default_config()
        };
        let actor: PcActor = PcActor::new(CpuLinAlg::new(), config, &mut rng).unwrap();
        assert_eq!(actor.rezero_alpha.len(), 2);
    }

    #[test]
    fn test_rezero_alpha_initialized_to_rezero_init() {
        let mut rng = make_rng();
        let config = PcActorConfig {
            rezero_init: 0.005,
            ..residual_two_hidden_config()
        };
        let actor: PcActor = PcActor::new(CpuLinAlg::new(), config, &mut rng).unwrap();
        assert!((actor.rezero_alpha[0] - 0.005).abs() < 1e-12);
    }

    #[test]
    fn test_residual_single_hidden_zero_rezero_alpha() {
        let mut rng = make_rng();
        let config = PcActorConfig {
            residual: true,
            ..default_config()
        };
        let actor: PcActor = PcActor::new(CpuLinAlg::new(), config, &mut rng).unwrap();
        assert!(actor.rezero_alpha.is_empty());
    }

    #[test]
    fn test_residual_single_hidden_accepted() {
        let mut rng = make_rng();
        let config = PcActorConfig {
            residual: true,
            ..default_config()
        };
        let result: Result<PcActor, _> = PcActor::new(CpuLinAlg::new(), config, &mut rng);
        assert!(result.is_ok());
    }

    // ── Local Learning (PC-based weight updates) Tests ──────────

    // ── Residual Inference Tests ──────────────────────────────

    #[test]
    fn test_residual_false_identical_to_non_residual() {
        let input = vec![1.0, -1.0, 0.5, -0.5, 0.0, 1.0, -1.0, 0.5, -0.5];
        let mut rng1 = make_rng();
        let actor1: PcActor =
            PcActor::new(CpuLinAlg::new(), two_hidden_config(), &mut rng1).unwrap();
        let result1 = actor1.infer(&input);

        let mut rng2 = make_rng();
        let config2 = PcActorConfig {
            residual: false,
            ..two_hidden_config()
        };
        let actor2: PcActor = PcActor::new(CpuLinAlg::new(), config2, &mut rng2).unwrap();
        let result2 = actor2.infer(&input);

        for (a, b) in result1.y_conv.iter().zip(result2.y_conv.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn test_residual_rezero_zero_second_hidden_near_identity() {
        let mut rng = make_rng();
        let config = PcActorConfig {
            rezero_init: 0.0,
            alpha: 0.0,
            ..residual_two_hidden_config()
        };
        let actor: PcActor = PcActor::new(CpuLinAlg::new(), config, &mut rng).unwrap();
        let result = actor.infer(&[0.5; 9]);
        let h0 = &result.hidden_states[0];
        let h1 = &result.hidden_states[1];
        for (a, b) in h0.iter().zip(h1.iter()) {
            assert!(
                (a - b).abs() < 1e-12,
                "With rezero_init=0, h[1] should equal h[0]"
            );
        }
    }

    #[test]
    fn test_residual_infer_all_outputs_finite() {
        let mut rng = make_rng();
        let actor: PcActor =
            PcActor::new(CpuLinAlg::new(), residual_two_hidden_config(), &mut rng).unwrap();
        let result = actor.infer(&[0.5; 9]);
        for &v in &result.y_conv {
            assert!(v.is_finite());
        }
        for &v in &result.latent_concat {
            assert!(v.is_finite());
        }
        assert!(result.surprise_score.is_finite());
    }

    #[test]
    fn test_residual_latent_concat_size() {
        let mut rng = make_rng();
        let actor: PcActor =
            PcActor::new(CpuLinAlg::new(), residual_two_hidden_config(), &mut rng).unwrap();
        let result = actor.infer(&[0.5; 9]);
        assert_eq!(result.latent_concat.len(), 54); // 27 + 27
    }

    #[test]
    fn test_residual_pc_loop_completes() {
        let mut rng = make_rng();
        let config = PcActorConfig {
            alpha: 0.03,
            max_steps: 5,
            ..residual_two_hidden_config()
        };
        let actor: PcActor = PcActor::new(CpuLinAlg::new(), config, &mut rng).unwrap();
        let result = actor.infer(&[0.5; 9]);
        assert!(result.steps_used > 0);
        assert!(result.steps_used <= 5);
    }

    #[test]
    fn test_residual_hidden_states_count() {
        let mut rng = make_rng();
        let actor: PcActor =
            PcActor::new(CpuLinAlg::new(), residual_two_hidden_config(), &mut rng).unwrap();
        let result = actor.infer(&[0.5; 9]);
        assert_eq!(result.hidden_states.len(), 2);
    }

    #[test]
    fn test_residual_infer_does_not_modify_weights() {
        let mut rng = make_rng();
        let actor: PcActor =
            PcActor::new(CpuLinAlg::new(), residual_two_hidden_config(), &mut rng).unwrap();
        let weights_before: Vec<Vec<f64>> = actor
            .layers
            .iter()
            .map(|l| l.weights.data.clone())
            .collect();
        let alpha_before = actor.rezero_alpha.clone();
        let _ = actor.infer(&[0.5; 9]);
        for (i, layer) in actor.layers.iter().enumerate() {
            assert_eq!(layer.weights.data, weights_before[i]);
        }
        assert_eq!(actor.rezero_alpha, alpha_before);
    }

    #[test]
    fn test_residual_three_hidden_infer_finite() {
        let mut rng = make_rng();
        let config = PcActorConfig {
            residual: true,
            hidden_layers: vec![
                LayerDef {
                    size: 27,
                    activation: Activation::Tanh,
                },
                LayerDef {
                    size: 27,
                    activation: Activation::Tanh,
                },
                LayerDef {
                    size: 27,
                    activation: Activation::Tanh,
                },
            ],
            ..default_config()
        };
        let actor: PcActor = PcActor::new(CpuLinAlg::new(), config, &mut rng).unwrap();
        let result = actor.infer(&[0.5; 9]);
        for &v in &result.y_conv {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_residual_tanh_components_populated() {
        let mut rng = make_rng();
        let actor: PcActor =
            PcActor::new(CpuLinAlg::new(), residual_two_hidden_config(), &mut rng).unwrap();
        let result = actor.infer(&[0.5; 9]);
        assert_eq!(result.tanh_components.len(), 2);
        assert!(result.tanh_components[0].is_none()); // layer 0: no skip
        assert!(result.tanh_components[1].is_some()); // layer 1: has skip
        assert_eq!(result.tanh_components[1].as_ref().unwrap().len(), 27);
    }

    #[test]
    fn test_residual_pc_prediction_uses_tanh_component_not_full_state() {
        // With rezero_init=1.0, h[1] = tanh_out + h[0] (significantly different
        // from tanh_out alone). If PC prediction uses h[1] instead of tanh_out,
        // the surprise score and convergence will differ.
        // Two runs with same weights: one with alpha=0 (no PC), one with alpha>0.
        // The PC loop should converge meaningfully (surprise decreases).
        let mut rng = make_rng();
        let config = PcActorConfig {
            rezero_init: 1.0,
            alpha: 0.1,
            max_steps: 20,
            tol: 0.001,
            min_steps: 1,
            ..residual_two_hidden_config()
        };
        let actor: PcActor = PcActor::new(CpuLinAlg::new(), config, &mut rng).unwrap();
        let result = actor.infer(&[1.0, -1.0, 0.5, -0.5, 0.0, 1.0, -1.0, 0.5, -0.5]);
        // With proper PC predictions, surprise should be finite and non-negative
        assert!(result.surprise_score.is_finite());
        assert!(result.surprise_score >= 0.0);
        // Prediction errors should all be finite
        for errors in &result.prediction_errors {
            for &e in errors {
                assert!(e.is_finite(), "PC prediction error not finite: {e}");
            }
        }
    }

    // ── Residual Backward Tests ────────────────────────────────

    #[test]
    fn test_residual_false_update_identical_to_non_residual() {
        let input = vec![1.0, -1.0, 0.5, -0.5, 0.0, 1.0, -1.0, 0.5, -0.5];
        let delta = vec![0.1; 9];

        let mut rng1 = make_rng();
        let mut actor1: PcActor =
            PcActor::new(CpuLinAlg::new(), two_hidden_config(), &mut rng1).unwrap();
        let infer1 = actor1.infer(&input);
        actor1.update_weights(&delta, &infer1, &input, 1.0, &[]);

        let mut rng2 = make_rng();
        let config2 = PcActorConfig {
            residual: false,
            ..two_hidden_config()
        };
        let mut actor2: PcActor = PcActor::new(CpuLinAlg::new(), config2, &mut rng2).unwrap();
        let infer2 = actor2.infer(&input);
        actor2.update_weights(&delta, &infer2, &input, 1.0, &[]);

        for i in 0..actor1.layers.len() {
            assert_eq!(actor1.layers[i].weights.data, actor2.layers[i].weights.data);
        }
    }

    #[test]
    fn test_residual_update_changes_all_layer_weights() {
        let mut rng = make_rng();
        let mut actor: PcActor =
            PcActor::new(CpuLinAlg::new(), residual_two_hidden_config(), &mut rng).unwrap();
        let input = vec![0.5; 9];
        let infer_result = actor.infer(&input);
        let w0 = actor.layers[0].weights.data.clone();
        let w1 = actor.layers[1].weights.data.clone();
        let w2 = actor.layers[2].weights.data.clone();
        actor.update_weights(&[0.1; 9], &infer_result, &input, 1.0, &[]);
        assert_ne!(actor.layers[0].weights.data, w0, "Layer 0 should change");
        assert_ne!(actor.layers[1].weights.data, w1, "Layer 1 should change");
        assert_ne!(
            actor.layers[2].weights.data, w2,
            "Output layer should change"
        );
    }

    #[test]
    fn test_residual_update_changes_rezero_alpha() {
        let mut rng = make_rng();
        let mut actor: PcActor =
            PcActor::new(CpuLinAlg::new(), residual_two_hidden_config(), &mut rng).unwrap();
        let input = vec![0.5; 9];
        let infer_result = actor.infer(&input);
        let alpha_before = actor.rezero_alpha.clone();
        actor.update_weights(&[0.1; 9], &infer_result, &input, 1.0, &[]);
        assert_ne!(
            actor.rezero_alpha, alpha_before,
            "rezero_alpha should be updated by backprop"
        );
    }

    #[test]
    fn test_residual_update_clips_weights() {
        let mut rng = make_rng();
        let mut actor: PcActor =
            PcActor::new(CpuLinAlg::new(), residual_two_hidden_config(), &mut rng).unwrap();
        let input = vec![1.0; 9];
        let infer_result = actor.infer(&input);
        actor.update_weights(&[1e6; 9], &infer_result, &input, 1.0, &[]);
        for layer in &actor.layers {
            for &w in &layer.weights.data {
                assert!(
                    w.abs() <= WEIGHT_CLIP + 1e-12,
                    "Weight {w} exceeds WEIGHT_CLIP"
                );
            }
        }
    }

    #[test]
    fn test_residual_gradient_stronger_than_non_residual() {
        let input = vec![1.0, -1.0, 0.5, -0.5, 0.0, 1.0, -1.0, 0.5, -0.5];
        let delta = vec![0.1; 9];

        // Non-residual 2 hidden layers (27, 27)
        let mut rng1 = make_rng();
        let config1 = PcActorConfig {
            hidden_layers: vec![
                LayerDef {
                    size: 27,
                    activation: Activation::Tanh,
                },
                LayerDef {
                    size: 27,
                    activation: Activation::Tanh,
                },
            ],
            ..default_config()
        };
        let mut actor1: PcActor = PcActor::new(CpuLinAlg::new(), config1, &mut rng1).unwrap();
        let w0_before1 = actor1.layers[0].weights.data.clone();
        let infer1 = actor1.infer(&input);
        actor1.update_weights(&delta, &infer1, &input, 1.0, &[]);
        let change1: f64 = actor1.layers[0]
            .weights
            .data
            .iter()
            .zip(w0_before1.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();

        // Residual 2 hidden layers (27, 27) with rezero_init=1.0
        let mut rng2 = make_rng();
        let config2 = PcActorConfig {
            rezero_init: 1.0,
            ..residual_two_hidden_config()
        };
        let mut actor2: PcActor = PcActor::new(CpuLinAlg::new(), config2, &mut rng2).unwrap();
        let w0_before2 = actor2.layers[0].weights.data.clone();
        let infer2 = actor2.infer(&input);
        actor2.update_weights(&delta, &infer2, &input, 1.0, &[]);
        let change2: f64 = actor2.layers[0]
            .weights
            .data
            .iter()
            .zip(w0_before2.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();

        assert!(
            change2 > change1,
            "Residual should propagate stronger gradient to layer 0: residual={change2:.6}, non-residual={change1:.6}"
        );
    }

    #[test]
    fn test_residual_hybrid_lambda_works() {
        let mut rng = make_rng();
        let config = PcActorConfig {
            local_lambda: 0.99,
            ..residual_two_hidden_config()
        };
        let mut actor: PcActor = PcActor::new(CpuLinAlg::new(), config, &mut rng).unwrap();
        let input = vec![0.5; 9];
        let infer_result = actor.infer(&input);
        let w0_before = actor.layers[0].weights.data.clone();
        actor.update_weights(&[0.1; 9], &infer_result, &input, 1.0, &[]);
        assert_ne!(actor.layers[0].weights.data, w0_before);
    }

    fn local_learning_config() -> PcActorConfig {
        PcActorConfig {
            local_lambda: 0.0,
            ..default_config()
        }
    }

    #[test]
    fn test_infer_prediction_errors_count_matches_hidden_layers() {
        let mut rng = make_rng();
        let actor: PcActor = PcActor::new(CpuLinAlg::new(), default_config(), &mut rng).unwrap();
        let result = actor.infer(&[0.0; 9]);
        assert_eq!(result.prediction_errors.len(), 1);
    }

    #[test]
    fn test_infer_prediction_errors_two_hidden() {
        let mut rng = make_rng();
        let actor: PcActor = PcActor::new(CpuLinAlg::new(), two_hidden_config(), &mut rng).unwrap();
        let result = actor.infer(&[0.0; 9]);
        assert_eq!(result.prediction_errors.len(), 2);
    }

    #[test]
    fn test_infer_prediction_errors_zero_hidden_is_empty() {
        let mut rng = make_rng();
        let config = PcActorConfig {
            hidden_layers: vec![],
            ..default_config()
        };
        let actor: PcActor = PcActor::new(CpuLinAlg::new(), config, &mut rng).unwrap();
        let result = actor.infer(&[0.5; 9]);
        assert!(result.prediction_errors.is_empty());
    }

    #[test]
    fn test_infer_prediction_errors_all_finite() {
        let mut rng = make_rng();
        let actor: PcActor = PcActor::new(CpuLinAlg::new(), default_config(), &mut rng).unwrap();
        let result = actor.infer(&[1.0, -1.0, 0.5, -0.5, 0.0, 1.0, -1.0, 0.5, -0.5]);
        for errors in &result.prediction_errors {
            for &e in errors {
                assert!(e.is_finite(), "prediction error not finite: {e}");
            }
        }
    }

    #[test]
    fn test_infer_prediction_errors_size_matches_hidden_layer_size() {
        let mut rng = make_rng();
        let actor: PcActor = PcActor::new(CpuLinAlg::new(), default_config(), &mut rng).unwrap();
        let result = actor.infer(&[0.0; 9]);
        // default_config has one hidden layer of size 18
        assert_eq!(result.prediction_errors[0].len(), 18);
    }

    #[test]
    fn test_local_learning_config_accepted() {
        let mut rng = make_rng();
        let config = local_learning_config();
        assert!((config.local_lambda).abs() < f64::EPSILON);
        let actor: Result<PcActor, _> = PcActor::new(CpuLinAlg::new(), config, &mut rng);
        assert!(actor.is_ok());
    }

    #[test]
    fn test_local_learning_update_changes_weights() {
        let mut rng = make_rng();
        let mut actor: PcActor =
            PcActor::new(CpuLinAlg::new(), local_learning_config(), &mut rng).unwrap();
        let input = vec![1.0, -1.0, 0.5, -0.5, 0.0, 1.0, -1.0, 0.5, -0.5];
        let infer_result = actor.infer(&input);
        let weights_before = actor.layers[0].weights.data.clone();
        let delta = vec![0.1; 9];
        actor.update_weights(&delta, &infer_result, &input, 1.0, &[]);
        assert_ne!(actor.layers[0].weights.data, weights_before);
    }

    #[test]
    fn test_local_learning_clips_weights() {
        let mut rng = make_rng();
        let mut actor: PcActor =
            PcActor::new(CpuLinAlg::new(), local_learning_config(), &mut rng).unwrap();
        let input = vec![1.0; 9];
        let infer_result = actor.infer(&input);
        let delta = vec![1e6; 9];
        actor.update_weights(&delta, &infer_result, &input, 1.0, &[]);
        for layer in &actor.layers {
            for &w in &layer.weights.data {
                assert!(
                    w.abs() <= WEIGHT_CLIP + 1e-12,
                    "Weight {w} exceeds WEIGHT_CLIP"
                );
            }
        }
    }

    #[test]
    fn test_local_learning_two_hidden_changes_both() {
        let mut rng = make_rng();
        let config = PcActorConfig {
            local_lambda: 0.0,
            ..two_hidden_config()
        };
        let mut actor: PcActor = PcActor::new(CpuLinAlg::new(), config, &mut rng).unwrap();
        let input = vec![0.5; 9];
        let infer_result = actor.infer(&input);
        let w0_before = actor.layers[0].weights.data.clone();
        let w1_before = actor.layers[1].weights.data.clone();
        let delta = vec![0.1; 9];
        actor.update_weights(&delta, &infer_result, &input, 1.0, &[]);
        assert_ne!(
            actor.layers[0].weights.data, w0_before,
            "Layer 0 should change"
        );
        assert_ne!(
            actor.layers[1].weights.data, w1_before,
            "Layer 1 should change"
        );
    }

    #[test]
    fn test_local_learning_differs_from_backprop() {
        let input = vec![1.0, -1.0, 0.5, -0.5, 0.0, 1.0, -1.0, 0.5, -0.5];
        let delta = vec![0.1; 9];

        // Backprop actor
        let mut rng1 = make_rng();
        let mut bp_actor: PcActor =
            PcActor::new(CpuLinAlg::new(), default_config(), &mut rng1).unwrap();
        let bp_infer = bp_actor.infer(&input);
        bp_actor.update_weights(&delta, &bp_infer, &input, 1.0, &[]);

        // Local learning actor (same initial weights)
        let mut rng2 = make_rng();
        let mut ll_actor: PcActor =
            PcActor::new(CpuLinAlg::new(), local_learning_config(), &mut rng2).unwrap();
        let ll_infer = ll_actor.infer(&input);
        ll_actor.update_weights(&delta, &ll_infer, &input, 1.0, &[]);

        // Hidden layer weights should differ between the two approaches
        assert_ne!(
            bp_actor.layers[0].weights.data, ll_actor.layers[0].weights.data,
            "Local learning should produce different weight updates than backprop"
        );
    }

    // ── Hybrid Learning (local_lambda) Tests ────────────────────

    fn hybrid_config(lambda: f64) -> PcActorConfig {
        PcActorConfig {
            local_lambda: lambda,
            ..default_config()
        }
    }

    #[test]
    fn test_local_lambda_one_equals_backprop() {
        let input = vec![1.0, -1.0, 0.5, -0.5, 0.0, 1.0, -1.0, 0.5, -0.5];
        let delta = vec![0.1; 9];

        // Pure backprop (local_learning=false, default)
        let mut rng1 = make_rng();
        let mut bp_actor: PcActor =
            PcActor::new(CpuLinAlg::new(), default_config(), &mut rng1).unwrap();
        let bp_infer = bp_actor.infer(&input);
        bp_actor.update_weights(&delta, &bp_infer, &input, 1.0, &[]);

        // lambda=1.0 should be identical to backprop
        let mut rng2 = make_rng();
        let mut lam_actor: PcActor =
            PcActor::new(CpuLinAlg::new(), hybrid_config(1.0), &mut rng2).unwrap();
        let lam_infer = lam_actor.infer(&input);
        lam_actor.update_weights(&delta, &lam_infer, &input, 1.0, &[]);

        assert_eq!(
            bp_actor.layers[0].weights.data, lam_actor.layers[0].weights.data,
            "lambda=1.0 should produce identical weights to pure backprop"
        );
    }

    #[test]
    fn test_local_lambda_zero_equals_local_learning() {
        let input = vec![1.0, -1.0, 0.5, -0.5, 0.0, 1.0, -1.0, 0.5, -0.5];
        let delta = vec![0.1; 9];

        // Pure local (local_learning=true)
        let mut rng1 = make_rng();
        let mut ll_actor: PcActor =
            PcActor::new(CpuLinAlg::new(), local_learning_config(), &mut rng1).unwrap();
        let ll_infer = ll_actor.infer(&input);
        ll_actor.update_weights(&delta, &ll_infer, &input, 1.0, &[]);

        // lambda=0.0 should be identical to pure local
        let mut rng2 = make_rng();
        let mut lam_actor: PcActor =
            PcActor::new(CpuLinAlg::new(), hybrid_config(0.0), &mut rng2).unwrap();
        let lam_infer = lam_actor.infer(&input);
        lam_actor.update_weights(&delta, &lam_infer, &input, 1.0, &[]);

        assert_eq!(
            ll_actor.layers[0].weights.data, lam_actor.layers[0].weights.data,
            "lambda=0.0 should produce identical weights to pure local learning"
        );
    }

    #[test]
    fn test_local_lambda_half_differs_from_both_pure_modes() {
        let input = vec![1.0, -1.0, 0.5, -0.5, 0.0, 1.0, -1.0, 0.5, -0.5];
        let delta = vec![0.1; 9];

        // Pure backprop
        let mut rng1 = make_rng();
        let mut bp_actor: PcActor =
            PcActor::new(CpuLinAlg::new(), default_config(), &mut rng1).unwrap();
        let bp_infer = bp_actor.infer(&input);
        bp_actor.update_weights(&delta, &bp_infer, &input, 1.0, &[]);

        // Pure local
        let mut rng2 = make_rng();
        let mut ll_actor: PcActor =
            PcActor::new(CpuLinAlg::new(), local_learning_config(), &mut rng2).unwrap();
        let ll_infer = ll_actor.infer(&input);
        ll_actor.update_weights(&delta, &ll_infer, &input, 1.0, &[]);

        // Hybrid lambda=0.5
        let mut rng3 = make_rng();
        let mut hy_actor: PcActor =
            PcActor::new(CpuLinAlg::new(), hybrid_config(0.5), &mut rng3).unwrap();
        let hy_infer = hy_actor.infer(&input);
        hy_actor.update_weights(&delta, &hy_infer, &input, 1.0, &[]);

        assert_ne!(
            hy_actor.layers[0].weights.data, bp_actor.layers[0].weights.data,
            "lambda=0.5 should differ from pure backprop"
        );
        assert_ne!(
            hy_actor.layers[0].weights.data, ll_actor.layers[0].weights.data,
            "lambda=0.5 should differ from pure local"
        );
    }

    #[test]
    fn test_local_lambda_changes_weights() {
        let mut rng = make_rng();
        let mut actor: PcActor =
            PcActor::new(CpuLinAlg::new(), hybrid_config(0.5), &mut rng).unwrap();
        let input = vec![1.0, -1.0, 0.5, -0.5, 0.0, 1.0, -1.0, 0.5, -0.5];
        let infer_result = actor.infer(&input);
        let weights_before = actor.layers[0].weights.data.clone();
        let delta = vec![0.1; 9];
        actor.update_weights(&delta, &infer_result, &input, 1.0, &[]);
        assert_ne!(actor.layers[0].weights.data, weights_before);
    }

    #[test]
    fn test_local_lambda_clips_weights() {
        let mut rng = make_rng();
        let mut actor: PcActor =
            PcActor::new(CpuLinAlg::new(), hybrid_config(0.5), &mut rng).unwrap();
        let input = vec![1.0; 9];
        let infer_result = actor.infer(&input);
        let delta = vec![1e6; 9];
        actor.update_weights(&delta, &infer_result, &input, 1.0, &[]);
        for layer in &actor.layers {
            for &w in &layer.weights.data {
                assert!(
                    w.abs() <= WEIGHT_CLIP + 1e-12,
                    "Weight {w} exceeds WEIGHT_CLIP"
                );
            }
        }
    }

    #[test]
    fn test_local_lambda_negative_returns_error() {
        let mut rng = make_rng();
        let config = hybrid_config(-0.1);
        let result: Result<PcActor, _> = PcActor::new(CpuLinAlg::new(), config, &mut rng);
        assert!(result.is_err());
    }

    #[test]
    fn test_local_lambda_above_one_returns_error() {
        let mut rng = make_rng();
        let config = hybrid_config(1.1);
        let result: Result<PcActor, _> = PcActor::new(CpuLinAlg::new(), config, &mut rng);
        assert!(result.is_err());
    }

    // ── Phase 5 Cycle 5.1: Crossover same topology ─────────────

    // ── Phase 5 Cycle 5.2: Crossover child smaller ──────────────

    // ── Phase 5 Cycle 5.3: Crossover parents differ ─────────────

    // ── Phase 5 Cycle 5.4: Crossover child larger ───────────────

    // ── Phase 5 Cycle 5.5: Crossover layer count mismatch ───────

    // ── Phase 5 Cycle 5.6: Crossover residual components ────────

    // ── Fix #1: Column permutation propagation ──────────────────

    // ── Fix #5: Empty hidden_layers guard ────────────────────────

    // ── from_weights dimension validation tests ──────────────────────

    /// Helper: build valid PcActorWeights from a config by constructing
    /// an actor and extracting its weights.
    fn valid_weights_for(config: &PcActorConfig) -> crate::serializer::PcActorWeights {
        let mut rng = make_rng();
        let actor = PcActor::<CpuLinAlg>::new(CpuLinAlg::new(), config.clone(), &mut rng).unwrap();
        actor.to_weights()
    }

    #[test]
    fn test_from_weights_valid_returns_ok() {
        let config = default_config();
        let weights = valid_weights_for(&config);
        let result = PcActor::<CpuLinAlg>::from_weights(CpuLinAlg::new(), config, weights);
        assert!(result.is_ok());
    }

    #[test]
    fn test_from_weights_wrong_weight_rows_returns_err() {
        let config = default_config(); // input=9, hidden=[18], output=9
        let mut weights = valid_weights_for(&config);
        // Layer 0 should be 18x9; corrupt rows to 10x9
        weights.layers[0].weights = crate::matrix::Matrix::zeros(10, 9);
        weights.layers[0].bias = vec![0.0; 10];
        let result = PcActor::<CpuLinAlg>::from_weights(CpuLinAlg::new(), config, weights);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, PcError::DimensionMismatch { .. }),
            "Expected DimensionMismatch, got: {err}"
        );
    }

    #[test]
    fn test_from_weights_wrong_weight_cols_returns_err() {
        let config = default_config(); // input=9, hidden=[18], output=9
        let mut weights = valid_weights_for(&config);
        // Layer 0 should be 18x9; corrupt cols to 18x5
        weights.layers[0].weights = crate::matrix::Matrix::zeros(18, 5);
        let result = PcActor::<CpuLinAlg>::from_weights(CpuLinAlg::new(), config, weights);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, PcError::DimensionMismatch { .. }),
            "Expected DimensionMismatch, got: {err}"
        );
    }

    #[test]
    fn test_from_weights_wrong_bias_length_returns_err() {
        let config = default_config(); // hidden=[18], so layer 0 bias should be len 18
        let mut weights = valid_weights_for(&config);
        weights.layers[0].bias = vec![0.0; 5]; // wrong length
        let result = PcActor::<CpuLinAlg>::from_weights(CpuLinAlg::new(), config, weights);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, PcError::DimensionMismatch { .. }),
            "Expected DimensionMismatch, got: {err}"
        );
    }

    #[test]
    fn test_from_weights_wrong_output_layer_dims_returns_err() {
        let config = default_config(); // output layer should be 9x18
        let mut weights = valid_weights_for(&config);
        let last = weights.layers.len() - 1;
        weights.layers[last].weights = crate::matrix::Matrix::zeros(9, 10); // wrong cols
        let result = PcActor::<CpuLinAlg>::from_weights(CpuLinAlg::new(), config, weights);
        assert!(result.is_err());
    }

    #[test]
    fn test_from_weights_wrong_rezero_alpha_count_returns_err() {
        let mut config = default_config();
        config.hidden_layers = vec![
            LayerDef {
                size: 18,
                activation: Activation::Tanh,
            },
            LayerDef {
                size: 18,
                activation: Activation::Tanh,
            },
        ];
        config.residual = true;
        let mut weights = valid_weights_for(&config);
        // residual with 2 hidden layers expects 1 rezero_alpha; give 0
        weights.rezero_alpha = vec![];
        let result = PcActor::<CpuLinAlg>::from_weights(CpuLinAlg::new(), config, weights);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, PcError::DimensionMismatch { .. }),
            "Expected DimensionMismatch, got: {err}"
        );
    }

    #[test]
    fn test_from_weights_wrong_skip_projection_dims_returns_err() {
        // N1: skip projection dimensions (rows/cols) should be validated
        let mut config = default_config();
        config.hidden_layers = vec![
            LayerDef {
                size: 27,
                activation: Activation::Softsign,
            },
            LayerDef {
                size: 18,
                activation: Activation::Softsign,
            },
        ];
        config.residual = true;
        let mut weights = valid_weights_for(&config);
        // Skip projection should be 18x27; corrupt to 10x5
        weights.skip_projections[0] = Some(crate::matrix::Matrix::zeros(10, 5));
        let result = PcActor::<CpuLinAlg>::from_weights(CpuLinAlg::new(), config, weights);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, PcError::DimensionMismatch { .. }),
            "Expected DimensionMismatch, got: {err}"
        );
    }

    #[test]
    fn test_from_weights_wrong_skip_projections_count_returns_err() {
        let mut config = default_config();
        config.hidden_layers = vec![
            LayerDef {
                size: 18,
                activation: Activation::Tanh,
            },
            LayerDef {
                size: 18,
                activation: Activation::Tanh,
            },
        ];
        config.residual = true;
        let mut weights = valid_weights_for(&config);
        // Should have 1 skip_projection; give 3
        weights.skip_projections = vec![None, None, None];
        let result = PcActor::<CpuLinAlg>::from_weights(CpuLinAlg::new(), config, weights);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, PcError::DimensionMismatch { .. }),
            "Expected DimensionMismatch, got: {err}"
        );
    }

    // ── Polyak Update Tests ─────────────────────────────────────────

    /// Helper: creates two actors with the same config but different seeds.
    fn make_two_actors(config: &PcActorConfig) -> (PcActor, PcActor) {
        let mut rng_a = StdRng::seed_from_u64(42);
        let mut rng_b = StdRng::seed_from_u64(99);
        let a = PcActor::new(CpuLinAlg::new(), config.clone(), &mut rng_a).unwrap();
        let b = PcActor::new(CpuLinAlg::new(), config.clone(), &mut rng_b).unwrap();
        (a, b)
    }

    /// Helper: collects all weight and bias values from an actor into a flat vector.
    fn snapshot_weights(actor: &PcActor) -> Vec<f64> {
        let b = &actor.backend;
        let mut vals = Vec::new();
        for layer in &actor.layers {
            let rows = b.mat_rows(&layer.weights);
            let cols = b.mat_cols(&layer.weights);
            for r in 0..rows {
                for c in 0..cols {
                    vals.push(b.mat_get(&layer.weights, r, c));
                }
            }
            let len = b.vec_len(&layer.bias);
            for i in 0..len {
                vals.push(b.vec_get(&layer.bias, i));
            }
        }
        for &alpha in &actor.rezero_alpha {
            vals.push(alpha);
        }
        for m in actor.skip_projections.iter().flatten() {
            let rows = b.mat_rows(m);
            let cols = b.mat_cols(m);
            for r in 0..rows {
                for c in 0..cols {
                    vals.push(b.mat_get(m, r, c));
                }
            }
        }
        vals
    }

    #[test]
    fn test_polyak_update_tau_zero_no_change() {
        let (mut delayed, other) = make_two_actors(&default_config());
        let before = snapshot_weights(&delayed);
        delayed.polyak_update_from(&other, 0.0).unwrap();
        let after = snapshot_weights(&delayed);
        assert_eq!(before, after, "tau=0 must leave weights unchanged");
    }

    #[test]
    fn test_polyak_update_tau_one_full_copy() {
        let (mut delayed, other) = make_two_actors(&default_config());
        delayed.polyak_update_from(&other, 1.0).unwrap();
        let delayed_snap = snapshot_weights(&delayed);
        let other_snap = snapshot_weights(&other);
        assert_eq!(
            delayed_snap, other_snap,
            "tau=1 must produce weights identical to other"
        );
    }

    #[test]
    fn test_polyak_update_partial_interpolation() {
        let config = default_config();
        let (mut delayed, other) = make_two_actors(&config);
        let before = snapshot_weights(&delayed);
        let other_snap = snapshot_weights(&other);
        let tau = 0.5;
        delayed.polyak_update_from(&other, tau).unwrap();
        let after = snapshot_weights(&delayed);
        for (i, (&a, (&b, &o))) in after
            .iter()
            .zip(before.iter().zip(other_snap.iter()))
            .enumerate()
        {
            let expected = tau * o + (1.0 - tau) * b;
            assert!(
                (a - expected).abs() < 1e-12,
                "tau=0.5 mismatch at index {i}: got {a}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_polyak_update_rejects_topology_mismatch() {
        let (mut actor_a, _) = make_two_actors(&default_config());
        let mut rng = StdRng::seed_from_u64(77);
        let actor_b = PcActor::new(CpuLinAlg::new(), two_hidden_config(), &mut rng).unwrap();
        let result = actor_a.polyak_update_from(&actor_b, 0.5);
        assert!(result.is_err());
        assert!(
            matches!(result.unwrap_err(), PcError::DimensionMismatch { .. }),
            "Expected DimensionMismatch for topology mismatch"
        );
    }

    #[test]
    fn test_copy_weights_from_exact() {
        let (mut delayed, other) = make_two_actors(&default_config());
        delayed.copy_weights_from(&other).unwrap();
        let delayed_snap = snapshot_weights(&delayed);
        let other_snap = snapshot_weights(&other);
        assert_eq!(
            delayed_snap, other_snap,
            "copy_weights_from must produce byte-exact equality"
        );
    }

    #[test]
    fn test_copy_weights_from_rejects_topology_mismatch() {
        let (mut actor_a, _) = make_two_actors(&default_config());
        let mut rng = StdRng::seed_from_u64(77);
        let actor_b = PcActor::new(CpuLinAlg::new(), two_hidden_config(), &mut rng).unwrap();
        let result = actor_a.copy_weights_from(&actor_b);
        assert!(result.is_err());
        assert!(
            matches!(result.unwrap_err(), PcError::DimensionMismatch { .. }),
            "Expected DimensionMismatch for topology mismatch"
        );
    }

    fn residual_hetero_config() -> PcActorConfig {
        PcActorConfig {
            residual: true,
            hidden_layers: vec![
                LayerDef {
                    size: 18,
                    activation: Activation::Softsign,
                },
                LayerDef {
                    size: 12,
                    activation: Activation::Softsign,
                },
            ],
            ..default_config()
        }
    }

    /// Drifts all layer weights, biases, rezero alphas, and skip projections
    /// by a fixed offset to create a measurable difference between actors.
    fn drift_actor_weights(actor: &mut PcActor, offset: f64) {
        let b = &actor.backend;
        for layer in &mut actor.layers {
            let rows = b.mat_rows(&layer.weights);
            let cols = b.mat_cols(&layer.weights);
            for r in 0..rows {
                for c in 0..cols {
                    let v = b.mat_get(&layer.weights, r, c);
                    b.mat_set(&mut layer.weights, r, c, v + offset);
                }
            }
            let len = b.vec_len(&layer.bias);
            for i in 0..len {
                let v = b.vec_get(&layer.bias, i);
                b.vec_set(&mut layer.bias, i, v + offset);
            }
        }
        for alpha in &mut actor.rezero_alpha {
            *alpha += offset;
        }
        for m in actor.skip_projections.iter_mut().flatten() {
            let rows = b.mat_rows(m);
            let cols = b.mat_cols(m);
            for r in 0..rows {
                for c in 0..cols {
                    let v = b.mat_get(m, r, c);
                    b.mat_set(m, r, c, v + offset);
                }
            }
        }
    }

    #[test]
    fn test_polyak_update_with_residual_actors() {
        let config = residual_hetero_config();
        let (mut target, mut source) = make_two_actors(&config);

        // Verify skip projections are allocated (heterogeneous sizes)
        assert_eq!(target.skip_projections.len(), 1);
        assert!(target.skip_projections[0].is_some());

        // Drift source weights so target and source differ measurably
        drift_actor_weights(&mut source, 1.0);

        let before = snapshot_weights(&target);
        let source_snap = snapshot_weights(&source);
        let tau = 0.5;

        target.polyak_update_from(&source, tau).unwrap();

        let after = snapshot_weights(&target);
        // snapshot_weights includes layers, rezero_alpha, AND skip_projections
        assert_eq!(before.len(), after.len(), "snapshot length must not change");
        for (i, (&a, (&b, &o))) in after
            .iter()
            .zip(before.iter().zip(source_snap.iter()))
            .enumerate()
        {
            let expected = tau * o + (1.0 - tau) * b;
            assert!(
                (a - expected).abs() < 1e-12,
                "residual polyak mismatch at index {i}: got {a}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_copy_weights_from_with_residual_actors() {
        let config = residual_hetero_config();
        let (mut target, mut source) = make_two_actors(&config);

        // Verify skip projections are allocated (heterogeneous sizes)
        assert_eq!(target.skip_projections.len(), 1);
        assert!(target.skip_projections[0].is_some());

        // Drift source weights so target and source differ
        drift_actor_weights(&mut source, 1.0);

        target.copy_weights_from(&source).unwrap();

        let target_snap = snapshot_weights(&target);
        let source_snap = snapshot_weights(&source);
        assert_eq!(
            target_snap, source_snap,
            "copy_weights_from must produce byte-exact equality for residual actors"
        );
    }
}
