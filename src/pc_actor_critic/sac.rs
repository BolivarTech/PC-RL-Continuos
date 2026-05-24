// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-05-24

//! SAC orchestration for canonical Soft Actor-Critic continuous mode (v6.0.0).
//!
//! Houses methods that operate on the twin Q-critics and their Polyak-averaged
//! target copies.  The critic/actor/temperature update logic and the off-policy
//! learning loop are added in later tasks (T9–T12).
//!
//! # Polyak update rule
//!
//! ```text
//! θ_target ← (1 − τ) · θ_target + τ · θ_live
//! ```
//!
//! where `τ = config.polyak_tau` (default 0.005).  Both `q1_target` and
//! `q2_target` are updated simultaneously.  This is a no-op when the agent
//! has no SAC critics (discrete mode).

use crate::layer::Layer;
use crate::linalg::LinAlg;
use crate::pc_actor_critic::PcActorCritic;

impl<L: LinAlg> PcActorCritic<L> {
    /// Current SAC entropy temperature `α = exp(log_alpha)`.
    ///
    /// Always `> 0` by construction (`exp` is strictly positive).
    /// Used as the entropy coefficient in the SAC policy-loss and
    /// temperature-update rule.
    ///
    /// # Examples
    ///
    /// ```
    /// // α = exp(0.0) = 1.0 initially (log_alpha_init = 0.0 default).
    /// ```
    #[allow(dead_code)] // wired in T10-T12
    pub(crate) fn alpha(&self) -> f64 {
        self.log_alpha.exp()
    }

    /// Automatic temperature update toward the target entropy `H_target`.
    ///
    /// Updates `log_alpha` by gradient descent on the dual objective
    /// `J(α) = −α · (logp_mean + H_target)`, where
    /// `H_target = config.target_entropy.unwrap_or(-(action_dim as f64))`.
    ///
    /// Update rule:
    /// ```text
    /// grad     = −α · (logp_mean + H_target)
    /// log_alpha -= alpha_lr · grad
    /// log_alpha  = clamp(log_alpha, LOG_ALPHA_MIN, LOG_ALPHA_MAX)
    /// ```
    ///
    /// When entropy `= −logp_mean < H_target` (i.e. `logp_mean + H_target > 0`),
    /// `grad < 0` → `log_alpha` increases → `α` rises to encourage exploration.
    /// When entropy `> H_target`, `α` falls to let the policy sharpen.
    ///
    /// `log_alpha` is clamped to `[LOG_ALPHA_MIN, LOG_ALPHA_MAX]` after each step
    /// to keep `α = exp(log_alpha)` numerically finite under extreme inputs.
    /// Non-finite gradients are silently dropped (NaN-safety).
    ///
    /// # Arguments
    ///
    /// * `logp_mean` — mean log-probability of sampled actions under the
    ///   current policy (batch average of `log π(a|s)`).
    #[allow(dead_code)] // wired in T10-T12
    pub(crate) fn sac_temperature_update(&mut self, logp_mean: f64) {
        /// Lower bound on `log_alpha`; `exp(-20) ≈ 2e-9` (effectively zero temperature).
        const LOG_ALPHA_MIN: f64 = -20.0;
        /// Upper bound on `log_alpha`; `exp(20) ≈ 4.9e8` (very high entropy pressure).
        const LOG_ALPHA_MAX: f64 = 20.0;

        let action_dim = self
            .config
            .q_critic
            .as_ref()
            .map(|q| q.action_dim)
            .unwrap_or(0);
        let h_target = self.config.target_entropy.unwrap_or(-(action_dim as f64));
        let grad = -(self.log_alpha.exp()) * (logp_mean + h_target);
        if grad.is_finite() {
            self.log_alpha -= self.config.alpha_lr * grad;
            self.log_alpha = self.log_alpha.clamp(LOG_ALPHA_MIN, LOG_ALPHA_MAX);
        }
    }

    /// Test helper: returns current `α = exp(log_alpha)`.
    ///
    /// Thin wrapper around [`alpha`](Self::alpha) so tests can call it without
    /// the `#[allow(dead_code)]` suppression.
    #[cfg(test)]
    pub(crate) fn alpha_for_test(&self) -> f64 {
        self.alpha()
    }

    /// Returns `true` when SAC twin Q-critics are present (continuous SAC mode).
    ///
    /// Equivalent to `self.q1.is_some()`.  Used by tests and future task
    /// callers to gate SAC-specific code paths without pattern-matching on all
    /// four Option fields.
    #[allow(dead_code)] // called by tests + wired in T10
    pub(crate) fn has_sac_critics(&self) -> bool {
        self.q1.is_some()
    }

    /// Soft Polyak update of both target Q-critics toward their live counterparts.
    ///
    /// For each `(q_target, q_live)` pair applies
    /// `θ_target ← (1 − τ) · θ_target + τ · θ_live` element-wise over all
    /// layer weights and biases, where `τ = self.config.polyak_tau`.
    ///
    /// This is a no-op when no SAC critics are present (`!has_sac_critics()`).
    ///
    /// # Borrow-checker strategy
    ///
    /// `q1_target` and `q1` share `self`.  To avoid a double-borrow we
    /// temporarily `take` each target out of its `Option`, update it against the
    /// live critic (which remains in `self`), then put it back.  The `Option`
    /// slot is `None` only during the few lines of the update; no public method
    /// is called on `self` inside that window, so the invariant is maintained.
    #[allow(dead_code)] // wired in T10
    pub(crate) fn polyak_update_targets(&mut self) {
        if !self.has_sac_critics() {
            return;
        }

        let tau = self.config.polyak_tau;

        // --- q1_target ← blend toward q1 ---
        if let Some(mut target) = self.q1_target.take() {
            if let Some(ref live) = self.q1 {
                for (t_layer, l_layer) in target.layers.iter_mut().zip(live.layers.iter()) {
                    Self::polyak_layer(t_layer, l_layer, tau, &self.backend);
                }
            }
            self.q1_target = Some(target);
        }

        // --- q2_target ← blend toward q2 ---
        if let Some(mut target) = self.q2_target.take() {
            if let Some(ref live) = self.q2 {
                for (t_layer, l_layer) in target.layers.iter_mut().zip(live.layers.iter()) {
                    Self::polyak_layer(t_layer, l_layer, tau, &self.backend);
                }
            }
            self.q2_target = Some(target);
        }
    }

    /// Blends one target layer toward a live layer with Polyak rate `tau`.
    ///
    /// ```text
    /// target_w[r,c] ← (1 − τ) · target_w[r,c] + τ · live_w[r,c]
    /// target_b[i]   ← (1 − τ) · target_b[i]   + τ · live_b[i]
    /// ```
    ///
    /// Pure function on the two layers — no `&self` needed, which avoids
    /// a three-way borrow conflict when the caller already holds `&self.backend`.
    #[allow(dead_code)] // called by polyak_update_targets; wired in T10
    fn polyak_layer(target: &mut Layer<L>, live: &Layer<L>, tau: f64, backend: &L) {
        let one_minus_tau = 1.0 - tau;

        let rows = backend.mat_rows(&target.weights);
        let cols = backend.mat_cols(&target.weights);
        for r in 0..rows {
            for c in 0..cols {
                let t = backend.mat_get(&target.weights, r, c);
                let l = backend.mat_get(&live.weights, r, c);
                backend.mat_set(&mut target.weights, r, c, one_minus_tau * t + tau * l);
            }
        }

        let len = backend.vec_len(&target.bias);
        for i in 0..len {
            let t = backend.vec_get(&target.bias, i);
            let l = backend.vec_get(&live.bias, i);
            backend.vec_set(&mut target.bias, i, one_minus_tau * t + tau * l);
        }
    }
}
