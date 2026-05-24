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
