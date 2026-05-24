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
use crate::pc_actor_critic::replay::{Action, ReplayTransition};
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

    /// Computes the soft-Bellman target `y` for a single replay transition.
    ///
    /// Runs PC inference on `next_state` with the LIVE actor to draw a fresh
    /// reparameterized action `a'`, computes `log π(a'|s')`, evaluates both
    /// target Q-critics, and returns:
    ///
    /// ```text
    /// y = r + γ·(1−done)·(min(q1_target(s',a'), q2_target(s',a')) − α·log π(a'|s'))
    /// ```
    ///
    /// Returns `None` if the transition contains a non-Continuous action,
    /// or if any intermediate value (sampled action, log-prob, Q-values,
    /// or final target) is non-finite.  The caller increments
    /// `sac_skipped_critic_updates` for each `None`.
    ///
    /// # Borrow-checker strategy
    ///
    /// Inference (`&self.actor`) and target-critic forward (`&self.q1_target`)
    /// are pure reads; they complete before any mutable borrow of `self.rng`.
    /// We snapshot the actor output as a host `Vec<f64>` and the target Q-values
    /// as scalars, then advance `self.rng` for the fresh action sample — all
    /// before the mutable `q1/q2.update` calls in `sac_critic_update`.
    pub(crate) fn sac_bellman_target(&mut self, t: &ReplayTransition) -> Option<f64> {
        let action_dim = self.config.q_critic.as_ref()?.action_dim;
        let gamma = self.config.gamma;
        let alpha = self.alpha();

        // --- actor inference on next_state (immutable borrow of self.actor) ---
        // Snapshot y_conv as a host Vec<f64> immediately so the borrow ends.
        let y_next = {
            let infer = self.actor.infer(&t.next_state);
            self.backend.vec_to_vec(&infer.y_conv)
        };
        let (mu_n, ls_n) = super::split_mu_log_sigma(&y_next, action_dim);

        // --- sample fresh a' using self.rng (mutable borrow of rng only) ---
        let (a_next_raw, a_next) = super::sample_squashed_action(&mu_n, &ls_n, &mut self.rng);
        let logp_next = super::squashed_log_prob(&mu_n, &ls_n, &a_next_raw);

        // --- target Q-values (immutable borrows of q1_target / q2_target) ---
        let q1t = self.q1_target.as_ref()?.forward(&t.next_state, &a_next);
        let q2t = self.q2_target.as_ref()?.forward(&t.next_state, &a_next);

        let q_next = q1t.min(q2t);
        let done_mask = if t.done { 1.0 } else { 0.0 };
        let y = t.reward + gamma * (1.0 - done_mask) * (q_next - alpha * logp_next);

        // Guard: skip transitions where any intermediate value is non-finite.
        if !a_next.iter().all(|x| x.is_finite())
            || !logp_next.is_finite()
            || !q_next.is_finite()
            || !y.is_finite()
        {
            return None;
        }

        Some(y)
    }

    /// SAC soft-Bellman critic update over a replay batch.
    ///
    /// For each transition in `batch`:
    ///
    /// 1. Compute soft-Bellman target `y` via [`sac_bellman_target`](Self::sac_bellman_target).
    /// 2. Update both live Q-critics toward `y` with MSE loss.
    ///
    /// Returns the mean `0.5 * (MSE_q1 + MSE_q2)` over non-skipped transitions.
    /// Returns `0.0` when the entire batch is skipped.
    ///
    /// # Borrow-checker strategy (two-pass)
    ///
    /// Computing targets requires immutable borrows of `self.actor`,
    /// `self.q1_target`, `self.q2_target`, and a mutable borrow of `self.rng`.
    /// Updating `q1` / `q2` requires mutable borrows of those fields.
    /// Rust forbids simultaneous `&self` and `&mut self` borrows, so we split
    /// into two passes:
    ///
    /// 1. **Target pass** (immutable + rng) — call `sac_bellman_target` for
    ///    every transition, collecting `(executed_a, y)` pairs.
    /// 2. **Update pass** (mutable q1/q2) — iterate the collected pairs and
    ///    call `q1.update` / `q2.update`.
    ///
    /// # Arguments
    ///
    /// * `batch` — Slice of replay transitions (SAC uses Continuous actions).
    ///
    /// # Returns
    ///
    /// Mean MSE loss over the batch.
    #[allow(dead_code)] // wired in T12
    pub(crate) fn sac_critic_update(&mut self, batch: &[ReplayTransition]) -> f64 {
        // --- Pass 1: compute all soft-Bellman targets ---
        // Each entry is (state, executed_a_squashed, next_state is in t, y).
        // We store (state clone, executed_action_squashed, y) to avoid holding
        // references into `batch` across the mutable update pass.
        let mut targets: Vec<(Vec<f64>, Vec<f64>, f64)> = Vec::with_capacity(batch.len());

        for t in batch {
            let a_raw = match &t.action {
                Action::Continuous(v) => v.clone(),
                Action::Discrete(_) => {
                    self.sac_skipped_critic_updates += 1;
                    continue;
                }
            };
            // Executed action is the squashed (tanh) version of the stored pre-squash a_raw.
            let a_squashed: Vec<f64> = a_raw.iter().map(|x| x.tanh()).collect();

            match self.sac_bellman_target(t) {
                Some(y) => targets.push((t.state.clone(), a_squashed, y)),
                None => {
                    self.sac_skipped_critic_updates += 1;
                }
            }
        }

        if targets.is_empty() {
            return 0.0;
        }

        // --- Pass 2: update live Q-critics with collected targets ---
        let mut total = 0.0_f64;
        let n = targets.len() as f64;

        for (state, a_squashed, y) in &targets {
            let l1 = self
                .q1
                .as_mut()
                .expect("sac_critic_update: q1 must be Some in SAC mode")
                .update(state, a_squashed, *y);
            let l2 = self
                .q2
                .as_mut()
                .expect("sac_critic_update: q2 must be Some in SAC mode")
                .update(state, a_squashed, *y);
            total += 0.5 * (l1 + l2);
        }

        total / n
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
