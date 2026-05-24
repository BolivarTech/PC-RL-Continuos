// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-05-24

//! Action-value critic `Q(s, a)` for canonical SAC (v6.0.0). Input is
//! `state ⊕ action`, output a scalar action-value (Linear). Distinct from
//! `MlpCritic` (the discrete V-critic). `action_gradient` (∇_a Q) is added in T3.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activation::Activation;
    use crate::layer::LayerDef;
    use crate::linalg::cpu::CpuLinAlg;
    use rand::{rngs::StdRng, SeedableRng};

    #[test]
    fn test_qcritic_forward_finite_scalar() {
        let mut rng = StdRng::seed_from_u64(42);
        let cfg = QCriticConfig {
            state_dim: 3,
            action_dim: 1,
            hidden_layers: vec![LayerDef { size: 16, activation: Activation::Tanh }],
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
            hidden_layers: vec![LayerDef { size: 16, activation: Activation::Tanh }],
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
            hidden_layers: vec![LayerDef { size: 16, activation: Activation::Tanh }],
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
