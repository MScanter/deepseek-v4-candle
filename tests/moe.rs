//! MoE routing/expert tests.
//!
//! The Gate is the risky, novel part (V4's `sqrtsoftplus` scoring; bias that shifts the top-k
//! *selection* but not the routing *weights*; non-softmax renormalization; `route_scale`), so it
//! is golden-tested against the pure-Python `moe_gate_golden.py`. The golden config is built so
//! the bias FLIPS the pick for every token — if weights were read post-bias, or selection ignored
//! the bias, the numbers would not match.

use candle_core::{Device, Tensor};
use deepseek_v4_candle::moe::{Expert, Gate, Moe, ScoreFunc};

/// Deterministic `[o, i]` tensor (sin-of-arange), matching `moe_golden.py`'s `det_at` so the MoE
/// combine golden is reproducible without hardcoding every expert weight.
fn det_at(o: usize, i: usize, start: i64, dev: &Device) -> candle_core::Result<Tensor> {
    let n = (o * i) as i64;
    Tensor::arange(start, start + n, dev)?
        .to_dtype(candle_core::DType::F32)?
        .affine(0.7, 1.0)?
        .sin()?
        .affine(0.3, 0.0)?
        .reshape((o, i))
}

/// Build the golden Gate: dim 3, 4 experts, top-2, `route_scale` 1.5, sqrtsoftplus.
fn golden_gate(dev: &Device) -> candle_core::Result<Gate> {
    Ok(Gate {
        weight: Tensor::from_vec(
            vec![0.6f32, -0.2, 0.3, -0.4, 0.5, 0.1, 0.2, 0.3, -0.5, 0.1, -0.1, 0.7],
            (4, 3),
            dev,
        )?,
        bias: Some(Tensor::from_vec(vec![0.05f32, 0.40, -0.30, 0.10], (4,), dev)?),
        topk: 2,
        route_scale: 1.5,
        score_func: ScoreFunc::SqrtSoftplus,
    })
}

/// Golden routing: `sqrt(softplus(x·Wᵀ))`, bias added for top-2 selection only, weights gathered
/// from the *pre-bias* scores then normalized to sum 1 and scaled by 1.5. Both tokens are
/// constructed so the bias flips the selection (raw top-2 ≠ biased top-2).
#[test]
fn gate_routes_with_bias_selection_prebias_weights() -> candle_core::Result<()> {
    let dev = Device::Cpu;
    let gate = golden_gate(&dev)?;
    let x = Tensor::from_vec(vec![0.8f32, -0.5, 0.3, -0.2, 0.9, -0.6], (2, 3), &dev)?;

    let (weights, indices) = gate.route(&x)?;
    assert_eq!(weights.dims(), &[2, 2]);
    assert_eq!(indices.dims(), &[2, 2]);

    // Biased top-2 (flipped from raw): tok0 -> {0,1}, tok1 -> {1,3}.
    assert_eq!(indices.flatten_all()?.to_vec1::<i64>()?, vec![0i64, 1, 1, 3]);

    // Weights from PRE-bias scores at those indices, normalized * route_scale.
    let w = weights.flatten_all()?.to_vec1::<f32>()?;
    let golden = [0.908_507_f32, 0.591_493, 0.884_437, 0.615_563];
    for (k, (a, b)) in w.iter().zip(golden.iter()).enumerate() {
        assert!((a - b).abs() < 1e-5, "weights[{k}] = {a}, expected {b}");
    }
    Ok(())
}

/// Invariant: for the non-softmax score functions the per-token routing weights are renormalized,
/// so each token's weights sum to exactly `route_scale` (here 1.5) regardless of the raw scores.
#[test]
fn gate_weights_sum_to_route_scale() -> candle_core::Result<()> {
    let dev = Device::Cpu;
    let gate = golden_gate(&dev)?;
    let x = Tensor::from_vec(vec![0.8f32, -0.5, 0.3, -0.2, 0.9, -0.6], (2, 3), &dev)?;

    let (weights, _) = gate.route(&x)?;
    let sums = weights.sum(candle_core::D::Minus1)?.flatten_all()?.to_vec1::<f32>()?;
    for (i, s) in sums.iter().enumerate() {
        assert!((s - 1.5).abs() < 1e-5, "token {i} weights sum = {s}, expected 1.5");
    }
    Ok(())
}

/// Build the golden Expert: dim 2, inter 3, given `swiglu_limit`.
fn golden_expert(limit: f64, dev: &Device) -> candle_core::Result<Expert> {
    Ok(Expert {
        w1: Tensor::from_vec(vec![0.5f32, -0.3, 0.2, 0.4, -0.6, 0.1], (3, 2), dev)?,
        w2: Tensor::from_vec(vec![0.3f32, -0.1, 0.6, 0.2, 0.5, -0.3], (2, 3), dev)?,
        w3: Tensor::from_vec(vec![0.1f32, 0.7, -0.2, 0.3, 0.5, -0.4], (3, 2), dev)?,
        swiglu_limit: limit,
    })
}

/// SwiGLU FFN with no clamp and no routing weight: `w2 · (silu(w1·x) * (w3·x))`. Golden from
/// `moe_expert_golden.py`.
#[test]
fn expert_swiglu_ffn() -> candle_core::Result<()> {
    let dev = Device::Cpu;
    let expert = golden_expert(0.0, &dev)?;
    let x = Tensor::from_vec(vec![0.8f32, -0.4], (1, 2), &dev)?;

    let y = expert.forward(&x, None)?;
    assert_eq!(y.dims(), &[1, 2]);

    let got = y.flatten_all()?.to_vec1::<f32>()?;
    let golden = [-0.084_712_f32, 0.019_528];
    for (k, (a, b)) in got.iter().zip(golden.iter()).enumerate() {
        assert!((a - b).abs() < 1e-5, "y[{k}] = {a}, expected {b}");
    }
    Ok(())
}

/// SwiGLU with the asymmetric clamp (`limit = 0.3`: up clamped both sides, gate on max only) plus a
/// per-token routing weight (1.25) applied before the down-projection. Both clamps trigger here
/// (gate `0.52 -> 0.3`, up `0.56 -> 0.3`). Golden from `moe_expert_golden.py`.
#[test]
fn expert_swiglu_clamp_and_weight() -> candle_core::Result<()> {
    let dev = Device::Cpu;
    let expert = golden_expert(0.3, &dev)?;
    let x = Tensor::from_vec(vec![0.8f32, -0.4], (1, 2), &dev)?;
    let w = Tensor::from_vec(vec![1.25f32], (1, 1), &dev)?;

    let y = expert.forward(&x, Some(&w))?;
    let got = y.flatten_all()?.to_vec1::<f32>()?;
    let golden = [-0.056_549_f32, 0.013_195];
    for (k, (a, b)) in got.iter().zip(golden.iter()).enumerate() {
        assert!((a - b).abs() < 1e-5, "y[{k}] = {a}, expected {b}");
    }
    Ok(())
}

// ---- MoE combine (route -> per-expert -> scatter + shared) ----

/// Build the golden MoE: the verified `golden_gate` (dim 3, 4 experts, top-2) plus 4 routed experts
/// and 1 shared expert with deterministic `det_at` weights (offsets matching `moe_golden.py`).
fn golden_moe(dev: &Device) -> candle_core::Result<Moe> {
    let (dim, inter) = (3usize, 3usize);
    let routed = (0..4)
        .map(|i| {
            let o = 1000 + i * 100;
            Ok(Expert {
                w1: det_at(inter, dim, o, dev)?,
                w3: det_at(inter, dim, o + 30, dev)?,
                w2: det_at(dim, inter, o + 60, dev)?,
                swiglu_limit: 0.0,
            })
        })
        .collect::<candle_core::Result<Vec<_>>>()?;
    let shared = Expert {
        w1: det_at(inter, dim, 2000, dev)?,
        w3: det_at(inter, dim, 2030, dev)?,
        w2: det_at(dim, inter, 2060, dev)?,
        swiglu_limit: 0.0,
    };
    Ok(Moe { gate: golden_gate(dev)?, experts: routed, shared })
}

/// The full combine: each token is routed to its top-2 experts (weights from the gate), each expert
/// processes only its tokens, results scatter-add back, then the always-on shared expert adds over
/// all tokens. Routing here is the verified `[0,1,1,3]`, so expert 1 handles BOTH tokens, expert 2
/// handles none (the empty-expert path), and the shared expert handles both. Golden from
/// `moe_golden.py`.
#[test]
fn moe_routes_and_combines() -> candle_core::Result<()> {
    let dev = Device::Cpu;
    let moe = golden_moe(&dev)?;
    let x = Tensor::from_vec(vec![0.8f32, -0.5, 0.3, -0.2, 0.9, -0.6], (2, 3), &dev)?;

    let y = moe.forward(&x)?;
    assert_eq!(y.dims(), &[2, 3]);

    let got = y.flatten_all()?.to_vec1::<f32>()?;
    let golden = [0.002_387_f32, 0.001_787, -0.004_191, 0.003_381, -0.000_879, -0.002_493];
    for (k, (a, b)) in got.iter().zip(golden.iter()).enumerate() {
        assert!((a - b).abs() < 1e-5, "y[{k}] = {a}, expected {b}");
    }
    Ok(())
}
