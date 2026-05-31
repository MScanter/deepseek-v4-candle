//! Full-model tests: parallel LM head + the assembled Transformer.
//!
//! The head's simplified-`hc_pre` collapse is novel enough to pin to a pure-Python golden
//! (`head_golden.py`, math-only). The Transformer itself is integration wiring over already-golden
//! units, so — like the Block — it is checked on the contracts the wiring must preserve: the head
//! emits `[b, vocab]` logits for the last position, and those logits respond to an *earlier* token
//! (causal attention reads the past), proving context actually flows embed → blocks → head.
//!
//! Deterministic `det_at` weights throughout (shared via `common`); never candle's flaky CPU `randn`.

mod common;
use common::*;

use candle_core::{Device, Tensor};
use deepseek_v4_candle::model::Head;

/// `Head::forward` matches the pure-Python `ParallelHead` golden (b=1, s=3, hc=2, dim=4, vocab=5).
///
/// hc_head collapses the streams with the `pre` gate (`sigmoid(mixes*scale+base)+eps`), then
/// `get_logits(norm(x))` takes ONLY the last position → `[b, vocab]`. Golden from `head_golden.py`.
#[test]
fn head_matches_reference() -> candle_core::Result<()> {
    let dev = Device::Cpu;
    // x[b,s,hc,d] = det_at(b*s*hc, dim, 800) reshaped.
    let x = det_at(6, 4, 800, &dev)?.reshape((1, 3, 2, 4))?;
    let head = Head {
        weight: det_at(5, 4, 200, &dev)?,                 // lm_head [vocab, dim]
        norm: det_at(1, 4, 120, &dev)?.reshape((4,))?,    // RMSNorm gamma [dim]
        hc_fn: det_at(2, 8, 50, &dev)?,                   // [hc, hc*dim]
        hc_base: det_at(1, 2, 90, &dev)?.reshape((2,))?,  // [hc]
        hc_scale: Tensor::new(&[1.5f32], &dev)?,          // [1]
        hc: 2,
        eps: 1e-6,
        hc_eps: 1e-6,
    };

    let logits = head.forward(&x)?;
    assert_eq!(logits.dims(), &[1, 5]);

    let got = logits.flatten_all()?.to_vec1::<f32>()?;
    let want = [0.149066, -0.091325, 0.023031, 0.047924, -0.113341];
    for (g, w) in got.iter().zip(want.iter()) {
        assert!((g - w).abs() < 1e-5, "head logit {g} vs golden {w}");
    }
    Ok(())
}

/// The assembled Transformer emits `[b, vocab]` logits (one row per sequence, last position).
#[test]
fn transformer_output_shape() -> candle_core::Result<()> {
    let dev = Device::Cpu;
    let model = toy_transformer(&dev)?;
    let input_ids = Tensor::from_vec(vec![1u32, 3, 0, 2], (1, 4), &dev)?;
    let logits = model.forward(&input_ids, 0)?;
    assert_eq!(logits.dims(), &[1, VOCAB]);
    Ok(())
}

/// Context flows through the stack: perturbing an *earlier* token changes the last-position logits.
/// Attention is causal, so position s-1 reads positions 0..s-1 — a different token at position 2
/// must move the final logits. (A stub that ignores context leaves them identical → red.)
#[test]
fn transformer_uses_context() -> candle_core::Result<()> {
    let dev = Device::Cpu;
    let model = toy_transformer(&dev)?;
    let ids1 = Tensor::from_vec(vec![1u32, 3, 0, 2], (1, 4), &dev)?;
    let ids2 = Tensor::from_vec(vec![1u32, 3, 5, 2], (1, 4), &dev)?; // differ at position 2 (not last)

    let l1 = model.forward(&ids1, 0)?.flatten_all()?.to_vec1::<f32>()?;
    let l2 = model.forward(&ids2, 0)?.flatten_all()?.to_vec1::<f32>()?;

    let changed = l1.iter().zip(l2.iter()).any(|(a, b)| (a - b).abs() > 1e-6);
    assert!(changed, "transformer ignored an earlier-token change: {l1:?} vs {l2:?}");
    Ok(())
}

/// Full end-to-end parity: the assembled Transformer matches a pure-Python golden chaining
/// embed → block → head, derived independently from the reference (`end_to_end_golden.py`). Every
/// unit is already golden-pinned; this pins the one thing unit goldens can't — the *composition
/// order* across the whole stack. Tolerance is looser than the per-unit goldens because the Rust
/// path is f32 while the golden is f64, and the error accumulates through the (deep) network; any
/// real wiring bug would still be off by ~0.1+, far above the bound.
#[test]
fn transformer_matches_reference() -> candle_core::Result<()> {
    let dev = Device::Cpu;
    let model = toy_transformer(&dev)?;
    let input_ids = Tensor::from_vec(vec![1u32, 3, 0, 2], (1, 4), &dev)?;
    let logits = model.forward(&input_ids, 0)?;
    assert_eq!(logits.dims(), &[1, VOCAB]);

    let got = logits.flatten_all()?.to_vec1::<f32>()?;
    // Measured max |Δ| ≈ 1.0e-5 (f32 vs the f64 golden); 1e-4 leaves headroom without masking bugs.
    let want = [0.48474, -0.635526, -1.470525, -1.645452, -1.081788, -0.032544];
    for (g, w) in got.iter().zip(want.iter()) {
        assert!((g - w).abs() < 1e-4, "transformer logit {g} vs golden {w}");
    }
    Ok(())
}
