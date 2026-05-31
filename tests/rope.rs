//! YaRN RoPE tests.
//!
//! The strong test (`yarn_freqs_match_reference`) pins the *actual* YaRN-interpolated
//! frequencies against golden values produced by the reference formula in
//! `inference/model.py::precompute_freqs_cis` (reproduced with pure `math` in
//! `reference/.../rope_golden.py`). The others check rotation invariants that any
//! correct RoPE must satisfy: position 0 is the identity, the transform preserves
//! norm, and `inverse` de-rotates back to the input.
//!
//! Toy config: dim=8, base=10000, original_seq_len=64, factor=16, beta_fast=32, beta_slow=1.

use candle_core::{Device, Tensor};
use deepseek_v4_candle::rope::Rope;

const DIM: usize = 8;
const BASE: f64 = 10000.0;
const ORIG: usize = 64;
const FACTOR: f64 = 16.0;
const BETA_FAST: f64 = 32.0;
const BETA_SLOW: f64 = 1.0;
const MAX_SEQ: usize = 8;

fn toy_rope(dev: &Device) -> candle_core::Result<Rope> {
    Rope::new(DIM, MAX_SEQ, ORIG, BASE, FACTOR, BETA_FAST, BETA_SLOW, dev)
}

#[test]
fn yarn_freqs_match_reference() -> candle_core::Result<()> {
    let dev = Device::Cpu;
    let rope = toy_rope(&dev)?;

    // cos/sin are [max_seq, dim/2].
    let cos = rope.cos().to_vec2::<f32>()?;
    let sin = rope.sin().to_vec2::<f32>()?;
    assert_eq!(cos.len(), MAX_SEQ);
    assert_eq!(cos[0].len(), DIM / 2);

    // Position 0: cos=1, sin=0 (no rotation).
    for &c in &cos[0] {
        assert!((c - 1.0).abs() < 1e-5, "cos[0] = {c}, expected 1");
    }
    for &s in &sin[0] {
        assert!(s.abs() < 1e-6, "sin[0] = {s}, expected 0");
    }

    // Position 3: golden values from the reference YaRN formula.
    let gcos = [-0.989_992_5_f32, 0.987_326_7, 0.999_998_2, 0.999_999_98];
    let gsin = [0.14112_f32, 0.158_701_16, 0.001_874_999, 0.000_187_5];
    for i in 0..DIM / 2 {
        assert!(
            (cos[3][i] - gcos[i]).abs() < 1e-5,
            "cos[3][{i}] = {}, expected {}",
            cos[3][i],
            gcos[i]
        );
        assert!(
            (sin[3][i] - gsin[i]).abs() < 1e-5,
            "sin[3][{i}] = {}, expected {}",
            sin[3][i],
            gsin[i]
        );
    }
    Ok(())
}

#[test]
fn pos0_is_identity() -> candle_core::Result<()> {
    let dev = Device::Cpu;
    let rope = toy_rope(&dev)?;
    // [b=1, s=2, h=1, dim]
    let x = Tensor::randn(0f32, 1f32, (1, 2, 1, DIM), &dev)?;
    let y = rope.apply(&x, 0, false)?;

    let x0 = x.narrow(1, 0, 1)?.flatten_all()?.to_vec1::<f32>()?;
    let y0 = y.narrow(1, 0, 1)?.flatten_all()?.to_vec1::<f32>()?;
    for (a, b) in x0.iter().zip(y0.iter()) {
        assert!((a - b).abs() < 1e-6, "pos-0 changed: {a} -> {b}");
    }
    Ok(())
}

#[test]
fn norm_preserved() -> candle_core::Result<()> {
    let dev = Device::Cpu;
    let rope = toy_rope(&dev)?;
    let x = Tensor::randn(0f32, 1f32, (1, 5, 2, DIM), &dev)?;
    let y = rope.apply(&x, 0, false)?;

    let nx = x.sqr()?.sum_all()?.to_scalar::<f32>()?;
    let ny = y.sqr()?.sum_all()?.to_scalar::<f32>()?;
    assert!((nx - ny).abs() < 1e-3, "norm not preserved: {nx} vs {ny}");
    Ok(())
}

#[test]
fn apply_at_uses_per_row_positions() -> candle_core::Result<()> {
    // `apply_at` ropes each row at its own (possibly strided) position — the KV compressor
    // ropes block `i` at position `i*ratio`. Validate against the golden contiguous `apply`.
    let dev = Device::Cpu;
    let rope = toy_rope(&dev)?;
    let x = Tensor::randn(0f32, 1f32, (1, 2, 1, DIM), &dev)?;

    // Strided positions [0, 2]: row 0 rotated at pos 0, row 1 at pos 2.
    let got = rope.apply_at(&x, &[0, 2], false)?;
    let want0 = rope.apply(&x.narrow(1, 0, 1)?, 0, false)?;
    let want2 = rope.apply(&x.narrow(1, 1, 1)?, 2, false)?;
    let want = Tensor::cat(&[&want0, &want2], 1)?;

    let g = got.flatten_all()?.to_vec1::<f32>()?;
    let w = want.flatten_all()?.to_vec1::<f32>()?;
    for (i, (p, q)) in g.iter().zip(w.iter()).enumerate() {
        assert!((p - q).abs() < 1e-6, "apply_at mismatch at {i}: {p} vs {q}");
    }
    Ok(())
}

#[test]
fn inverse_round_trips() -> candle_core::Result<()> {
    let dev = Device::Cpu;
    let rope = toy_rope(&dev)?;
    let x = Tensor::randn(0f32, 1f32, (1, 4, 2, DIM), &dev)?;
    let y = rope.apply(&x, 0, false)?;
    let z = rope.apply(&y, 0, true)?; // inverse / de-rotate

    let xv = x.flatten_all()?.to_vec1::<f32>()?;
    let zv = z.flatten_all()?.to_vec1::<f32>()?;
    for (a, b) in xv.iter().zip(zv.iter()) {
        assert!((a - b).abs() < 1e-5, "inverse did not round-trip: {a} vs {b}");
    }
    Ok(())
}
