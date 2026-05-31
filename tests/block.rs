//! Decoder Block tests (mHC-wrapped attention + MoE).
//!
//! The Block is integration wiring: every risky unit it composes — sink attention + YaRN (golden
//! in `attention`/`rope`), the Sinkhorn mHC mix (golden in `mhc`), and MoE routing/experts (golden
//! in `moe`) — is already pinned to the reference. So, like the `Mla` tests, the Block is checked
//! on the contracts the *wiring* must preserve: the `[b,s,hc,dim]` stream shape round-trips, and
//! the whole stack stays causal (perturbing the last token cannot change earlier positions). mHC
//! mixes streams per-token and never across positions, so causality must survive the block.
//!
//! Toy builders (deterministic `det_at` weights) are shared via `common` — candle's CPU `randn` is
//! a shared global the parallel test threads consume nondeterministically, so we never use it.

mod common;
use common::*;

use candle_core::{Device, Tensor};
use deepseek_v4_candle::rope::Rope;

/// The block preserves the `[b, s, hc, dim]` residual-stream shape.
#[test]
fn block_output_shape() -> candle_core::Result<()> {
    let dev = Device::Cpu;
    let block = toy_block(&dev)?;
    let rope = Rope::new(RD, 16, 0, 10000.0, 1.0, 32.0, 1.0, &dev)?;
    let x = det_at(4 * HC, DIM, 800, &dev)?.reshape((1, 4, HC, DIM))?;
    let out = block.forward(&x, &rope, 0)?;
    assert_eq!(out.dims(), &[1, 4, HC, DIM]);
    Ok(())
}

/// The whole block stays causal: mHC mixes streams per-token (never across positions), attention is
/// windowed-causal, and the MoE is per-token — so perturbing the last token leaves earlier outputs
/// unchanged across every residual stream.
#[test]
fn block_is_causal() -> candle_core::Result<()> {
    let dev = Device::Cpu;
    let block = toy_block(&dev)?;
    let rope = Rope::new(RD, 16, 0, 10000.0, 1.0, 32.0, 1.0, &dev)?;

    let x = det_at(4 * HC, DIM, 800, &dev)?.reshape((1, 4, HC, DIM))?;
    let out1 = block.forward(&x, &rope, 0)?;

    // x2 differs from x only at the last position (all hc streams).
    let head = x.narrow(1, 0, 3)?;
    let new_last = det_at(HC, DIM, 9001, &dev)?.reshape((1, 1, HC, DIM))?;
    let x2 = Tensor::cat(&[&head, &new_last], 1)?;
    let out2 = block.forward(&x2, &rope, 0)?;

    let a = out1.narrow(1, 0, 3)?.flatten_all()?.to_vec1::<f32>()?;
    let b = out2.narrow(1, 0, 3)?.flatten_all()?.to_vec1::<f32>()?;
    for (x, y) in a.iter().zip(b.iter()) {
        assert!((x - y).abs() < 1e-5, "block non-causal: earlier output changed {x} vs {y}");
    }
    Ok(())
}
