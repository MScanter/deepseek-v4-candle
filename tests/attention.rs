//! Attention tests.
//!
//! `sink_attention_matches_reference` pins the core attention math — including the
//! learnable `attn_sink` term that sits in the softmax *denominator* with no matching
//! value — against golden values from the reference `sparse_attn` math (computed with
//! pure Python in `sdpa_golden.py`). K and V are the same single latent `kv` (MQA).
//! `causal_query0_ignores_later_keys` checks the causal mask directly.
//!
//! The `Mla` tests are intrinsic (no golden weights): the projections + RoPE + grouped
//! output are mechanical wiring, so we check the contract that must hold regardless of
//! weights — output shape, and that the whole stack is causal (perturbing the last token
//! cannot change earlier positions' outputs). The risky math (sink attention, YaRN) is
//! golden-tested in its own units.

use candle_core::{DType, Device, Tensor};
use deepseek_v4_candle::attention::{sdpa_with_sink, sparse_attn, Mla};
use deepseek_v4_candle::rope::Rope;
use deepseek_v4_candle::sparse::{Compressor, Indexer};

/// Toy inputs shared by the sdpa tests. q:[1,2,2,2] (b,s,h,d), kv:[1,2,2] (b,n,d), sink:[2].
fn toy(dev: &Device) -> candle_core::Result<(Tensor, Tensor, Tensor, f64)> {
    let q = Tensor::from_vec(vec![1f32, 0., 0., 1., 1., 1., 2., 0.], (1, 2, 2, 2), dev)?;
    let kv = Tensor::from_vec(vec![1f32, 0., 0., 2.], (1, 2, 2), dev)?;
    let sink = Tensor::from_vec(vec![0.5f32, -0.3], (2,), dev)?;
    let scale = 2f64.powf(-0.5);
    Ok((q, kv, sink, scale))
}

#[test]
fn sink_attention_matches_reference() -> candle_core::Result<()> {
    let dev = Device::Cpu;
    let (q, kv, sink, scale) = toy(&dev)?;

    let o = sdpa_with_sink(&q, &kv, &sink, scale, true)?;
    assert_eq!(o.dims(), &[1, 2, 2, 2]);

    let got = o.flatten_all()?.to_vec1::<f32>()?;
    let golden = [
        0.551_592_4_f32, 0.0, // o[0][0]
        0.574_442_5, 0.0, // o[0][1]
        0.260_345_6, 1.056_021_7, // o[1][0]
        0.702_631, 0.341_642_73, // o[1][1]
    ];
    for (k, (a, b)) in got.iter().zip(golden.iter()).enumerate() {
        assert!((a - b).abs() < 1e-5, "o[{k}] = {a}, expected {b}");
    }
    Ok(())
}

#[test]
fn causal_query0_ignores_later_keys() -> candle_core::Result<()> {
    let dev = Device::Cpu;
    let (q, kv1, sink, scale) = toy(&dev)?;
    let kv2 = Tensor::from_vec(vec![1f32, 0., 9., -5.], (1, 2, 2), &dev)?;

    let o1 = sdpa_with_sink(&q, &kv1, &sink, scale, true)?;
    let o2 = sdpa_with_sink(&q, &kv2, &sink, scale, true)?;

    let a = o1.narrow(1, 0, 1)?.flatten_all()?.to_vec1::<f32>()?;
    let b = o2.narrow(1, 0, 1)?.flatten_all()?.to_vec1::<f32>()?;
    for (x, y) in a.iter().zip(b.iter()) {
        assert!((x - y).abs() < 1e-6, "causal leak at query 0: {x} vs {y}");
    }
    Ok(())
}

// ---- sparse_attn (index-gathered sink attention) ----

/// `sparse_attn` with `topk_idxs` encoding the *causal* key set must reproduce the
/// dense causal golden exactly: causality lives in the indices, not a mask flag.
/// q0 attends key {0}; q1 attends keys {0,1} (the `-1` slot is a no-op).
#[test]
fn sparse_attn_causal_pattern_matches_dense() -> candle_core::Result<()> {
    let dev = Device::Cpu;
    let (q, kv, sink, scale) = toy(&dev)?;
    let idxs = Tensor::from_vec(vec![0i64, -1, 0, 1], (1, 2, 2), &dev)?;

    let o = sparse_attn(&q, &kv, &sink, &idxs, scale)?;
    assert_eq!(o.dims(), &[1, 2, 2, 2]);

    let got = o.flatten_all()?.to_vec1::<f32>()?;
    let golden = [
        0.551_592_4_f32, 0.0, // o[0][0]
        0.574_442_5, 0.0, // o[0][1]
        0.260_345_6, 1.056_021_7, // o[1][0]
        0.702_631, 0.341_642_73, // o[1][1]
    ];
    for (k, (a, b)) in got.iter().zip(golden.iter()).enumerate() {
        assert!((a - b).abs() < 1e-5, "o[{k}] = {a}, expected {b}");
    }
    Ok(())
}

/// `-1` index slots are dropped entirely (gather 0, score `-inf`), so padding the
/// index rows with extra `-1`s cannot change the output.
#[test]
fn sparse_attn_ignores_padding() -> candle_core::Result<()> {
    let dev = Device::Cpu;
    let (q, kv, sink, scale) = toy(&dev)?;
    let tight = Tensor::from_vec(vec![0i64, -1, 0, 1], (1, 2, 2), &dev)?;
    let padded = Tensor::from_vec(vec![0i64, -1, -1, 0, 1, -1], (1, 2, 3), &dev)?;

    let o1 = sparse_attn(&q, &kv, &sink, &tight, scale)?;
    let o2 = sparse_attn(&q, &kv, &sink, &padded, scale)?;

    let a = o1.flatten_all()?.to_vec1::<f32>()?;
    let b = o2.flatten_all()?.to_vec1::<f32>()?;
    for (x, y) in a.iter().zip(b.iter()) {
        assert!((x - y).abs() < 1e-6, "padding changed output: {x} vs {y}");
    }
    Ok(())
}

// ---- MLA struct (toy dims) ----
const DIM: usize = 16;
const QLR: usize = 8; // q_lora_rank
const H: usize = 2; // n_heads
const HD: usize = 8; // head_dim (H*HD = 16)
const RD: usize = 4; // rope_head_dim
const G: usize = 2; // o_groups
const OLR: usize = 4; // o_lora_rank

fn toy_mla(dev: &Device) -> candle_core::Result<Mla> {
    let w = |o: usize, i: usize| Tensor::randn(0f32, 0.3f32, (o, i), dev);
    Ok(Mla {
        wq_a: w(QLR, DIM)?,
        q_norm: Tensor::ones((QLR,), DType::F32, dev)?,
        wq_b: w(H * HD, QLR)?,
        wkv: w(HD, DIM)?,
        kv_norm: Tensor::ones((HD,), DType::F32, dev)?,
        wo_a: w(G * OLR, (H * HD) / G)?,
        wo_b: w(DIM, G * OLR)?,
        attn_sink: Tensor::randn(0f32, 0.3f32, (H,), dev)?,
        n_heads: H,
        head_dim: HD,
        rope_head_dim: RD,
        n_groups: G,
        o_lora_rank: OLR,
        window_size: 128, // >= test seqlens, so attention is full-causal unless overridden
        compress_ratio: 0, // pure sliding-window; HCA/CSA tests override this + add a compressor
        compressor: None,
        indexer: None,
        eps: 1e-6,
        scale: (HD as f64).powf(-0.5),
    })
}

/// Deterministic, varied tensor `[o, i]` from `sin` of a linear index starting at `start`
/// (no RNG — `Device::set_seed` is unsupported on the CPU backend in candle 0.9.2, and the
/// HCA "output changed" assertion needs reproducible weights). The values span roughly
/// `[-0.3, 0.3]` and vary per element, so attention weights stay non-degenerate.
fn det_at(o: usize, i: usize, start: i64, dev: &Device) -> candle_core::Result<Tensor> {
    let n = (o * i) as i64;
    Tensor::arange(start, start + n, dev)?
        .to_dtype(DType::F32)?
        .affine(0.7, 1.0)? // spread successive indices ~0.7 rad apart
        .sin()?
        .affine(0.3, 0.0)?
        .reshape((o, i))
}
fn det(o: usize, i: usize, dev: &Device) -> candle_core::Result<Tensor> {
    det_at(o, i, 0, dev)
}

/// Toy KV compressor (deterministic weights) matching the MLA's `head_dim`/`rope_head_dim`, so
/// its compressed blocks concatenate onto the latent `kv`. The CSA overlap path (`ratio == 4`)
/// projects to `2*head_dim` (`coff = 2`); otherwise `head_dim`. So `wkv`/`wgate` are
/// `[coff*head_dim, dim]`, the within-block position embedding `ape` is `[ratio, coff*head_dim]`,
/// gamma `norm` is `[head_dim]`.
fn toy_compressor(ratio: usize, dev: &Device) -> candle_core::Result<Compressor> {
    let pd = if ratio == 4 { 2 * HD } else { HD }; // overlap projection width
    Ok(Compressor {
        wkv: det_at(pd, DIM, 100, dev)?,
        wgate: det_at(pd, DIM, 200, dev)?,
        ape: det_at(ratio, pd, 300, dev)?,
        norm: Tensor::ones((HD,), DType::F32, dev)?,
        compress_ratio: ratio,
        head_dim: HD,
        rope_head_dim: RD,
        eps: 1e-6,
    })
}

#[test]
fn mla_output_shape() -> candle_core::Result<()> {
    let dev = Device::Cpu;
    let mla = toy_mla(&dev)?;
    let rope = Rope::new(RD, 8, 0, 10000.0, 1.0, 32.0, 1.0, &dev)?;
    let x = Tensor::randn(0f32, 1f32, (1, 3, DIM), &dev)?;
    let out = mla.forward(&x, &rope, 0)?;
    assert_eq!(out.dims(), &[1, 3, DIM]);
    Ok(())
}

#[test]
fn mla_windowed_locality() -> candle_core::Result<()> {
    // With window 2, query `i` attends only to keys {i-1, i}. Perturbing position 0's input
    // must leave positions 2 and 3 unchanged (0 is outside their window), even though 0 and 1
    // (whose windows include position 0) may change.
    let dev = Device::Cpu;
    let mut mla = toy_mla(&dev)?;
    mla.window_size = 2;
    let rope = Rope::new(RD, 8, 0, 10000.0, 1.0, 32.0, 1.0, &dev)?;

    let x = Tensor::randn(0f32, 1f32, (1, 4, DIM), &dev)?;
    let out1 = mla.forward(&x, &rope, 0)?;

    // x2 differs from x only at position 0.
    let new_first = Tensor::randn(0f32, 1f32, (1, 1, DIM), &dev)?;
    let tail = x.narrow(1, 1, 3)?;
    let x2 = Tensor::cat(&[&new_first, &tail], 1)?;
    let out2 = mla.forward(&x2, &rope, 0)?;

    // Positions 2 and 3 are outside position 0's window → unchanged.
    let a = out1.narrow(1, 2, 2)?.flatten_all()?.to_vec1::<f32>()?;
    let b = out2.narrow(1, 2, 2)?.flatten_all()?.to_vec1::<f32>()?;
    for (x, y) in a.iter().zip(b.iter()) {
        assert!((x - y).abs() < 1e-5, "window leak: out[2..4] changed {x} vs {y}");
    }
    Ok(())
}

#[test]
fn mla_is_causal() -> candle_core::Result<()> {
    let dev = Device::Cpu;
    let mla = toy_mla(&dev)?;
    let rope = Rope::new(RD, 8, 0, 10000.0, 1.0, 32.0, 1.0, &dev)?;

    let x = Tensor::randn(0f32, 1f32, (1, 4, DIM), &dev)?;
    let out1 = mla.forward(&x, &rope, 0)?;

    // x2 differs from x only at the last position.
    let head = x.narrow(1, 0, 3)?;
    let new_last = Tensor::randn(0f32, 1f32, (1, 1, DIM), &dev)?;
    let x2 = Tensor::cat(&[&head, &new_last], 1)?;
    let out2 = mla.forward(&x2, &rope, 0)?;

    // Positions 0..3 must be unchanged (they cannot attend to the perturbed last token).
    let a = out1.narrow(1, 0, 3)?.flatten_all()?.to_vec1::<f32>()?;
    let b = out2.narrow(1, 0, 3)?.flatten_all()?.to_vec1::<f32>()?;
    for (x, y) in a.iter().zip(b.iter()) {
        assert!((x - y).abs() < 1e-5, "non-causal: earlier output changed {x} vs {y}");
    }
    Ok(())
}

// ---- HCA variant (sliding-window + deterministic KV compression) ----

/// Build an HCA-style MLA (deterministic weights): a small sliding window plus a KV compressor
/// of ratio `ratio`, with `indexer = None` so block selection is the deterministic
/// `get_compress_topk_idxs`. Deterministic so the "output changed" assertion is reproducible.
fn toy_mla_hca(ratio: usize, window: usize, dev: &Device) -> candle_core::Result<Mla> {
    Ok(Mla {
        wq_a: det(QLR, DIM, dev)?,
        q_norm: Tensor::ones((QLR,), DType::F32, dev)?,
        wq_b: det_at(H * HD, QLR, 17, dev)?,
        wkv: det_at(HD, DIM, 29, dev)?,
        kv_norm: Tensor::ones((HD,), DType::F32, dev)?,
        wo_a: det_at(G * OLR, (H * HD) / G, 41, dev)?,
        wo_b: det_at(DIM, G * OLR, 53, dev)?,
        attn_sink: det_at(1, H, 7, dev)?.reshape((H,))?,
        n_heads: H,
        head_dim: HD,
        rope_head_dim: RD,
        n_groups: G,
        o_lora_rank: OLR,
        window_size: window,
        compress_ratio: ratio,
        compressor: Some(toy_compressor(ratio, dev)?),
        indexer: None,
        eps: 1e-6,
        scale: (HD as f64).powf(-0.5),
    })
}

/// The defining HCA behaviour: compressed blocks extend the receptive field *beyond* the
/// window. With window 2 and ratio 2, query 3's window is {2,3}, yet it also attends to
/// compressed block 0 (positions {0,1}). So — unlike the pure-window case
/// (`mla_windowed_locality`, where out[3] is independent of position 0) — perturbing
/// position 0 here *must* change out[3], routed through the compressed block.
#[test]
fn mla_hca_extends_receptive_field() -> candle_core::Result<()> {
    let dev = Device::Cpu;
    let mla = toy_mla_hca(2, 2, &dev)?;
    let rope = Rope::new(RD, 8, 0, 10000.0, 1.0, 32.0, 1.0, &dev)?;

    let x = det(4, DIM, &dev)?.reshape((1, 4, DIM))?;
    let out1 = mla.forward(&x, &rope, 0)?;

    // x2 differs from x only at position 0 (outside query 3's window {2,3}).
    let new_first = det_at(1, DIM, 9001, &dev)?.reshape((1, 1, DIM))?;
    let tail = x.narrow(1, 1, 3)?;
    let x2 = Tensor::cat(&[&new_first, &tail], 1)?;
    let out2 = mla.forward(&x2, &rope, 0)?;

    // out[3] must change: position 0 reaches query 3 via compressed block 0.
    let a = out1.narrow(1, 3, 1)?.flatten_all()?.to_vec1::<f32>()?;
    let b = out2.narrow(1, 3, 1)?.flatten_all()?.to_vec1::<f32>()?;
    let max_diff = a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
    assert!(max_diff > 1e-4, "compressed block did not extend receptive field: max_diff = {max_diff}");
    Ok(())
}

/// Compression must not break causality. With window 2 and ratio 2 over 4 tokens, the last
/// token (position 3) sits in block 1 ({2,3}), which becomes visible only to query 3. So
/// perturbing position 3 leaves outputs 0,1,2 unchanged.
#[test]
fn mla_hca_is_causal() -> candle_core::Result<()> {
    let dev = Device::Cpu;
    let mla = toy_mla_hca(2, 2, &dev)?;
    let rope = Rope::new(RD, 8, 0, 10000.0, 1.0, 32.0, 1.0, &dev)?;

    let x = det(4, DIM, &dev)?.reshape((1, 4, DIM))?;
    let out1 = mla.forward(&x, &rope, 0)?;

    let head = x.narrow(1, 0, 3)?;
    let new_last = det_at(1, DIM, 9001, &dev)?.reshape((1, 1, DIM))?;
    let x2 = Tensor::cat(&[&head, &new_last], 1)?;
    let out2 = mla.forward(&x2, &rope, 0)?;

    let a = out1.narrow(1, 0, 3)?.flatten_all()?.to_vec1::<f32>()?;
    let b = out2.narrow(1, 0, 3)?.flatten_all()?.to_vec1::<f32>()?;
    for (x, y) in a.iter().zip(b.iter()) {
        assert!((x - y).abs() < 1e-5, "HCA non-causal: earlier output changed {x} vs {y}");
    }
    Ok(())
}

/// HCA output keeps the MLA contract shape `[b, s, dim]` regardless of compression.
#[test]
fn mla_hca_output_shape() -> candle_core::Result<()> {
    let dev = Device::Cpu;
    let mla = toy_mla_hca(2, 2, &dev)?;
    let rope = Rope::new(RD, 8, 0, 10000.0, 1.0, 32.0, 1.0, &dev)?;
    let x = det(6, DIM, &dev)?.reshape((1, 6, DIM))?;
    let out = mla.forward(&x, &rope, 0)?;
    assert_eq!(out.dims(), &[1, 6, DIM]);
    Ok(())
}

// ---- CSA variant (sliding-window + learned KV-block selection) ----

/// Toy CSA Indexer (deterministic weights): scores compressed blocks via a small per-head
/// attention over its *own* overlapping compressor's KV, then keeps `index_topk` blocks.
/// `wq_b` is `[n_heads*head_dim, q_lora_rank]`, `weights_proj` is `[n_heads, dim]`; the indexer's
/// head/rope dims match the MLA's so its query scores the compressed KV.
fn toy_indexer(dev: &Device) -> candle_core::Result<Indexer> {
    Ok(Indexer {
        wq_b: det_at(H * HD, QLR, 61, dev)?,
        weights_proj: det_at(H, DIM, 67, dev)?,
        compressor: toy_compressor(4, dev)?,
        n_heads: H,
        head_dim: HD,
        rope_head_dim: RD,
        index_topk: 1, // keep a single block, so a strict subset of the visible blocks
        compress_ratio: 4,
        scale: (HD as f64).powf(-0.5),
    })
}

/// Build a CSA-style MLA: an HCA layer (ratio-4 overlapping compressor) whose deterministic
/// block selection is replaced by the learned [`Indexer`].
fn toy_mla_csa(window: usize, dev: &Device) -> candle_core::Result<Mla> {
    let mut mla = toy_mla_hca(4, window, dev)?;
    mla.indexer = Some(toy_indexer(dev)?);
    Ok(mla)
}

/// The defining CSA behaviour: the learned indexer *narrows* the compressed-key set. Over 12
/// tokens with ratio 4 there are 3 compressed blocks; at query 11 all three are visible, but the
/// indexer (`index_topk = 1`) keeps only the top-scoring one, dropping two blocks the
/// deterministic HCA path would attend to. So the CSA output at position 11 must differ from the
/// otherwise-identical layer with `indexer = None` (which attends to every visible block).
#[test]
fn mla_csa_narrows_to_selected_blocks() -> candle_core::Result<()> {
    let dev = Device::Cpu;
    let mla_csa = toy_mla_csa(4, &dev)?;
    let mut mla_all = toy_mla_csa(4, &dev)?;
    mla_all.indexer = None; // deterministic HCA: attend to ALL visible compressed blocks
    let rope = Rope::new(RD, 16, 0, 10000.0, 1.0, 32.0, 1.0, &dev)?;

    let x = det(12, DIM, &dev)?.reshape((1, 12, DIM))?;
    let o_csa = mla_csa.forward(&x, &rope, 0)?;
    let o_all = mla_all.forward(&x, &rope, 0)?;

    // Position 11 sees three compressed blocks; the learned top-1 drops two -> outputs differ.
    let a = o_csa.narrow(1, 11, 1)?.flatten_all()?.to_vec1::<f32>()?;
    let b = o_all.narrow(1, 11, 1)?.flatten_all()?.to_vec1::<f32>()?;
    let max_diff = a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
    assert!(max_diff > 1e-4, "learned selection did not narrow the key set: max_diff = {max_diff}");
    Ok(())
}

/// The learned selection must not break causality. Perturbing the last token (position 11, which
/// lives in compressed block 2 — visible only to query 11, and in the sliding window of query 11
/// alone) must leave outputs 0..=10 unchanged, both their attention values and their block picks.
#[test]
fn mla_csa_is_causal() -> candle_core::Result<()> {
    let dev = Device::Cpu;
    let mla = toy_mla_csa(4, &dev)?;
    let rope = Rope::new(RD, 16, 0, 10000.0, 1.0, 32.0, 1.0, &dev)?;

    let x = det(12, DIM, &dev)?.reshape((1, 12, DIM))?;
    let out1 = mla.forward(&x, &rope, 0)?;

    let head = x.narrow(1, 0, 11)?;
    let new_last = det_at(1, DIM, 9001, &dev)?.reshape((1, 1, DIM))?;
    let x2 = Tensor::cat(&[&head, &new_last], 1)?;
    let out2 = mla.forward(&x2, &rope, 0)?;

    let a = out1.narrow(1, 0, 11)?.flatten_all()?.to_vec1::<f32>()?;
    let b = out2.narrow(1, 0, 11)?.flatten_all()?.to_vec1::<f32>()?;
    for (x, y) in a.iter().zip(b.iter()) {
        assert!((x - y).abs() < 1e-5, "CSA non-causal: earlier output changed {x} vs {y}");
    }
    Ok(())
}
