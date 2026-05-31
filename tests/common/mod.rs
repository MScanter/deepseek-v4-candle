//! Shared deterministic toy builders for the integration tests.
//!
//! Every weight is `det_at` (sin-of-arange) rather than `randn`: candle's CPU RNG is a shared
//! global that parallel test threads consume in nondeterministic order, which makes any
//! tolerance assertion on its output flaky. Each integration-test binary includes this via
//! `mod common;`, so some helpers are unused per binary — hence the module-level allow.
#![allow(dead_code)]

use candle_core::{DType, Device, Tensor};
use deepseek_v4_candle::attention::Mla;
use deepseek_v4_candle::block::Block;
use deepseek_v4_candle::mhc::Hc;
use deepseek_v4_candle::model::{Head, Transformer};
use deepseek_v4_candle::moe::{Expert, Gate, Moe, ScoreFunc};
use deepseek_v4_candle::rope::Rope;

pub const DIM: usize = 8;
pub const HC: usize = 2; // hc_mult (residual streams)
pub const H: usize = 2; // n_heads
pub const HD: usize = 4; // head_dim (H*HD = DIM)
pub const RD: usize = 2; // rope_head_dim
pub const QLR: usize = 4; // q_lora_rank
pub const G: usize = 2; // o_groups
pub const OLR: usize = 2; // o_lora_rank
pub const NROUTED: usize = 4;
pub const TOPK: usize = 2;
pub const INTER: usize = 4; // moe_inter_dim
pub const MIXHC: usize = (2 + HC) * HC; // 8
pub const ITERS: usize = 20;
pub const VOCAB: usize = 6; // toy vocabulary

/// Deterministic `[o, i]` tensor (`0.3 * sin(0.7*(start+k) + 1)`), matching the Python goldens.
pub fn det_at(o: usize, i: usize, start: i64, dev: &Device) -> candle_core::Result<Tensor> {
    let n = (o * i) as i64;
    Tensor::arange(start, start + n, dev)?
        .to_dtype(DType::F32)?
        .affine(0.7, 1.0)?
        .sin()?
        .affine(0.3, 0.0)?
        .reshape((o, i))
}

/// Pure sliding-window MLA (no compressor/indexer) at the block's `DIM`.
pub fn toy_mla(dev: &Device) -> candle_core::Result<Mla> {
    Ok(Mla {
        wq_a: det_at(QLR, DIM, 1, dev)?,
        q_norm: Tensor::ones((QLR,), DType::F32, dev)?,
        wq_b: det_at(H * HD, QLR, 11, dev)?,
        wkv: det_at(HD, DIM, 23, dev)?,
        kv_norm: Tensor::ones((HD,), DType::F32, dev)?,
        wo_a: det_at(G * OLR, (H * HD) / G, 31, dev)?,
        wo_b: det_at(DIM, G * OLR, 43, dev)?,
        attn_sink: det_at(1, H, 7, dev)?.reshape((H,))?,
        n_heads: H,
        head_dim: HD,
        rope_head_dim: RD,
        n_groups: G,
        o_lora_rank: OLR,
        window_size: 128,
        compress_ratio: 0,
        compressor: None,
        indexer: None,
        eps: 1e-6,
        scale: (HD as f64).powf(-0.5),
    })
}

/// MoE at the block's `DIM`: 4 routed experts (top-2, sqrtsoftplus + bias) plus one shared expert.
pub fn toy_moe(dev: &Device) -> candle_core::Result<Moe> {
    let experts = (0..NROUTED)
        .map(|i| {
            let o = 100 + (i as i64) * 50;
            Ok(Expert {
                w1: det_at(INTER, DIM, o, dev)?,
                w3: det_at(INTER, DIM, o + 20, dev)?,
                w2: det_at(DIM, INTER, o + 40, dev)?,
                swiglu_limit: 0.0,
            })
        })
        .collect::<candle_core::Result<Vec<_>>>()?;
    Ok(Moe {
        gate: Gate {
            weight: det_at(NROUTED, DIM, 300, dev)?,
            bias: Some(det_at(1, NROUTED, 311, dev)?.reshape((NROUTED,))?),
            topk: TOPK,
            route_scale: 1.5,
            score_func: ScoreFunc::SqrtSoftplus,
        },
        experts,
        shared: Expert {
            w1: det_at(INTER, DIM, 400, dev)?,
            w3: det_at(INTER, DIM, 420, dev)?,
            w2: det_at(DIM, INTER, 440, dev)?,
            swiglu_limit: 0.0,
        },
    })
}

/// One Hyper-Connection mixer at the block's `DIM` / `HC`.
pub fn toy_hc(start: i64, dev: &Device) -> candle_core::Result<Hc> {
    Ok(Hc {
        hc_fn: det_at(MIXHC, HC * DIM, start, dev)?,
        hc_base: det_at(1, MIXHC, start + 97, dev)?.reshape((MIXHC,))?,
        hc_scale: Tensor::new(&[1.0f32, 1.0, 1.0], dev)?,
        hc: HC,
        sinkhorn_iters: ITERS,
        eps: 1e-6,
        norm_eps: 1e-6,
    })
}

/// A full toy decoder block (mHC-wrapped windowed-MLA + MoE) at `DIM` / `HC`.
pub fn toy_block(dev: &Device) -> candle_core::Result<Block> {
    Ok(Block {
        attn: toy_mla(dev)?,
        ffn: toy_moe(dev)?,
        attn_norm: Tensor::ones((DIM,), DType::F32, dev)?,
        ffn_norm: Tensor::ones((DIM,), DType::F32, dev)?,
        hc_attn: toy_hc(600, dev)?,
        hc_ffn: toy_hc(700, dev)?,
        eps: 1e-6,
    })
}

/// The parallel LM head at the block's `DIM` / `HC` (simplified-`hc_pre` collapse + lm_head).
pub fn toy_head(dev: &Device) -> candle_core::Result<Head> {
    Ok(Head {
        weight: det_at(VOCAB, DIM, 500, dev)?,
        norm: Tensor::ones((DIM,), DType::F32, dev)?,
        hc_fn: det_at(HC, HC * DIM, 520, dev)?,
        hc_base: det_at(1, HC, 560, dev)?.reshape((HC,))?,
        hc_scale: Tensor::new(&[1.0f32], dev)?,
        hc: HC,
        eps: 1e-6,
        hc_eps: 1e-6,
    })
}

/// A one-layer toy [`Transformer`]: embedding → toy block → toy head.
pub fn toy_transformer(dev: &Device) -> candle_core::Result<Transformer> {
    Ok(Transformer {
        embed: det_at(VOCAB, DIM, 800, dev)?,
        layers: vec![toy_block(dev)?],
        head: toy_head(dev)?,
        rope: Rope::new(RD, 16, 0, 10000.0, 1.0, 32.0, 1.0, dev)?,
        hc: HC,
    })
}
