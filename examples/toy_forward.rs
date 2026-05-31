//! Toy end-to-end forward pass through a one-layer DeepSeek-V4 model.
//!
//! Builds a tiny Transformer with deterministic weights (no checkpoint needed) and runs a prefill
//! forward, printing the next-token logits. Every weight is `0.3*sin(0.7*(start+k)+1)` — the same
//! deterministic fill the parity tests use — so the output is fully reproducible. The point is to
//! show that the public API is enough to assemble and run the whole architecture.
//!
//! Run with: `cargo run --example toy_forward`

use candle_core::{DType, Device, Result, Tensor};
use deepseek_v4_candle::attention::Mla;
use deepseek_v4_candle::block::Block;
use deepseek_v4_candle::mhc::Hc;
use deepseek_v4_candle::model::{Head, Transformer};
use deepseek_v4_candle::moe::{Expert, Gate, Moe, ScoreFunc};
use deepseek_v4_candle::rope::Rope;

// A shrunk V4: identical structure, tiny dims.
const DIM: usize = 8; // model dim
const HC: usize = 2; // hc_mult: parallel residual streams (mHC)
const H: usize = 2; // attention heads
const HD: usize = 4; // head dim (H*HD = DIM)
const RD: usize = 2; // rope head dim
const QLR: usize = 4; // query low-rank
const G: usize = 2; // output groups
const OLR: usize = 2; // output low-rank
const NROUTED: usize = 4; // routed experts
const TOPK: usize = 2; // experts per token
const INTER: usize = 4; // expert hidden dim
const MIXHC: usize = (2 + HC) * HC;
const VOCAB: usize = 6;

/// Deterministic `[o, i]` weight fill, matching the parity goldens.
fn det(dev: &Device, o: usize, i: usize, start: i64) -> Result<Tensor> {
    Tensor::arange(start, start + (o * i) as i64, dev)?
        .to_dtype(DType::F32)?
        .affine(0.7, 1.0)?
        .sin()?
        .affine(0.3, 0.0)?
        .reshape((o, i))
}

fn ones(dev: &Device, n: usize) -> Result<Tensor> {
    Tensor::ones((n,), DType::F32, dev)
}

/// One mHC mixer (Sinkhorn doubly-stochastic stream mixing), as used at each sublayer.
fn mhc(dev: &Device, start: i64) -> Result<Hc> {
    Ok(Hc {
        hc_fn: det(dev, MIXHC, HC * DIM, start)?,
        hc_base: det(dev, 1, MIXHC, start + 97)?.reshape((MIXHC,))?,
        hc_scale: Tensor::new(&[1.0f32, 1.0, 1.0], dev)?,
        hc: HC,
        sinkhorn_iters: 20,
        eps: 1e-6,
        norm_eps: 1e-6,
    })
}

fn main() -> Result<()> {
    let dev = Device::Cpu;

    // --- Multi-head Latent Attention (sliding-window variant: compress_ratio = 0) ---
    let attn = Mla {
        wq_a: det(&dev, QLR, DIM, 1)?,
        q_norm: ones(&dev, QLR)?,
        wq_b: det(&dev, H * HD, QLR, 11)?,
        wkv: det(&dev, HD, DIM, 23)?,
        kv_norm: ones(&dev, HD)?,
        wo_a: det(&dev, G * OLR, (H * HD) / G, 31)?,
        wo_b: det(&dev, DIM, G * OLR, 43)?,
        attn_sink: det(&dev, 1, H, 7)?.reshape((H,))?,
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
    };

    // --- Mixture-of-Experts (sqrtsoftplus gate + SwiGLU experts + shared expert) ---
    let experts = (0..NROUTED)
        .map(|i| {
            let o = 100 + i as i64 * 50;
            Ok(Expert {
                w1: det(&dev, INTER, DIM, o)?,
                w3: det(&dev, INTER, DIM, o + 20)?,
                w2: det(&dev, DIM, INTER, o + 40)?,
                swiglu_limit: 0.0,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let ffn = Moe {
        gate: Gate {
            weight: det(&dev, NROUTED, DIM, 300)?,
            bias: Some(det(&dev, 1, NROUTED, 311)?.reshape((NROUTED,))?),
            topk: TOPK,
            route_scale: 1.5,
            score_func: ScoreFunc::SqrtSoftplus,
        },
        experts,
        shared: Expert {
            w1: det(&dev, INTER, DIM, 400)?,
            w3: det(&dev, INTER, DIM, 420)?,
            w2: det(&dev, DIM, INTER, 440)?,
            swiglu_limit: 0.0,
        },
    };

    // --- One decoder block: each sublayer is mHC-wrapped (pre-collapse -> norm -> sublayer -> post) ---
    let block = Block {
        attn,
        ffn,
        attn_norm: ones(&dev, DIM)?,
        ffn_norm: ones(&dev, DIM)?,
        hc_attn: mhc(&dev, 600)?,
        hc_ffn: mhc(&dev, 700)?,
        eps: 1e-6,
    };

    // --- Parallel LM head (mHC-pre collapse -> RMSNorm -> last-position logits) ---
    let head = Head {
        weight: det(&dev, VOCAB, DIM, 500)?,
        norm: ones(&dev, DIM)?,
        hc_fn: det(&dev, HC, HC * DIM, 520)?,
        hc_base: det(&dev, 1, HC, 560)?.reshape((HC,))?,
        hc_scale: Tensor::new(&[1.0f32], &dev)?,
        hc: HC,
        eps: 1e-6,
        hc_eps: 1e-6,
    };

    let model = Transformer {
        embed: det(&dev, VOCAB, DIM, 800)?,
        layers: vec![block],
        head,
        rope: Rope::new(RD, 16, 0, 10000.0, 1.0, 32.0, 1.0, &dev)?,
        hc: HC,
    };

    // Prefill forward over a toy prompt.
    let input_ids = Tensor::from_vec(vec![1u32, 3, 0, 2], (1, 4), &dev)?;
    let logits = model.forward(&input_ids, 0)?; // [1, VOCAB]
    let row = logits.flatten_all()?.to_vec1::<f32>()?;
    let argmax = row
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .unwrap();

    println!("DeepSeek-V4 toy model: {DIM}-dim · {HC} mHC streams · {H} heads · {NROUTED}+1 experts · 1 layer");
    println!("input_ids                     : [1, 3, 0, 2]");
    println!("next-token logits ({VOCAB})          : {row:.4?}");
    println!("argmax (predicted next token) : {argmax}");
    Ok(())
}
