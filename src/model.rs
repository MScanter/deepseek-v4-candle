//! Full model: token embedding → mHC-expanded decoder stack → parallel LM head.
//!
//! Ports `ParallelEmbedding` (83-105), `ParallelHead` (703-735) and `Transformer`
//! (769-809) from inference/model.py.
//!
//! - **Embedding** is a plain row lookup (`weight[input_ids]`).
//! - **Head** collapses the `hc_mult` residual streams with a *simplified* `hc_pre`
//!   (the `pre` gate only — no Sinkhorn `post`/`comb`, since nothing is re-expanded
//!   at the head), RMS-normalizes, and projects the **last** position to vocab logits.
//!   The reference keeps the head's hc params + final `norm` on the `Transformer`; we
//!   consolidate them onto [`Head`] so one struct owns the whole logit computation.
//!
//! Single-rank only: the reference's tensor-parallel `world_size` sharding (vocab /
//! `part_vocab_size` splits + `all_gather`) collapses to the identity here. Hash-routing
//! MoE layers (which need `input_ids`) are deferred — every block here routes by score.

use crate::attention::{linear, rms_norm, Mla};
use crate::block::Block;
use crate::config::Config;
use crate::loader::SafeTensors;
use crate::mhc::Hc;
use crate::moe::{Expert, Gate, Moe, ScoreFunc};
use crate::rope::Rope;
use crate::sparse::{Compressor, Indexer};
use candle_core::{DType, Device, Error, Result, Tensor};

/// Numerically stable sigmoid: `1 / (1 + e^-x)` (affine(1,1) turns `e^-x` into `e^-x + 1`).
fn sigmoid(x: &Tensor) -> Result<Tensor> {
    x.neg()?.exp()?.affine(1.0, 1.0)?.recip()
}

/// The parallel LM head: collapse residual streams (simplified `hc_pre`), RMSNorm,
/// and project the last position to logits.
pub struct Head {
    /// `lm_head` weight `[vocab, dim]` (fp32).
    pub weight: Tensor,
    /// Final RMSNorm gamma `[dim]`.
    pub norm: Tensor,
    /// Stream-collapse projection `[hc, hc * dim]`.
    pub hc_fn: Tensor,
    /// Stream-collapse bias `[hc]`.
    pub hc_base: Tensor,
    /// Stream-collapse scale `[1]` (scalar).
    pub hc_scale: Tensor,
    /// Number of residual streams (`hc_mult`).
    pub hc: usize,
    /// RMSNorm epsilon.
    pub eps: f64,
    /// `hc_pre` epsilon (added to the `pre` gate).
    pub hc_eps: f64,
}

impl Head {
    /// `x`: `[b, s, hc, dim]` → logits `[b, vocab]` (last position only).
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let collapsed = self.collapse(x)?; // [b, s, dim]
        let normed = rms_norm(&collapsed, Some(&self.norm), self.eps)?;
        let (b, s, d) = normed.dims3()?;
        let last = normed.narrow(1, s - 1, 1)?.reshape((b, d))?; // last position only -> [b, dim]
        linear(&last, &self.weight) // [b, vocab]
    }

    /// Simplified `hc_head` (inference/model.py 728-735): the `pre` gate of `hc_pre` collapses the
    /// `hc` streams into one tensor. No Sinkhorn `post`/`comb` — nothing is re-expanded at the head.
    ///
    /// `x`: `[b, s, hc, dim]` → `[b, s, dim]`.
    fn collapse(&self, x: &Tensor) -> Result<Tensor> {
        let (b, s, hc, d) = x.dims4()?;
        let x = x.to_dtype(DType::F32)?;
        let xf = x.reshape((b * s, hc * d))?; // flatten the streams
        let rms = xf.sqr()?.mean_keepdim(1)?.affine(1.0, self.eps)?.powf(-0.5)?; // [b*s, 1]
        let mixes = linear(&xf, &self.hc_fn)?.broadcast_mul(&rms)?; // [b*s, hc]
        let s0 = self.hc_scale.to_vec1::<f32>()?[0] as f64;
        let pre = mixes.affine(s0, 0.0)?.broadcast_add(&self.hc_base)?;
        let pre = sigmoid(&pre)?.affine(1.0, self.hc_eps)?; // [b*s, hc]
        // y = sum_hc(pre[..., None] * x) -> [b, s, dim]
        pre.reshape((b, s, hc, 1))?.broadcast_mul(&x)?.sum(2)
    }
}

/// The full DeepSeek-V4 model: embed → expand to `hc` streams → blocks → head.
pub struct Transformer {
    /// Token embedding table `[vocab, dim]`.
    pub embed: Tensor,
    /// Stacked decoder layers.
    pub layers: Vec<Block>,
    /// Parallel LM head.
    pub head: Head,
    /// Per-layer YaRN rotary embeddings — one [`Rope`] per layer, since a compressed
    /// (HCA/CSA) layer uses `compress_rope_theta` + YaRN while a sliding-window layer uses
    /// the base `rope_theta` with YaRN disabled (`Attention.__init__`, model.py 475-482).
    /// `ropes[l]` is shared within layer `l` by the main q/kv, the compressor, and the indexer.
    pub ropes: Vec<Rope>,
    /// Number of residual streams (`hc_mult`).
    pub hc: usize,
}

impl Transformer {
    /// Assemble a [`Transformer`] from a [`Config`] and a loaded converted checkpoint, reading each
    /// weight by the name `inference/convert.py` emits (the inference model's attribute path) and the
    /// dtype the architecture stores it as (FP8/FP4 for the projections, bf16/f32 for the rest).
    pub fn from_config(cfg: &Config, st: &SafeTensors, dev: &Device) -> Result<Self> {
        // Scope guards — fail loudly on the paths not yet wired (consistent with the forward-path
        // deferrals). Hash-routed layers need the `tid2eid` table; HCA/CSA layers need the
        // compressor (+ indexer for CSA). Both are documented gaps, not silent wrong answers.
        if cfg.n_hash_layers > 0 {
            return Err(Error::Msg(format!(
                "from_config: hash-routing layers (n_hash_layers={}) not yet supported — Gate.route_hashed / tid2eid loading is deferred",
                cfg.n_hash_layers
            )));
        }

        let embed = st.auto_tensor("embed.weight", &[cfg.vocab_size, cfg.dim], dev)?;
        let layers = (0..cfg.n_layers)
            .map(|l| build_block(cfg, st, l, dev))
            .collect::<Result<Vec<_>>>()?;
        let head = build_head(cfg, st, dev)?;
        let ropes = (0..cfg.n_layers)
            .map(|l| build_rope(cfg, l, dev))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { embed, layers, head, ropes, hc: cfg.hc_mult })
    }

    /// `input_ids`: `[b, s]` (integer ids) → logits `[b, vocab]` (last position).
    pub fn forward(&self, input_ids: &Tensor, start_pos: usize) -> Result<Tensor> {
        let (b, s) = input_ids.dims2()?;
        let dim = self.embed.dim(1)?;

        // Embedding row lookup, then expand to `hc` identical residual streams.
        let ids = input_ids.flatten_all()?.to_dtype(DType::U32)?;
        let h = self.embed.index_select(&ids, 0)?.reshape((b, s, dim))?; // [b, s, dim]
        let mut h = h.unsqueeze(2)?.broadcast_as((b, s, self.hc, dim))?.contiguous()?;

        for (l, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h, &self.ropes[l], start_pos)?; // [b, s, hc, dim]
        }
        self.head.forward(&h) // [b, vocab]
    }
}

// --- `from_config` builders --------------------------------------------------------------------
//
// Each loads one sub-module from the checkpoint. Projection weights go through [`SafeTensors::linear`]
// (FP8/FP4 when a `.scale` sibling is present, else the unquantized weight by dtype); norms, biases,
// `attn_sink`, and the hc params — never quantized — go through [`SafeTensors::auto_tensor`].

/// The YaRN [`Rope`] for layer `layer`, picking the base/YaRN regime by compression ratio
/// (`Attention.__init__`, model.py 475-482): a sliding-window layer (`compress_ratio == 0`)
/// disables YaRN (`original_seq_len = 0`) and uses the base `rope_theta`; a compressed (HCA/CSA)
/// layer enables YaRN over `original_seq_len` and uses `compress_rope_theta`.
fn build_rope(cfg: &Config, layer: usize, dev: &Device) -> Result<Rope> {
    let (original_seq_len, theta) = if cfg.compress_ratios[layer] != 0 {
        (cfg.original_seq_len, cfg.compress_rope_theta)
    } else {
        (0, cfg.rope_theta)
    };
    Rope::new(
        cfg.rope_head_dim,
        cfg.max_seq_len,
        original_seq_len,
        theta,
        cfg.rope_factor,
        cfg.beta_fast,
        cfg.beta_slow,
        dev,
    )
}

/// Map the config's `score_func` string to the [`ScoreFunc`] enum.
fn score_func(name: &str) -> Result<ScoreFunc> {
    match name {
        "sqrtsoftplus" => Ok(ScoreFunc::SqrtSoftplus),
        "softmax" => Ok(ScoreFunc::Softmax),
        "sigmoid" => Ok(ScoreFunc::Sigmoid),
        other => Err(Error::Msg(format!("from_config: unknown score_func {other:?}"))),
    }
}

/// One [`Mla`] (layer `layer`). `wq_a/wq_b/wkv/wo_b` are FP8 in a real checkpoint; `wo_a` is the
/// pre-dequantized bf16 special-case (no scale) — `linear` handles both by scale presence. A
/// compressed layer (`compress_ratio > 0`) also carries a KV [`Compressor`]; a CSA layer
/// (`compress_ratio == 4`) additionally carries the learned [`Indexer`].
fn build_mla(cfg: &Config, st: &SafeTensors, layer: usize, dev: &Device) -> Result<Mla> {
    let p = format!("layers.{layer}.attn");
    let ratio = cfg.compress_ratios[layer];
    Ok(Mla {
        wq_a: st.linear(&format!("{p}.wq_a"), cfg.q_lora_rank, cfg.dim, false, dev)?,
        q_norm: st.auto_tensor(&format!("{p}.q_norm.weight"), &[cfg.q_lora_rank], dev)?,
        wq_b: st.linear(&format!("{p}.wq_b"), cfg.n_heads * cfg.head_dim, cfg.q_lora_rank, false, dev)?,
        wkv: st.linear(&format!("{p}.wkv"), cfg.head_dim, cfg.dim, false, dev)?,
        kv_norm: st.auto_tensor(&format!("{p}.kv_norm.weight"), &[cfg.head_dim], dev)?,
        wo_a: st.linear(
            &format!("{p}.wo_a"),
            cfg.o_groups * cfg.o_lora_rank,
            cfg.n_heads * cfg.head_dim / cfg.o_groups,
            false,
            dev,
        )?,
        wo_b: st.linear(&format!("{p}.wo_b"), cfg.dim, cfg.o_groups * cfg.o_lora_rank, false, dev)?,
        attn_sink: st.auto_tensor(&format!("{p}.attn_sink"), &[cfg.n_heads], dev)?,
        n_heads: cfg.n_heads,
        head_dim: cfg.head_dim,
        rope_head_dim: cfg.rope_head_dim,
        n_groups: cfg.o_groups,
        o_lora_rank: cfg.o_lora_rank,
        window_size: cfg.window_size,
        compress_ratio: ratio,
        compressor: if ratio > 0 {
            Some(build_compressor(&format!("{p}.compressor"), ratio, cfg.head_dim, cfg, st, dev)?)
        } else {
            None
        },
        indexer: if ratio == 4 { Some(build_indexer(&p, cfg, st, dev)?) } else { None },
        eps: cfg.norm_eps,
        scale: (cfg.head_dim as f64).powf(-0.5),
    })
}

/// One KV [`Compressor`] at `prefix` (`layers.N.attn.compressor` for the main path, or
/// `layers.N.attn.indexer.compressor` for the CSA indexer's own). `coff = 2` for the CSA overlap
/// (`ratio == 4`), else `1`; `wkv`/`wgate` are stored bf16 (no `.scale` → `linear` reads them
/// unquantized), `ape` is an fp32 `nn.Parameter` (no `.weight`), `norm` is the RMSNorm gamma.
/// `rope_head_dim` is the *global* `cfg.rope_head_dim` regardless of `head_dim` (model.py 287).
fn build_compressor(
    prefix: &str,
    ratio: usize,
    head_dim: usize,
    cfg: &Config,
    st: &SafeTensors,
    dev: &Device,
) -> Result<Compressor> {
    let coff = if ratio == 4 { 2 } else { 1 };
    Ok(Compressor {
        wkv: st.linear(&format!("{prefix}.wkv"), coff * head_dim, cfg.dim, false, dev)?,
        wgate: st.linear(&format!("{prefix}.wgate"), coff * head_dim, cfg.dim, false, dev)?,
        ape: st.auto_tensor(&format!("{prefix}.ape"), &[ratio, coff * head_dim], dev)?,
        norm: st.auto_tensor(&format!("{prefix}.norm.weight"), &[head_dim], dev)?,
        compress_ratio: ratio,
        head_dim,
        rope_head_dim: cfg.rope_head_dim,
        eps: cfg.norm_eps,
    })
}

/// The CSA learned [`Indexer`] for the layer attention at `p` (`layers.N.attn`). Its query proj
/// `wq_b` and per-head `weights_proj` are bf16; it owns an overlapping [`Compressor`] at
/// `index_head_dim` (always `ratio == 4`). The per-head weight is scaled by `n_heads^-0.5` inside
/// `select`, so `scale` here is just `index_head_dim^-0.5`.
fn build_indexer(p: &str, cfg: &Config, st: &SafeTensors, dev: &Device) -> Result<Indexer> {
    let (ih, ihd) = (cfg.index_n_heads, cfg.index_head_dim);
    Ok(Indexer {
        wq_b: st.linear(&format!("{p}.indexer.wq_b"), ih * ihd, cfg.q_lora_rank, false, dev)?,
        weights_proj: st.linear(&format!("{p}.indexer.weights_proj"), ih, cfg.dim, false, dev)?,
        compressor: build_compressor(&format!("{p}.indexer.compressor"), 4, ihd, cfg, st, dev)?,
        n_heads: ih,
        head_dim: ihd,
        rope_head_dim: cfg.rope_head_dim,
        index_topk: cfg.index_topk,
        compress_ratio: 4,
        scale: (ihd as f64).powf(-0.5),
    })
}

/// One SwiGLU [`Expert`]. `fp4` selects the FP4 dequant for routed experts (shared experts are FP8).
fn build_expert(prefix: &str, cfg: &Config, st: &SafeTensors, fp4: bool, dev: &Device) -> Result<Expert> {
    Ok(Expert {
        w1: st.linear(&format!("{prefix}.w1"), cfg.moe_inter_dim, cfg.dim, fp4, dev)?,
        w2: st.linear(&format!("{prefix}.w2"), cfg.dim, cfg.moe_inter_dim, fp4, dev)?,
        w3: st.linear(&format!("{prefix}.w3"), cfg.moe_inter_dim, cfg.dim, fp4, dev)?,
        swiglu_limit: cfg.swiglu_limit,
    })
}

/// The [`Moe`] for layer `layer`: routed experts (FP4), the FP8 shared expert, and the gate.
fn build_moe(cfg: &Config, st: &SafeTensors, layer: usize, dev: &Device) -> Result<Moe> {
    let p = format!("layers.{layer}.ffn");
    let experts = (0..cfg.n_routed_experts)
        .map(|j| build_expert(&format!("{p}.experts.{j}"), cfg, st, true, dev))
        .collect::<Result<Vec<_>>>()?;
    let shared = build_expert(&format!("{p}.shared_experts"), cfg, st, false, dev)?;
    let gate = Gate {
        weight: st.linear(&format!("{p}.gate"), cfg.n_routed_experts, cfg.dim, false, dev)?,
        bias: Some(st.auto_tensor(&format!("{p}.gate.bias"), &[cfg.n_routed_experts], dev)?),
        topk: cfg.n_activated_experts,
        route_scale: cfg.route_scale,
        score_func: score_func(&cfg.score_func)?,
    };
    Ok(Moe { gate, experts, shared })
}

/// One [`Hc`] mixer. `prefix` is the site stem (`layers.N.hc_attn` / `layers.N.hc_ffn`); the three
/// params are `{prefix}_fn` `[mix_hc, hc*dim]`, `{prefix}_base` `[mix_hc]`, `{prefix}_scale` `[3]`.
fn build_hc(prefix: &str, cfg: &Config, st: &SafeTensors, dev: &Device) -> Result<Hc> {
    Ok(Hc {
        hc_fn: st.auto_tensor(&format!("{prefix}_fn"), &[cfg.mix_hc(), cfg.hc_mult * cfg.dim], dev)?,
        hc_base: st.auto_tensor(&format!("{prefix}_base"), &[cfg.mix_hc()], dev)?,
        hc_scale: st.auto_tensor(&format!("{prefix}_scale"), &[3], dev)?,
        hc: cfg.hc_mult,
        sinkhorn_iters: cfg.hc_sinkhorn_iters,
        eps: cfg.hc_eps,
        norm_eps: cfg.norm_eps,
    })
}

/// One decoder [`Block`]: mHC-wrapped attention + MoE, with the two pre-sublayer RMSNorms.
fn build_block(cfg: &Config, st: &SafeTensors, layer: usize, dev: &Device) -> Result<Block> {
    Ok(Block {
        attn: build_mla(cfg, st, layer, dev)?,
        ffn: build_moe(cfg, st, layer, dev)?,
        attn_norm: st.auto_tensor(&format!("layers.{layer}.attn_norm.weight"), &[cfg.dim], dev)?,
        ffn_norm: st.auto_tensor(&format!("layers.{layer}.ffn_norm.weight"), &[cfg.dim], dev)?,
        hc_attn: build_hc(&format!("layers.{layer}.hc_attn"), cfg, st, dev)?,
        hc_ffn: build_hc(&format!("layers.{layer}.hc_ffn"), cfg, st, dev)?,
        eps: cfg.norm_eps,
    })
}

/// The LM [`Head`]: it consolidates the reference's top-level `head` (bf16, no scale), final `norm`,
/// and head mHC params (`hc_head_{fn,base,scale}`, `hc_head_scale` is `[1]`) into one struct.
fn build_head(cfg: &Config, st: &SafeTensors, dev: &Device) -> Result<Head> {
    Ok(Head {
        weight: st.linear("head", cfg.vocab_size, cfg.dim, false, dev)?,
        norm: st.auto_tensor("norm.weight", &[cfg.dim], dev)?,
        hc_fn: st.auto_tensor("hc_head_fn", &[cfg.hc_mult, cfg.hc_mult * cfg.dim], dev)?,
        hc_base: st.auto_tensor("hc_head_base", &[cfg.hc_mult], dev)?,
        hc_scale: st.auto_tensor("hc_head_scale", &[1], dev)?,
        hc: cfg.hc_mult,
        eps: cfg.norm_eps,
        hc_eps: cfg.hc_eps,
    })
}
