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

use crate::attention::{linear, rms_norm};
use crate::block::Block;
use crate::rope::Rope;
use candle_core::{DType, Result, Tensor};

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
    /// Shared YaRN rotary embeddings.
    pub rope: Rope,
    /// Number of residual streams (`hc_mult`).
    pub hc: usize,
}

impl Transformer {
    /// `input_ids`: `[b, s]` (integer ids) → logits `[b, vocab]` (last position).
    pub fn forward(&self, input_ids: &Tensor, start_pos: usize) -> Result<Tensor> {
        let (b, s) = input_ids.dims2()?;
        let dim = self.embed.dim(1)?;

        // Embedding row lookup, then expand to `hc` identical residual streams.
        let ids = input_ids.flatten_all()?.to_dtype(DType::U32)?;
        let h = self.embed.index_select(&ids, 0)?.reshape((b, s, dim))?; // [b, s, dim]
        let mut h = h.unsqueeze(2)?.broadcast_as((b, s, self.hc, dim))?.contiguous()?;

        for layer in &self.layers {
            h = layer.forward(&h, &self.rope, start_pos)?; // [b, s, hc, dim]
        }
        self.head.forward(&h) // [b, vocab]
    }
}
