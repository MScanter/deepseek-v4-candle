//! One V4 decoder block: mHC-wrapped attention + Mixture-of-Experts.
//!
//! Ports `Block` (inference/model.py 647-700). Instead of a plain residual, the block carries
//! `hc_mult` parallel residual streams (see [`crate::mhc`]). Each sublayer is wrapped as
//! `hc_pre → RMSNorm → sublayer → hc_post`: `hc_pre` collapses the streams into one tensor for the
//! sublayer, `hc_post` re-expands the sublayer output back into the streams via the learned `post`
//! gate and the doubly-stochastic combination matrix.

use crate::attention::{rms_norm, Mla};
use crate::mhc::Hc;
use crate::moe::Moe;
use crate::rope::Rope;
use candle_core::{Result, Tensor};

/// A single decoder layer operating on the `[b, s, hc, dim]` residual streams.
pub struct Block {
    /// Multi-head Latent Attention sublayer.
    pub attn: Mla,
    /// Mixture-of-Experts feed-forward sublayer.
    pub ffn: Moe,
    /// RMSNorm gamma `[dim]`, applied to the collapsed stream before attention.
    pub attn_norm: Tensor,
    /// RMSNorm gamma `[dim]`, applied to the collapsed stream before the MoE.
    pub ffn_norm: Tensor,
    /// Hyper-Connection mixer wrapping the attention sublayer.
    pub hc_attn: Hc,
    /// Hyper-Connection mixer wrapping the MoE sublayer.
    pub hc_ffn: Hc,
    /// RMSNorm epsilon for `attn_norm` / `ffn_norm`.
    pub eps: f64,
}

impl Block {
    /// `x`: `[b, s, hc, dim]` → `[b, s, hc, dim]`.
    pub fn forward(&self, x: &Tensor, rope: &Rope, start_pos: usize) -> Result<Tensor> {
        // --- attention sublayer ---
        let (collapsed, post, comb) = self.hc_attn.pre(x)?; // [b,s,dim], gates for re-expansion
        let normed = rms_norm(&collapsed, Some(&self.attn_norm), self.eps)?;
        let attended = self.attn.forward(&normed, rope, start_pos)?; // [b,s,dim]
        let x = self.hc_attn.post(&attended, x, &post, &comb)?; // [b,s,hc,dim]

        // --- MoE feed-forward sublayer ---
        let (collapsed, post, comb) = self.hc_ffn.pre(&x)?;
        let normed = rms_norm(&collapsed, Some(&self.ffn_norm), self.eps)?;
        // MoE routes per token: flatten [b,s,dim] -> [b*s,dim], then restore the shape.
        let (b, s, d) = normed.dims3()?;
        let ff = self.ffn.forward(&normed.reshape((b * s, d))?)?.reshape((b, s, d))?;
        self.hc_ffn.post(&ff, &x, &post, &comb)
    }
}
