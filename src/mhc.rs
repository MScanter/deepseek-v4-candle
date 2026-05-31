//! Manifold-Constrained Hyper-Connections (mHC).
//!
//! V4 replaces the standard residual connection with `hc_mult` parallel residual
//! streams that are mixed by a *doubly-stochastic* (Birkhoff-polytope) matrix,
//! obtained per-token via the Sinkhorn–Knopp algorithm. Doubly-stochastic mixing
//! preserves signal magnitude, which is what keeps wide residual streams stable.
//!
//! This is a direct port of `hc_split_sinkhorn` (inference/kernel.py) and the
//! `hc_pre` / `hc_post` methods of `Block` (inference/model.py). The math is the
//! source of truth; the only deviation is that we run dense candle ops on CPU
//! instead of the fused tilelang GPU kernel.

use candle_core::{DType, Result, Tensor, D};
use candle_nn::ops::softmax;

/// Numerically stable sigmoid: `1 / (1 + e^-x)`.
fn sigmoid(x: &Tensor) -> Result<Tensor> {
    // affine(1, 1) turns e^-x into e^-x + 1
    x.neg()?.exp()?.affine(1.0, 1.0)?.recip()
}

/// Split a per-token mixing vector into the mHC gates and project the combination
/// matrix onto the Birkhoff polytope (doubly-stochastic) via Sinkhorn–Knopp.
///
/// `mixes`: `[n, (2 + hc) * hc]`, `hc_scale`: `[3]`, `hc_base`: `[(2 + hc) * hc]`.
/// Returns `(pre [n, hc], post [n, hc], comb [n, hc, hc])`.
///
/// Mirrors `hc_split_sinkhorn_kernel` in inference/kernel.py:
/// - `pre  = sigmoid(mixes[:, :hc]      * scale0 + base[:hc]) + eps`
/// - `post = 2 * sigmoid(mixes[:, hc:2hc] * scale1 + base[hc:2hc])`
/// - `comb = mixes[:, 2hc:] * scale2 + base[2hc:]`, then row-softmax, then
///   `sinkhorn_iters` alternating row/column normalizations.
pub fn hc_split_sinkhorn(
    mixes: &Tensor,
    hc_scale: &Tensor,
    hc_base: &Tensor,
    hc: usize,
    sinkhorn_iters: usize,
    eps: f64,
) -> Result<(Tensor, Tensor, Tensor)> {
    let n = mixes.dim(0)?;
    let s = hc_scale.to_vec1::<f32>()?;
    let (s0, s1, s2) = (s[0] as f64, s[1] as f64, s[2] as f64);

    // pre = sigmoid(mixes[:, :hc] * s0 + base[:hc]) + eps
    let pre = mixes
        .narrow(1, 0, hc)?
        .affine(s0, 0.0)?
        .broadcast_add(&hc_base.narrow(0, 0, hc)?)?;
    let pre = sigmoid(&pre)?.affine(1.0, eps)?;

    // post = 2 * sigmoid(mixes[:, hc:2hc] * s1 + base[hc:2hc])
    let post = mixes
        .narrow(1, hc, hc)?
        .affine(s1, 0.0)?
        .broadcast_add(&hc_base.narrow(0, hc, hc)?)?;
    let post = sigmoid(&post)?.affine(2.0, 0.0)?;

    // comb logits: [n, hc, hc]
    let comb = mixes
        .narrow(1, 2 * hc, hc * hc)?
        .affine(s2, 0.0)?
        .broadcast_add(&hc_base.narrow(0, 2 * hc, hc * hc)?)?
        .reshape((n, hc, hc))?;

    // comb = softmax(comb, dim=-1) + eps, then one column-normalization.
    let mut comb = softmax(&comb, D::Minus1)?.affine(1.0, eps)?;
    let col = comb.sum_keepdim(1)?.affine(1.0, eps)?; // column sums -> [n, 1, hc]
    comb = comb.broadcast_div(&col)?;

    // Sinkhorn iterations: alternate row then column normalization.
    for _ in 0..sinkhorn_iters.saturating_sub(1) {
        let row = comb.sum_keepdim(D::Minus1)?.affine(1.0, eps)?; // [n, hc, 1]
        comb = comb.broadcast_div(&row)?;
        let col = comb.sum_keepdim(1)?.affine(1.0, eps)?; // [n, 1, hc]
        comb = comb.broadcast_div(&col)?;
    }

    Ok((pre, post, comb))
}

/// The learned parameters of one Hyper-Connection mixer (one per attn / ffn site).
pub struct Hc {
    /// `[mix_hc, hc * dim]` — projects the (flattened, RMS-scaled) streams to the mixing vector.
    pub hc_fn: Tensor,
    /// `[mix_hc]`
    pub hc_base: Tensor,
    /// `[3]`
    pub hc_scale: Tensor,
    pub hc: usize,
    pub sinkhorn_iters: usize,
    pub eps: f64,
    pub norm_eps: f64,
}

impl Hc {
    /// `hc_pre`: collapse the `hc` residual streams into a single tensor for the
    /// sublayer, and return the `post` / `comb` weights needed to re-expand.
    ///
    /// `x`: `[b, s, hc, dim]` → `(y [b, s, dim], post [b, s, hc], comb [b, s, hc, hc])`.
    pub fn pre(&self, x: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let (b, s, hcn, d) = x.dims4()?;
        let x = x.to_dtype(DType::F32)?;
        let xf = x.reshape((b * s, hcn * d))?;

        // RMS scale over the flattened streams (no learnable weight here).
        let var = xf.sqr()?.mean_keepdim(1)?; // [b*s, 1]
        let rms = var.affine(1.0, self.norm_eps)?.powf(-0.5)?; // [b*s, 1]
        let mixes = xf.matmul(&self.hc_fn.t()?)?.broadcast_mul(&rms)?; // [b*s, mix_hc]

        let (pre, post, comb) = hc_split_sinkhorn(
            &mixes,
            &self.hc_scale,
            &self.hc_base,
            self.hc,
            self.sinkhorn_iters,
            self.eps,
        )?;

        // y = sum_hc(pre[..., None] * x)  -> [b, s, dim]
        let pre = pre.reshape((b, s, hcn, 1))?;
        let y = pre.broadcast_mul(&x)?.sum(2)?;

        Ok((y, post.reshape((b, s, hcn))?, comb.reshape((b, s, hcn, hcn))?))
    }

    /// `hc_post`: re-expand the sublayer output back into `hc` residual streams.
    ///
    /// `y[b,s,k,:] = post[b,s,k] * x[b,s,:] + sum_j comb[b,s,j,k] * residual[b,s,j,:]`.
    pub fn post(
        &self,
        x: &Tensor,            // [b, s, dim] (sublayer output)
        residual: &Tensor,     // [b, s, hc, dim]
        post: &Tensor,         // [b, s, hc]
        comb: &Tensor,         // [b, s, hc, hc]
    ) -> Result<Tensor> {
        let (b, s, d) = x.dims3()?;
        let hc = self.hc;
        let x = x.to_dtype(DType::F32)?;
        let residual = residual.to_dtype(DType::F32)?;

        // term1: post[..., None] * x[..., None, :]  -> [b, s, hc, dim]
        let term1 = post
            .reshape((b, s, hc, 1))?
            .broadcast_mul(&x.reshape((b, s, 1, d))?)?;

        // term2: sum_j comb[b,s,j,k] * residual[b,s,j,:]  -> [b, s, hc(k), dim]
        let comb_e = comb.reshape((b, s, hc, hc, 1))?; // [b,s,j,k,1]
        let res_e = residual.reshape((b, s, hc, 1, d))?; // [b,s,j,1,dim]
        let term2 = comb_e.broadcast_mul(&res_e)?.sum(2)?; // sum over j -> [b,s,k,dim]

        term1 + term2
    }
}
