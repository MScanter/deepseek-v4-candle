//! Multi-head Latent Attention (MLA) with a learnable softmax sink.
//!
//! [`sparse_attn`] ports the index-gathered kernel of the same name (inference/kernel.py);
//! [`sdpa_with_sink`] is its dense full-causal special case (key set = all past positions).
//! The reference uses a FlashAttention-style online softmax, but the running-max
//! subtraction cancels, so the closed form is:
//!
//! ```text
//! o[b,i,h,:] = Σ_j  w[b,h,i,j] · kv[b,j,:]
//! w[b,h,i,j] = exp(s_ij - m) / ( Σ_k exp(s_ik - m) + exp(sink[h] - m) )
//! s_ij       = scale · (q[b,i,h,:] · kv[b,j,:])
//! ```
//!
//! The `attn_sink[h]` term lives only in the denominator (no matching value), so it
//! drains probability mass — the model can attend to "nothing". Key and value are the
//! same single latent `kv` (MQA: one KV head shared across all query heads).
//!
//! [`Mla`] wires the V4 projections around it: low-rank Q (`wq_a → q_norm → wq_b` plus a
//! weightless per-head RMS), the single latent KV (`wkv → kv_norm`), interleaved YaRN
//! RoPE on the last `rope_head_dim` dims (and an *inverse* RoPE on the attention output),
//! and the grouped low-rank output projection (`wo_a` per group, then `wo_b`).
//!
//! [`Mla`] is the full-causal building block. Sliding-window / compressed (CSA/HCA) key
//! selection feeds [`sparse_attn`] via per-query index sets (built in a later step). The
//! FP8 activation-quant simulation of the KV non-RoPE dims (a QAT artifact) is omitted here.

use crate::rope::Rope;
use crate::sparse::{compress_topk_idxs, window_topk_idxs, Compressor, Indexer};
use candle_core::{DType, Device, Result, Tensor, D};

/// Scaled-dot-product attention with a per-head softmax sink.
///
/// - `q`:    `[b, s, h, d]`
/// - `kv`:   `[b, n, d]` (single latent head, used as both key and value)
/// - `sink`: `[h]`
/// - `causal`: query `i` (absolute position `n - s + i`) attends to keys `0..=(n-s+i)`.
///
/// Returns `o`: `[b, s, h, d]`.
pub fn sdpa_with_sink(
    q: &Tensor,
    kv: &Tensor,
    sink: &Tensor,
    scale: f64,
    causal: bool,
) -> Result<Tensor> {
    let mask = if causal {
        Some(causal_mask(q.dim(1)?, kv.dim(1)?, q.device())?)
    } else {
        None
    };
    attn_core(q, kv, sink, scale, mask.as_ref())
}

/// Index-gathered sink attention — ports `sparse_attn` (inference/kernel.py).
///
/// For each query `(b, i)`, `topk_idxs[b, i, :]` lists the key positions it attends to,
/// with a `-1` slot meaning "drop" (gather `0`, score `-inf`). Causality and the
/// sliding-window / compressed-KV *selection* are baked into the indices upstream, so this
/// carries no causal flag. The attention is otherwise identical to [`sdpa_with_sink`] — the
/// same per-head softmax sink in the denominator only — restricted to the gathered keys.
///
/// We realise the gather as a dense additive mask `[b, 1, s, n]`, which is exact at the toy
/// scale we test against; the reference fuses gather + online-softmax in one kernel for speed.
///
/// - `q`:         `[b, s, h, d]`
/// - `kv`:        `[b, n, d]` (single latent head; key = value)
/// - `sink`:      `[h]`
/// - `topk_idxs`: `[b, s, topk]`, integer key indices in `-1..n`.
///
/// Returns `o`: `[b, s, h, d]`.
pub fn sparse_attn(
    q: &Tensor,
    kv: &Tensor,
    sink: &Tensor,
    topk_idxs: &Tensor,
    scale: f64,
) -> Result<Tensor> {
    let (b, s, _, _) = q.dims4()?;
    let n = kv.dim(1)?;
    let (_, _, topk) = topk_idxs.dims3()?;
    let idxs = topk_idxs.to_dtype(DType::I64)?.flatten_all()?.to_vec1::<i64>()?;

    // Scatter each query's key indices into a dense additive mask: 0 = keep, -inf = drop.
    let mut mask = vec![f32::NEG_INFINITY; b * s * n];
    for bi in 0..b {
        for i in 0..s {
            let row = (bi * s + i) * topk;
            for t in 0..topk {
                let idx = idxs[row + t];
                if idx >= 0 && (idx as usize) < n {
                    mask[(bi * s + i) * n + idx as usize] = 0.0;
                }
            }
        }
    }
    let mask = Tensor::from_vec(mask, (b, 1, s, n), q.device())?;
    attn_core(q, kv, sink, scale, Some(&mask))
}

/// Sink-softmax attention core shared by [`sdpa_with_sink`] and [`sparse_attn`].
///
/// `mask` is an additive bias broadcast over heads to `[b, h, s, n]` (`0` keep, `-inf`
/// drop); `None` makes every key visible. Key and value are the same latent `kv` (MQA).
/// The running-max subtraction is exact only if each query keeps ≥1 key — the reference
/// always includes self-attention, so this holds.
fn attn_core(
    q: &Tensor,
    kv: &Tensor,
    sink: &Tensor,
    scale: f64,
    mask: Option<&Tensor>,
) -> Result<Tensor> {
    let (b, _, h, d) = q.dims4()?;
    let n = kv.dim(1)?;
    let q = q.to_dtype(DType::F32)?;
    let kv = kv.to_dtype(DType::F32)?;

    // [b,s,h,d] -> [b,h,s,d]; kv [b,n,d] -> [b,h,n,d] (broadcast the single head).
    let qh = q.transpose(1, 2)?.contiguous()?;
    let kvh = kv.unsqueeze(1)?.broadcast_as((b, h, n, d))?.contiguous()?;

    // scores[b,h,s,n] = scale · q·kvᵀ, plus the additive mask.
    let scores = qh
        .matmul(&kvh.transpose(2, 3)?.contiguous()?)?
        .affine(scale, 0.0)?;
    let scores = match mask {
        Some(m) => scores.broadcast_add(m)?,
        None => scores,
    };

    // Softmax over keys with the sink term in the denominator only.
    let m = scores.max_keepdim(D::Minus1)?; // [b,h,s,1], max over visible keys
    let exp_scores = scores.broadcast_sub(&m)?.exp()?; // masked keys -> exp(-inf) = 0
    let sum_keys = exp_scores.sum_keepdim(D::Minus1)?; // [b,h,s,1]
    let sink_term = sink.reshape((1, h, 1, 1))?.broadcast_sub(&m)?.exp()?; // [b,h,s,1]
    let denom = (sum_keys + sink_term)?;
    let weights = exp_scores.broadcast_div(&denom)?; // [b,h,s,n]

    // o[b,h,s,d] = weights · kv  ->  [b,s,h,d]
    weights.matmul(&kvh)?.transpose(1, 2)?.contiguous()
}

/// Additive causal mask `[1, 1, s, n]`: `0` where attended, `-inf` otherwise.
/// Queries align to the last `s` of the `n` key positions (offset `n - s`).
fn causal_mask(s: usize, n: usize, dev: &Device) -> Result<Tensor> {
    let offset = n - s;
    let mut data = vec![0f32; s * n];
    for i in 0..s {
        for j in 0..n {
            if j > i + offset {
                data[i * n + j] = f32::NEG_INFINITY;
            }
        }
    }
    Tensor::from_vec(data, (1, 1, s, n), dev)
}

/// `F.linear`: `y = x · Wᵀ` for weight `W` shaped `[out, in]`. Works on any
/// `[..., in]` input.
pub(crate) fn linear(x: &Tensor, w: &Tensor) -> Result<Tensor> {
    let dims = x.dims().to_vec();
    let in_f = *dims.last().expect("non-scalar input");
    let rows: usize = dims[..dims.len() - 1].iter().product();
    let out_f = w.dim(0)?;
    let y = x.reshape((rows, in_f))?.matmul(&w.t()?.contiguous()?)?;
    let mut out_dims = dims[..dims.len() - 1].to_vec();
    out_dims.push(out_f);
    y.reshape(out_dims)
}

/// RMSNorm over the last dim: `gamma · x / sqrt(mean(x²) + eps)`. `gamma = None`
/// gives the weightless variant V4 applies per-head to Q.
pub(crate) fn rms_norm(x: &Tensor, gamma: Option<&Tensor>, eps: f64) -> Result<Tensor> {
    let x = x.to_dtype(DType::F32)?;
    let var = x.sqr()?.mean_keepdim(D::Minus1)?;
    let normed = x.broadcast_div(&var.affine(1.0, eps)?.sqrt()?)?;
    match gamma {
        Some(g) => normed.broadcast_mul(&g.to_dtype(DType::F32)?),
        None => Ok(normed),
    }
}

/// Apply RoPE to only the last `rd` dims of `x` (`[b, s, h, d]`), leaving the first
/// `d - rd` ("nope") dims untouched.
fn rope_tail(rope: &Rope, x: &Tensor, start_pos: usize, rd: usize, inverse: bool) -> Result<Tensor> {
    let d = x.dim(D::Minus1)?;
    let nope = d - rd;
    let head = x.narrow(D::Minus1, 0, nope)?.contiguous()?;
    let tail = x.narrow(D::Minus1, nope, rd)?.contiguous()?;
    let tail = rope.apply(&tail, start_pos, inverse)?;
    Tensor::cat(&[&head, &tail], D::Minus1)
}

/// Like [`rope_tail`] but rotates each row at its own absolute position (for the KV
/// compressor's strided block positions `0, ratio, 2*ratio, ...`). Forward only.
pub(crate) fn rope_tail_at(rope: &Rope, x: &Tensor, positions: &[usize], rd: usize) -> Result<Tensor> {
    let d = x.dim(D::Minus1)?;
    let nope = d - rd;
    let head = x.narrow(D::Minus1, 0, nope)?.contiguous()?;
    let tail = x.narrow(D::Minus1, nope, rd)?.contiguous()?;
    let tail = rope.apply_at(&tail, positions, false)?;
    Tensor::cat(&[&head, &tail], D::Minus1)
}

/// Multi-head Latent Attention block (one per layer). Fields are the loaded weights
/// (`[out, in]` for projections, `[dim]` for norms) plus the layer's scalar dims.
pub struct Mla {
    /// `[q_lora_rank, dim]`
    pub wq_a: Tensor,
    /// `[q_lora_rank]` RMSNorm gamma
    pub q_norm: Tensor,
    /// `[n_heads * head_dim, q_lora_rank]`
    pub wq_b: Tensor,
    /// `[head_dim, dim]`
    pub wkv: Tensor,
    /// `[head_dim]` RMSNorm gamma
    pub kv_norm: Tensor,
    /// `[n_groups * o_lora_rank, n_heads * head_dim / n_groups]`
    pub wo_a: Tensor,
    /// `[dim, n_groups * o_lora_rank]`
    pub wo_b: Tensor,
    /// `[n_heads]`
    pub attn_sink: Tensor,
    pub n_heads: usize,
    pub head_dim: usize,
    pub rope_head_dim: usize,
    pub n_groups: usize,
    pub o_lora_rank: usize,
    /// Sliding-window size: query `i` attends only to keys `max(i - window + 1, 0) ..= i`.
    pub window_size: usize,
    /// Per-layer KV-compression ratio: `0` = pure sliding-window, `>0` = window + compressed
    /// KV (HCA at `128`, CSA at `4`). Indexes `compress_ratios[layer_id]` in the reference.
    pub compress_ratio: usize,
    /// The KV compressor, present iff `compress_ratio > 0`. Its compressed blocks are
    /// concatenated onto the latent `kv` and addressed by the compressed key indices.
    pub compressor: Option<Compressor>,
    /// The learned block selector, present iff this is a CSA layer (`compress_ratio == 4`).
    /// When `Some`, it replaces the deterministic [`compress_topk_idxs`] selection with a
    /// top-k over learned per-query block scores.
    pub indexer: Option<Indexer>,
    pub eps: f64,
    pub scale: f64,
}

impl Mla {
    /// `x`: `[b, s, dim]` → `[b, s, dim]`. `rope` supplies the positional tables for
    /// this layer; `start_pos` is the absolute position of the first query.
    pub fn forward(&self, x: &Tensor, rope: &Rope, start_pos: usize) -> Result<Tensor> {
        let (b, s, _) = x.dims3()?;
        let (h, hd, rd) = (self.n_heads, self.head_dim, self.rope_head_dim);

        // Q: low-rank down (`qr`, also fed to the CSA indexer), up, weightless per-head RMS, RoPE.
        let qr = rms_norm(&linear(x, &self.wq_a)?, Some(&self.q_norm), self.eps)?;
        let q = linear(&qr, &self.wq_b)?.reshape((b, s, h, hd))?;
        let q = rms_norm(&q, None, self.eps)?;
        let q = rope_tail(rope, &q, start_pos, rd, false)?;

        // KV: single latent head, RoPE on the rope dims.
        let kv = rms_norm(&linear(x, &self.wkv)?, Some(&self.kv_norm), self.eps)?;
        let kv = rope_tail(rope, &kv.reshape((b, s, 1, hd))?, start_pos, rd, false)?;
        let kv = kv.reshape((b, s, hd))?;

        // Sparse attention. Window keys index the latent `kv` by absolute position; a
        // compressed (HCA/CSA) layer pools `compress_ratio` tokens per block, appends those
        // blocks to `kv`, and extends the per-query index set with the blocks' (offset)
        // positions — then one `sparse_attn` runs over the window ∪ compressed key union.
        let k_win = self.window_size.min(s);
        let window = window_topk_idxs(self.window_size, s, x.device())?
            .unsqueeze(0)?
            .broadcast_as((b, s, k_win))?
            .contiguous()?; // [b, s, k_win]
        let (kv, idxs) = match &self.compressor {
            Some(comp) => {
                let kv_compress = comp.compress(x, rope)?; // [b, n_blocks, hd]
                let offset = s; // compressed blocks sit after the `s` window-KV rows
                // HCA selects every visible block deterministically; CSA's learned indexer keeps a
                // top-k subset, scored against its own (separate) compressed KV.
                let cidxs = match &self.indexer {
                    Some(idx) => idx.select(x, &qr, rope)?, // [b, s, index_topk]
                    None => {
                        let c = compress_topk_idxs(self.compress_ratio, s, offset, x.device())?;
                        let cols = c.dim(1)?;
                        c.unsqueeze(0)?.broadcast_as((b, s, cols))?.contiguous()?
                    }
                };
                let kv = Tensor::cat(&[&kv, &kv_compress], 1)?;
                let idxs = Tensor::cat(&[&window, &cidxs], D::Minus1)?; // [b, s, k_win + k_compress]
                (kv, idxs)
            }
            None => (kv, window),
        };
        let idxs = idxs.contiguous()?;
        let o = sparse_attn(&q, &kv, &self.attn_sink, &idxs, self.scale)?;
        let o = rope_tail(rope, &o, start_pos, rd, true)?;

        // Grouped low-rank output projection: einsum("bsgd,grd->bsgr"), then wo_b.
        let (g, r) = (self.n_groups, self.o_lora_rank);
        let din = h * hd / g;
        let o = o.reshape((b, s, g, din))?.permute((2, 0, 1, 3))?.reshape((g, b * s, din))?;
        let wa = self.wo_a.reshape((g, r, din))?.transpose(1, 2)?.contiguous()?; // [g, din, r]
        let og = o
            .matmul(&wa)? // [g, b*s, r]
            .reshape((g, b, s, r))?
            .permute((1, 2, 0, 3))? // [b, s, g, r]
            .reshape((b, s, g * r))?;
        linear(&og, &self.wo_b)
    }
}
