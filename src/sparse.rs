//! Sparse KV selection: the per-query key-index sets for the hybrid attention variants.
//!
//! Each layer attends not to all past keys but to a selected subset — a sliding window,
//! plus (for the CSA/HCA layers) compressed-KV positions. The selection is expressed as
//! `topk_idxs` (key indices, `-1` = empty slot) and consumed by
//! [`crate::attention::sparse_attn`]. These functions port the prefill (`start_pos == 0`)
//! branches of `get_window_topk_idxs` / `get_compress_topk_idxs` (inference/model.py).

use crate::attention::{linear, rms_norm, rope_tail_at};
use crate::rope::Rope;
use candle_core::{DType, Device, Result, Tensor, D};

/// Softmax over an arbitrary dim, computed stably (subtract the running max first).
fn softmax_dim(x: &Tensor, dim: usize) -> Result<Tensor> {
    let m = x.max_keepdim(dim)?;
    let e = x.broadcast_sub(&m)?.exp()?;
    e.broadcast_div(&e.sum_keepdim(dim)?)
}

/// Gated mean-pool over the block axis (dim 2): softmax the per-dimension `gates` across the
/// block's tokens, then take that weighted sum of `values`. Both inputs are
/// `[b, n_blocks, n_tok, d]` (`n_tok = ratio`, or `2*ratio` for the overlap path); returns
/// `[b, n_blocks, d]`. The learned pooling at the heart of the KV compressor —
/// `(kv * score.softmax(dim=n_tok)).sum(dim=n_tok)`.
fn gated_pool(values: &Tensor, gates: &Tensor) -> Result<Tensor> {
    let w = softmax_dim(gates, 2)?;
    values.mul(&w)?.sum(2)
}

/// Build the overlapping `2*ratio`-token windows for the CSA compressor (`compress_ratio == 4`).
/// Input `t`: `[b, n_blocks, ratio, 2d]` (the projection's two halves). Each output block keeps
/// its own tokens' *second*-half dims (`d:2d`) and prepends the *previous* block's tokens'
/// *first*-half dims (`0:d`); block 0 has no predecessor, so its overlap half is `fill`
/// (`-inf` for the gate so softmax zeroes it). Returns `[b, n_blocks, 2*ratio, d]`. Ports
/// `Compressor.overlap_transform`.
fn overlap_windows(t: &Tensor, d: usize, fill: f32) -> Result<Tensor> {
    let (b, nb, ratio, _) = t.dims4()?;
    let normal = t.narrow(3, d, d)?; // current block, 2nd-half dims  -> positions [ratio:2ratio]
    let first = t.narrow(3, 0, d)?; // 1st-half dims, the overlap source
    let fill_block = Tensor::full(fill, (b, 1, ratio, d), t.device())?;
    // Shift `first` down one block (block i gets block i-1's first half; block 0 gets `fill`).
    let prev = if nb > 1 {
        Tensor::cat(&[&fill_block, &first.narrow(1, 0, nb - 1)?], 1)?
    } else {
        fill_block
    };
    Tensor::cat(&[&prev, &normal], 2) // [b, nb, 2*ratio, d]
}

/// Sliding-window causal key indices for a prefill of `seqlen` tokens, window `window`.
///
/// Returns `[seqlen, k]` (`k = min(seqlen, window)`), where row `i` lists the key positions
/// query `i` attends to — `max(i - window + 1, 0) ..= i` — right-padded with `-1` when fewer
/// than `k` keys exist (the first `window - 1` queries). Ports the `start_pos == 0` branch of
/// `get_window_topk_idxs`.
pub fn window_topk_idxs(window: usize, seqlen: usize, dev: &Device) -> Result<Tensor> {
    let k = window.min(seqlen);
    let mut data = vec![-1i64; seqlen * k];
    for i in 0..seqlen {
        let lo = (i + 1).saturating_sub(window); // max(i - window + 1, 0)
        for j in 0..k {
            let key = lo + j;
            if key <= i {
                data[i * k + j] = key as i64;
            }
        }
    }
    Tensor::from_vec(data, (seqlen, k), dev)
}

/// Compressed-KV causal key indices for a prefill of `seqlen` tokens, compression ratio
/// `ratio` (the HCA / deterministic path — no learned selection).
///
/// Returns `[seqlen, seqlen / ratio]`: row `i` lists the compressed blocks visible to query
/// `i`. Block `c` pools the `ratio` consecutive tokens at positions `c*ratio ..(c+1)*ratio`,
/// and becomes visible only once fully in the past — i.e. for `c < (i + 1) / ratio` — sitting
/// at cache index `c + offset` (where the compressed KVs are concatenated after the window
/// KVs); the remaining slots are `-1`. Ports the `start_pos == 0` branch of
/// `get_compress_topk_idxs`.
pub fn compress_topk_idxs(ratio: usize, seqlen: usize, offset: usize, dev: &Device) -> Result<Tensor> {
    let cols = seqlen / ratio;
    let mut data = vec![-1i64; seqlen * cols];
    for i in 0..seqlen {
        let visible = ((i + 1) / ratio).min(cols);
        for c in 0..visible {
            data[i * cols + c] = (c + offset) as i64;
        }
    }
    Tensor::from_vec(data, (seqlen, cols), dev)
}

/// Learned KV compressor — the prefill path for both HCA (non-overlap) and CSA (overlap).
///
/// Pools each block of `compress_ratio` consecutive tokens into a single compressed KV vector
/// via gated (softmax) pooling, then RMS-norms it and RoPEs its tail dims at the block's
/// leading position. For CSA (`compress_ratio == 4`) the windows overlap: `wkv`/`wgate` project
/// to `2*head_dim` and each block also pools the previous block's tokens (see
/// [`overlap_windows`]). Ports the `start_pos == 0` branch of `Compressor.forward`
/// (inference/model.py) for sequences whose length is a multiple of `compress_ratio`.
///
/// Deliberately omitted (and documented as such): the `seqlen % ratio` remainder carry, the
/// incremental-decode `kv_state`/`score_state` buffers, the Hadamard rotation + FP4/FP8
/// activation-quant QAT simulation (a no-op for full-precision scoring — see [`crate::sparse`]).
pub struct Compressor {
    /// `[coff*head_dim, dim]` — KV projection (`coff = 2` for the CSA overlap path, else `1`).
    pub wkv: Tensor,
    /// `[coff*head_dim, dim]` — gate projection.
    pub wgate: Tensor,
    /// `[ratio, coff*head_dim]` — within-block absolute position embedding, added to the gate.
    pub ape: Tensor,
    /// `[head_dim]` — RMSNorm gamma.
    pub norm: Tensor,
    pub compress_ratio: usize,
    pub head_dim: usize,
    pub rope_head_dim: usize,
    pub eps: f64,
}

impl Compressor {
    /// `x`: `[b, s, dim]` (prefill, `s % compress_ratio == 0`) → compressed KV
    /// `[b, s / ratio, head_dim]`. `rope` supplies the positional tables; it is unused when
    /// `rope_head_dim == 0`.
    pub fn compress(&self, x: &Tensor, rope: &Rope) -> Result<Tensor> {
        let (b, s, _) = x.dims3()?;
        let (ratio, d, rd) = (self.compress_ratio, self.head_dim, self.rope_head_dim);
        let nb = s / ratio;
        let overlap = ratio == 4; // CSA layers use overlapping windows (coff = 2)
        let proj_d = if overlap { 2 * d } else { d }; // wkv/wgate output width

        // Project to KV and gate, fold into [b, n_blocks, ratio, proj_d], add the within-block APE.
        let kv = linear(x, &self.wkv)?.reshape((b, nb, ratio, proj_d))?;
        let ape = self.ape.reshape((1, 1, ratio, proj_d))?;
        let score = linear(x, &self.wgate)?
            .reshape((b, nb, ratio, proj_d))?
            .broadcast_add(&ape)?;

        // Gated softmax pool over the block's tokens -> [b, nb, d]. The overlap path stitches
        // each block's `2*ratio` window (own 2nd-half dims + previous block's 1st-half dims)
        // first, with `-inf` gates masking block 0's absent predecessor.
        let pooled = if overlap {
            let kv = overlap_windows(&kv, d, 0.0)?;
            let score = overlap_windows(&score, d, f32::NEG_INFINITY)?;
            gated_pool(&kv, &score)?
        } else {
            gated_pool(&kv, &score)?
        };

        // RMSNorm.
        let pooled = rms_norm(&pooled, Some(&self.norm), self.eps)?;
        if rd == 0 {
            return Ok(pooled);
        }

        // RoPE each compressed block at its leading token's position i*ratio.
        let positions: Vec<usize> = (0..nb).map(|i| i * ratio).collect();
        let roped = rope_tail_at(rope, &pooled.reshape((b, nb, 1, d))?, &positions, rd)?;
        roped.reshape((b, nb, d))
    }
}

/// Learned KV-block selector for the CSA layers (`compress_ratio == 4`) — the prefill path of
/// `Indexer.forward` (inference/model.py 402-433).
///
/// Scores every compressed block against each query through a small per-head attention, then
/// keeps the `index_topk` highest-scoring *visible* blocks. The score for query `i`, block `t` is
/// `sum_h relu(q[i,h] · kv[t]) * weight[i,h]`, where the compressed KVs come from this indexer's
/// own (overlapping) [`Compressor`], the per-head queries from projecting the low-rank `qr` by
/// `wq_b`, and the per-head `weight` from `weights_proj(x)` scaled by `head_dim^-0.5 · n_heads^-0.5`.
/// Future blocks (`t >= (i+1)/ratio`) are masked out (causal); the chosen blocks are returned as
/// cache indices `block + offset` (`offset = seqlen`, where the compressed KVs are concatenated
/// after the window KVs), `-1`-padded when fewer than `index_topk` blocks are visible — exactly
/// the shape [`compress_topk_idxs`] produces for the deterministic HCA path.
///
/// Deliberately omitted (provably exact at full precision, documented in `indexer_golden.py`): the
/// Hadamard `rotate_activation` and FP4 act-quant. The same orthogonal rotation is applied to both
/// `q` and `kv`, so it cancels in the score dot product (`(Hq)·(Hk) = q·k`); `relu`/`top-k` then
/// see identical values. FP4 is a QAT artifact, like the omitted [`Compressor`] act-quant.
pub struct Indexer {
    /// `[n_heads * head_dim, q_lora_rank]` — projects the low-rank query `qr` to per-head queries.
    pub wq_b: Tensor,
    /// `[n_heads, dim]` — per-head scoring weights projected from `x`.
    pub weights_proj: Tensor,
    /// The indexer's own KV compressor (overlapping, CSA `coff = 2`).
    pub compressor: Compressor,
    pub n_heads: usize,
    pub head_dim: usize,
    pub rope_head_dim: usize,
    /// How many blocks to keep per query (`min`'d with the block count).
    pub index_topk: usize,
    pub compress_ratio: usize,
    /// `head_dim^-0.5` — the per-head weight is further scaled by `n_heads^-0.5`.
    pub scale: f64,
}

impl Indexer {
    /// `x`: `[b, s, dim]` (prefill, `s % compress_ratio == 0`); `qr`: `[b, s, q_lora_rank]` the
    /// low-rank query. Returns the selected block indices `[b, s, k]` (`k = min(index_topk, nb)`),
    /// each as a cache index `block + offset` (`offset = s`) or `-1`. `rope` is unused at
    /// `rope_head_dim == 0`.
    pub fn select(&self, x: &Tensor, qr: &Tensor, rope: &Rope) -> Result<Tensor> {
        let (b, s, _) = x.dims3()?;
        let (h, hd, rd, ratio) = (
            self.n_heads,
            self.head_dim,
            self.rope_head_dim,
            self.compress_ratio,
        );
        let nb = s / ratio;
        let offset = s; // compressed blocks sit after the `s` window-KV rows
        let k = self.index_topk.min(nb);

        // Per-head queries: project qr by wq_b -> [b, s, h, hd], RoPE the tail dims at abs pos i.
        let q = linear(qr, &self.wq_b)?.reshape((b, s, h, hd))?;
        let q = if rd > 0 {
            let positions: Vec<usize> = (0..s).collect();
            rope_tail_at(rope, &q, &positions, rd)?
        } else {
            q
        };

        // Compressed KV blocks -> [b, nb, hd].
        let kv = self.compressor.compress(x, rope)?;

        // Per-head scoring weights: weights_proj(x) scaled by head_dim^-0.5 * n_heads^-0.5.
        let wsc = self.scale * (h as f64).powf(-0.5);
        let weights = linear(x, &self.weights_proj)?.affine(wsc, 0.0)?; // [b, s, h]

        // score[b,s,h,t] = relu(q[b,s,h] · kv[b,t]); then weight per head and sum over heads.
        let qm = q.reshape((b, s * h, hd))?;
        let kvt = kv.transpose(1, 2)?.contiguous()?; // [b, hd, nb]
        let scores = qm.matmul(&kvt)?.reshape((b, s, h, nb))?.relu()?;
        let scores = scores.broadcast_mul(&weights.unsqueeze(D::Minus1)?)?;
        let index_score = scores.sum(2)?; // [b, s, nb]

        // Causal top-k per query: among blocks visible at i (t < (i+1)/ratio), keep the k highest,
        // emit their cache indices (block + offset), pad with -1.
        let flat = index_score.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let mut out = vec![-1i64; b * s * k];
        for bi in 0..b {
            for i in 0..s {
                let visible = (i + 1) / ratio;
                let base = (bi * s + i) * nb;
                let mut cand: Vec<(f32, usize)> = (0..nb)
                    .filter(|&t| t < visible)
                    .map(|t| (flat[base + t], t))
                    .collect();
                cand.sort_by(|a, c| {
                    c.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
                });
                for (j, &(_, t)) in cand.iter().take(k).enumerate() {
                    out[(bi * s + i) * k + j] = (t + offset) as i64;
                }
            }
        }
        Tensor::from_vec(out, (b, s, k), x.device())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    /// Gated pooling over a single block of `ratio = 2` tokens (`d = 2`): each output dim is a
    /// softmax-over-the-2-tokens weighted average, with the gate taken per dimension.
    #[test]
    fn gated_pool_softmax_weights_over_block() -> Result<()> {
        let dev = Device::Cpu;
        // values: tok0 = [1,2], tok1 = [3,4];  gates: tok0 = [0,1], tok1 = [0,0].
        let values = Tensor::from_vec(vec![1f32, 2., 3., 4.], (1, 1, 2, 2), &dev)?;
        let gates = Tensor::from_vec(vec![0f32, 1., 0., 0.], (1, 1, 2, 2), &dev)?;

        let out = gated_pool(&values, &gates)?;
        assert_eq!(out.dims(), &[1, 1, 2]);

        let got = out.flatten_all()?.to_vec1::<f32>()?;
        // dim0: softmax([0,0]) = [.5,.5]      -> .5*1 + .5*3   = 2.0
        // dim1: softmax([1,0]) = [.7310586,.] -> .7310586*2 + .2689414*4 = 2.5378828
        let golden = [2.0_f32, 2.537_882_8];
        for (k, (a, b)) in got.iter().zip(golden.iter()).enumerate() {
            assert!((a - b).abs() < 1e-5, "out[{k}] = {a}, expected {b}");
        }
        Ok(())
    }
}
