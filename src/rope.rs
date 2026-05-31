//! YaRN rotary position embeddings (RoPE).
//!
//! Port of `precompute_freqs_cis` + `apply_rotary_emb` from inference/model.py.
//! The convention is *interleaved*: consecutive pairs `(x[2i], x[2i+1])` form a
//! complex number rotated by angle `pos * freq[i]`. YaRN interpolates the
//! low-frequency (long-range) dims by `1/factor` while leaving the high-frequency
//! dims untouched, with a smooth linear ramp between the `beta_fast` / `beta_slow`
//! correction bounds. Frequencies are computed in f64 (matching the reference's
//! float32 path closely) and stored as f32 cos/sin tables.

use candle_core::{DType, Device, Result, Tensor};

/// Precomputed cos/sin tables for one RoPE configuration.
pub struct Rope {
    cos: Tensor, // [max_seq, dim/2]
    sin: Tensor, // [max_seq, dim/2]
}

impl Rope {
    /// Build cos/sin tables with YaRN scaling.
    ///
    /// `original_seq_len == 0` disables YaRN (plain RoPE at `base`), matching the
    /// pure sliding-window branch in `Attention.__init__`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        dim: usize,
        max_seq: usize,
        original_seq_len: usize,
        base: f64,
        factor: f64,
        beta_fast: f64,
        beta_slow: f64,
        dev: &Device,
    ) -> Result<Self> {
        let half = dim / 2;

        // Base inverse frequencies: 1 / base^(2i/dim).
        let mut freqs: Vec<f64> = (0..half)
            .map(|i| 1.0 / base.powf((2 * i) as f64 / dim as f64))
            .collect();

        // YaRN frequency interpolation.
        if original_seq_len > 0 {
            let (low, high) = correction_range(beta_fast, beta_slow, dim, base, original_seq_len);
            let denom = if low == high { 0.001 } else { high - low };
            for (i, f) in freqs.iter_mut().enumerate() {
                let ramp = (((i as f64) - low) / denom).clamp(0.0, 1.0);
                let smooth = 1.0 - ramp;
                // keep = f (high-freq, smooth=1); interpolate = f/factor (low-freq, smooth=0).
                *f = *f / factor * (1.0 - smooth) + *f * smooth;
            }
        }

        // cos/sin over positions 0..max_seq.
        let mut cos = Vec::with_capacity(max_seq * half);
        let mut sin = Vec::with_capacity(max_seq * half);
        for p in 0..max_seq {
            for &f in &freqs {
                let angle = p as f64 * f;
                cos.push(angle.cos() as f32);
                sin.push(angle.sin() as f32);
            }
        }
        Ok(Self {
            cos: Tensor::from_vec(cos, (max_seq, half), dev)?,
            sin: Tensor::from_vec(sin, (max_seq, half), dev)?,
        })
    }

    /// `[max_seq, dim/2]` cosine table.
    pub fn cos(&self) -> &Tensor {
        &self.cos
    }
    /// `[max_seq, dim/2]` sine table.
    pub fn sin(&self) -> &Tensor {
        &self.sin
    }

    /// Rotate the last dim of `x` (`[b, s, h, d]`, `d` even) by the positional
    /// angles for `start_pos .. start_pos + s`. `inverse` de-rotates (conjugate).
    pub fn apply(&self, x: &Tensor, start_pos: usize, inverse: bool) -> Result<Tensor> {
        let s = x.dim(1)?;
        let cos = self.cos.narrow(0, start_pos, s)?;
        let sin = self.sin.narrow(0, start_pos, s)?;
        self.apply_rows(x, &cos, &sin, inverse)
    }

    /// Rotate the last dim of `x` (`[b, s, h, d]`) by the angles for the given absolute
    /// `positions` — one per query, `positions.len() == s`. Generalises [`apply`] to
    /// non-contiguous positions: the KV compressor ropes each block at its first token's
    /// position, i.e. the strided sequence `0, ratio, 2*ratio, ...`.
    pub fn apply_at(&self, x: &Tensor, positions: &[usize], inverse: bool) -> Result<Tensor> {
        let idx = Tensor::from_vec(
            positions.iter().map(|&p| p as u32).collect::<Vec<_>>(),
            (positions.len(),),
            self.cos.device(),
        )?;
        let cos = self.cos.index_select(&idx, 0)?;
        let sin = self.sin.index_select(&idx, 0)?;
        self.apply_rows(x, &cos, &sin, inverse)
    }

    /// Rotation core shared by [`apply`] and [`apply_at`]. `cos`/`sin` are the per-position
    /// rows `[s, half]` already selected for this call; `inverse` negates the sine (conjugate).
    fn apply_rows(&self, x: &Tensor, cos: &Tensor, sin: &Tensor, inverse: bool) -> Result<Tensor> {
        let (b, s, h, d) = x.dims4()?;
        let half = d / 2;
        let x = x.to_dtype(DType::F32)?;

        // [s, half] -> [1, s, 1, half] to broadcast over (b, h).
        let cos = cos.reshape((1, s, 1, half))?;
        let sin = sin.reshape((1, s, 1, half))?;
        let sin = if inverse { sin.neg()? } else { sin };

        // Split interleaved even/odd: [b, s, h, d] -> [b, s, h, half, 2].
        let xr = x.reshape((b, s, h, half, 2))?;
        let x_even = xr.narrow(4, 0, 1)?.contiguous()?.reshape((b, s, h, half))?;
        let x_odd = xr.narrow(4, 1, 1)?.contiguous()?.reshape((b, s, h, half))?;

        // (e + i·o)(cos + i·sin) = (e·cos - o·sin) + i·(e·sin + o·cos)
        let out_even = (x_even.broadcast_mul(&cos)? - x_odd.broadcast_mul(&sin)?)?;
        let out_odd = (x_even.broadcast_mul(&sin)? + x_odd.broadcast_mul(&cos)?)?;

        // Re-interleave: [b, s, h, half, 2] -> [b, s, h, d].
        Tensor::stack(&[&out_even, &out_odd], 4)?.reshape((b, s, h, d))
    }
}

/// `find_correction_dim`: the rotary dim at which a given number of rotations
/// occurs over `max_seq_len` positions.
fn correction_dim(num_rotations: f64, dim: usize, base: f64, max_seq_len: usize) -> f64 {
    (dim as f64) * (max_seq_len as f64 / (num_rotations * 2.0 * std::f64::consts::PI)).ln()
        / (2.0 * base.ln())
}

/// `find_correction_range`, clamped to `[0, dim-1]`.
fn correction_range(
    low_rot: f64,
    high_rot: f64,
    dim: usize,
    base: f64,
    max_seq_len: usize,
) -> (f64, f64) {
    let low = correction_dim(low_rot, dim, base, max_seq_len).floor().max(0.0);
    let high = correction_dim(high_rot, dim, base, max_seq_len)
        .ceil()
        .min((dim - 1) as f64);
    (low, high)
}
