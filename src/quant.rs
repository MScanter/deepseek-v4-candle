//! FP4 / FP8 weight dequantization — decode a checkpoint's low-precision weights to f32.
//!
//! V4 stores most weights as **FP8** (`e4m3`) and the MoE experts as **FP4** (`e2m1`), each with a
//! block-shared **`e8m0`** (UE8M0) scale (`config.json`: `dtype=fp8`, `expert_dtype=fp4`,
//! `scale_fmt=ue8m0`). The reference's quantized `linear` (inference/model.py 108-120) runs a fused
//! CUDA tilelang kernel: it `act_quant`s the *activations* to FP8 blocks, then `fp8_gemm`/`fp4_gemm`
//! dequantizes the weight on the fly. We can't run those kernels (CUDA-only) — and don't need to:
//! decoding the stored weights to f32 and feeding the existing f32 [`crate::attention::linear`]
//! realizes the checkpoint's weight VALUES exactly (bit-for-bit per element). The only thing we omit
//! is the activation quantization, which is a speed/footprint optimization (and a QAT artifact) that
//! only *loses* precision relative to f32 weights — so omitting it makes the math strictly more
//! accurate, not less. Hence: **bit-exact weight dequant here; activation-quant + fused GEMM left to
//! the (CUDA) reference.**
//!
//! ## Bit layouts (all little-endian per element)
//! - **`e4m3`** (FP8 weight): `S EEEE MMM`, exponent bias 7. Normal `(-1)^S·(1+M/8)·2^(E-7)`;
//!   subnormal at `E=0`: `(-1)^S·(M/8)·2^-6`. This is the *fn* variant: no infinities, `NaN` only at
//!   `S.1111.111` (`0x7F`/`0xFF`), max finite `448` (`0x7E`).
//! - **`e2m1`** (FP4 weight): `S EE M`, exponent bias 1 → magnitudes `{0,.5,1,1.5,2,3,4,6}`, sign in
//!   bit 3. Two nibbles pack into one byte (`float4_e2m1fn_x2`): logical element `2k` in the **low**
//!   nibble, `2k+1` in the **high** nibble.
//! - **`e8m0`** (UE8M0 scale): unsigned 8-bit exponent, value `2^(b-127)`; `NaN` at `0xFF`.
//!
//! ## Storage (inference/model.py `Linear.__init__`)
//! - FP8: weight `[out, in]`; scale `[⌈out/128⌉, ⌈in/128⌉]` — one scale per 128×128 weight tile.
//! - FP4: weight `[out, in/2]` (packed, logically `[out, in]`); scale `[out, in/32]` — per row, one
//!   scale per 32 elements along the input dim.
//!
//! Nibble-packing order is the one bit of layout we can't cross-check against torch locally; it
//! follows the documented `float4_e2m1fn_x2` convention (low nibble = even logical index).

use candle_core::{Device, Result, Tensor};

/// FP8 weight tile size: one `e8m0` scale per `FP8_BLOCK`×`FP8_BLOCK` weight tile (`block_size`).
pub const FP8_BLOCK: usize = 128;
/// FP4 block size: one `e8m0` scale per `FP4_BLOCK` weights along the input dim (`fp4_block_size`).
pub const FP4_BLOCK: usize = 32;

/// Decode one `e2m1` (FP4) nibble to f32. Bit 3 is the sign; bits 0-2 index the magnitude.
pub fn e2m1_decode(nibble: u8) -> f32 {
    // Magnitudes for E2M1: subnormal {0, .5}, then normal (1+M/2)·2^(E-1) for E=1,2,3.
    const MAG: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let sign = if nibble & 0x08 != 0 { -1.0 } else { 1.0 };
    sign * MAG[(nibble & 0x07) as usize]
}

/// Decode one `e4m3` (FP8) byte to f32. `0x7F`/`0xFF` are `NaN`; `E=0` is subnormal.
pub fn e4m3_decode(byte: u8) -> f32 {
    let sign = if byte & 0x80 != 0 { -1.0 } else { 1.0 };
    let exp = ((byte >> 3) & 0x0F) as i32;
    let mant = (byte & 0x07) as f32;
    if exp == 0 {
        sign * (mant / 8.0) * 2f32.powi(-6) // subnormal: (M/8)·2^-6
    } else if exp == 0x0F && mant == 7.0 {
        f32::NAN // S.1111.111 is the only NaN (fn variant: no infinities)
    } else {
        sign * (1.0 + mant / 8.0) * 2f32.powi(exp - 7) // normal: (1+M/8)·2^(E-7)
    }
}

/// Decode one `e8m0` (UE8M0) scale byte to f32: `2^(b-127)`, with `0xFF` → `NaN`.
pub fn e8m0_decode(byte: u8) -> f32 {
    if byte == 0xFF {
        f32::NAN
    } else {
        2f32.powi(byte as i32 - 127)
    }
}

/// Dequantize an FP8 (`e4m3`) weight `[rows, cols]` with an `e8m0` scale `[⌈rows/block⌉, ⌈cols/block⌉]`
/// (one scale per `block`×`block` tile) into an f32 `[rows, cols]` tensor.
pub fn fp8_weight_dequant(
    weight: &[u8],
    scale: &[u8],
    rows: usize,
    cols: usize,
    block: usize,
    dev: &Device,
) -> Result<Tensor> {
    let sc_cols = cols.div_ceil(block); // scale grid is [⌈rows/block⌉, ⌈cols/block⌉]
    let mut out = Vec::with_capacity(rows * cols);
    for i in 0..rows {
        for j in 0..cols {
            let w = e4m3_decode(weight[i * cols + j]);
            let s = e8m0_decode(scale[(i / block) * sc_cols + (j / block)]);
            out.push(w * s);
        }
    }
    Tensor::from_vec(out, (rows, cols), dev)
}

/// Dequantize an FP4 (`e2m1`) weight packed as `[rows, cols/2]` with an `e8m0` scale `[rows, cols/block]`
/// (per row, one scale per `block` elements) into an f32 `[rows, cols]` tensor.
pub fn fp4_weight_dequant(
    packed: &[u8],
    scale: &[u8],
    rows: usize,
    cols: usize,
    block: usize,
    dev: &Device,
) -> Result<Tensor> {
    let sc_cols = cols / block; // per-row scale grid [rows, cols/block]
    let packed_cols = cols / 2; // two fp4 nibbles per stored byte
    let mut out = Vec::with_capacity(rows * cols);
    for i in 0..rows {
        for j in 0..cols {
            let byte = packed[i * packed_cols + j / 2];
            let nibble = if j % 2 == 0 { byte & 0x0F } else { byte >> 4 }; // low = even col
            let w = e2m1_decode(nibble);
            let s = e8m0_decode(scale[i * sc_cols + j / block]);
            out.push(w * s);
        }
    }
    Tensor::from_vec(out, (rows, cols), dev)
}
