//! FP4/FP8 weight dequant: bit-exact decoder goldens + block-scaled dequant.
//!
//! These are spec goldens, not parity-vs-torch: torch isn't available locally, and the formats are
//! fully specified by the OCP/NVIDIA bit layouts, so every expected value is hand-derived from
//! `(-1)^S·(1+M/2^m)·2^(E-bias)` (see `src/quant.rs`). The dequant cases use a tiny `block` so a
//! couple of rows/cols already exercise multi-tile scale indexing and (for FP4) nibble unpacking.

use candle_core::Device;
use deepseek_v4_candle::quant::{
    e2m1_decode, e4m3_decode, e8m0_decode, fp4_weight_dequant, fp8_weight_dequant,
};

/// `e2m1` — all 16 nibbles. Magnitudes `{0,.5,1,1.5,2,3,4,6}` for bits 0-2; bit 3 is the sign.
#[test]
fn e2m1_decodes_full_table() {
    let mag = [0.0f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    for n in 0u8..8 {
        assert_eq!(e2m1_decode(n), mag[n as usize], "nibble {n:#04x}");
        assert_eq!(e2m1_decode(n | 0x08), -mag[n as usize], "nibble {:#04x}", n | 0x08);
    }
}

/// `e4m3` (fn variant): bias 7, subnormals at `E=0`, `NaN` only at `0x7F`/`0xFF`, max finite 448.
#[test]
fn e4m3_decodes_known_bytes() {
    assert_eq!(e4m3_decode(0x00), 0.0); //  0 0000 000
    assert_eq!(e4m3_decode(0x38), 1.0); //  0 0111 000 → (1+0)·2^0
    assert_eq!(e4m3_decode(0x40), 2.0); //  0 1000 000 → (1+0)·2^1
    assert_eq!(e4m3_decode(0x44), 3.0); //  0 1000 100 → (1+1/2)·2^1
    assert_eq!(e4m3_decode(0x7E), 448.0); // 0 1111 110 → (1+3/4)·2^8  (max finite)
    assert_eq!(e4m3_decode(0xC0), -2.0); // 1 1000 000
    assert_eq!(e4m3_decode(0xB8), -1.0); // 1 0111 000
    assert_eq!(e4m3_decode(0x01), 0.001953125); // 0 0000 001 → (1/8)·2^-6 = 2^-9 (subnormal)
    assert!(e4m3_decode(0x7F).is_nan()); // 0 1111 111
    assert!(e4m3_decode(0xFF).is_nan()); // 1 1111 111
}

/// `e8m0` (UE8M0) scale: unsigned exponent, value `2^(b-127)`, `NaN` at `0xFF`.
#[test]
fn e8m0_decodes_powers_of_two() {
    assert_eq!(e8m0_decode(0x7F), 1.0); // 2^0
    assert_eq!(e8m0_decode(0x80), 2.0); // 2^1
    assert_eq!(e8m0_decode(0x81), 4.0); // 2^2
    assert_eq!(e8m0_decode(0x7E), 0.5); // 2^-1
    assert!(e8m0_decode(0xFF).is_nan());
}

/// FP8 dequant, 2×4 with `block=2`: two column tiles get different scales (×2 then ×1); both rows
/// share the single scale row. `weight[i][j] = e4m3 · e8m0(scale[i/2][j/2])`.
#[test]
fn fp8_weight_dequant_blocks() -> candle_core::Result<()> {
    let dev = Device::Cpu;
    // row0 → [1,2,3,4], row1 → [1,1,2,2] (e4m3 bytes)
    let weight = [0x38, 0x40, 0x44, 0x48, 0x38, 0x38, 0x40, 0x40];
    let scale = [0x80, 0x7F]; // [1,2]: cols0-1 ×2, cols2-3 ×1
    let got = fp8_weight_dequant(&weight, &scale, 2, 4, 2, &dev)?;
    assert_eq!(got.dims(), &[2, 4]);
    let v = got.flatten_all()?.to_vec1::<f32>()?;
    assert_eq!(v, vec![2.0, 4.0, 3.0, 4.0, 2.0, 2.0, 2.0, 2.0]);
    Ok(())
}

/// FP8 scale shape uses **ceiling** division: 3 rows with `block=2` → 2 scale rows (rows 0-1 share
/// the first, row 2 the second). Pins `⌈rows/block⌉` / `⌈cols/block⌉` indexing.
#[test]
fn fp8_dequant_ceils_block_count() -> candle_core::Result<()> {
    let dev = Device::Cpu;
    // 3×2: row0 [1,2], row1 [1,1], row2 [2,2]
    let weight = [0x38, 0x40, 0x38, 0x38, 0x40, 0x40];
    let scale = [0x80, 0x7F]; // [2,1]: rows0-1 ×2, row2 ×1
    let got = fp8_weight_dequant(&weight, &scale, 3, 2, 2, &dev)?;
    let v = got.flatten_all()?.to_vec1::<f32>()?;
    assert_eq!(v, vec![2.0, 4.0, 2.0, 2.0, 2.0, 2.0]);
    Ok(())
}

/// FP4 dequant, 2×4 with `block=2`: unpack two nibbles per byte (low = even col), then apply the
/// per-row `[rows, cols/block]` scale. Exercises packing, sign nibbles, and per-row block scaling.
#[test]
fn fp4_weight_dequant_blocks() -> candle_core::Result<()> {
    let dev = Device::Cpu;
    // logical fp4: row0 [0.5,1,2,3] (nibbles 1,2,4,5), row1 [1.5,-0.5,6,-1] (nibbles 3,9,7,A)
    // packed low|high: row0 [0x21,0x54], row1 [0x93,0xA7]
    let packed = [0x21, 0x54, 0x93, 0xA7];
    // scale [2,2]: row0 (×2,×1), row1 (×1,×4)
    let scale = [0x80, 0x7F, 0x7F, 0x81];
    let got = fp4_weight_dequant(&packed, &scale, 2, 4, 2, &dev)?;
    assert_eq!(got.dims(), &[2, 4]);
    let v = got.flatten_all()?.to_vec1::<f32>()?;
    assert_eq!(v, vec![1.0, 2.0, 2.0, 3.0, 1.5, -0.5, 24.0, -4.0]);
    Ok(())
}
