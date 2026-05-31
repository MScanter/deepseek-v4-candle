//! Loader tests against `tests/fixtures/sample.safetensors`, a real safetensors file built by the
//! pure-Python (no-torch) encoder in `tests/fixtures/make_sample.py`. Python encodes f32 -> bits, the
//! loader decodes bits -> f32, so a green test is genuine cross-validation.

use candle_core::Device;
use deepseek_v4_candle::loader::SafeTensors;

const FIXTURE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/sample.safetensors");

fn flat(t: &candle_core::Tensor) -> Vec<f32> {
    t.flatten_all().unwrap().to_vec1::<f32>().unwrap()
}

/// dtype + shape are parsed out of the JSON header for each named tensor.
#[test]
fn reads_tensor_metadata() {
    let st = SafeTensors::load(FIXTURE).expect("load fixture");
    assert_eq!(
        st.info("w_f32").map(|i| (i.dtype.as_str(), i.shape.clone())),
        Some(("F32", vec![2, 3]))
    );
    assert_eq!(
        st.info("lin.weight").map(|i| (i.dtype.as_str(), i.shape.clone())),
        Some(("F8_E4M3", vec![2, 4]))
    );
    assert_eq!(st.info("w_bf16").map(|i| i.shape.clone()), Some(vec![4]));
    assert_eq!(st.info("missing").map(|i| i.shape.clone()), None);
}

/// Every tensor in the header is listed (and `__metadata__`, absent here, would be skipped).
#[test]
fn lists_all_tensor_names() {
    let st = SafeTensors::load(FIXTURE).expect("load fixture");
    assert_eq!(
        st.tensor_names(),
        vec![
            "exp.scale", "exp.weight", "exp32.scale", "exp32.weight", "lin.scale", "lin.weight",
            "w_bf16", "w_f32", "wo.weight"
        ]
    );
}

/// `data_offsets` slice the right raw bytes out of the data blob (after the 8 + header_len prefix).
#[test]
fn reads_raw_bytes_by_offset() {
    let st = SafeTensors::load(FIXTURE).expect("load fixture");
    assert_eq!(st.raw("lin.scale").ok(), Some(&[0x80u8][..]));
    assert_eq!(st.raw("exp.weight").ok(), Some(&[0x21u8, 0x54, 0x93, 0xA7][..]));
    assert_eq!(
        st.raw("lin.weight").ok(),
        Some(&[0x38u8, 0x40, 0x44, 0x48, 0x38, 0x38, 0x40, 0x40][..])
    );
    assert!(st.raw("missing").is_err());
}

/// `F32` bytes -> f32 tensor, shape preserved.
#[test]
fn dequants_f32_tensor() {
    let st = SafeTensors::load(FIXTURE).expect("load fixture");
    let t = st.f32_tensor("w_f32", &[2, 3], &Device::Cpu).unwrap();
    assert_eq!(t.dims(), &[2, 3]);
    assert_eq!(flat(&t), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

/// `BF16` bytes widened to f32.
#[test]
fn dequants_bf16_tensor() {
    let st = SafeTensors::load(FIXTURE).expect("load fixture");
    let t = st.bf16_tensor("w_bf16", &[4], &Device::Cpu).unwrap();
    assert_eq!(flat(&t), vec![1.0, 2.0, 0.5, -1.0]);
}

/// FP8 `lin`: weight rows [1,2,3,4]/[1,1,2,2] x one e8m0 scale 0x80 (=2) -> everything doubled.
#[test]
fn dequants_fp8_weight_with_scale() {
    let st = SafeTensors::load(FIXTURE).expect("load fixture");
    let t = st.fp8_tensor("lin", 2, 4, 128, &Device::Cpu).unwrap();
    assert_eq!(t.dims(), &[2, 4]);
    assert_eq!(flat(&t), vec![2.0, 4.0, 6.0, 8.0, 2.0, 2.0, 4.0, 4.0]);
}

/// FP4 `exp`: packed -> logical [0.5,1,2,3]/[1.5,-0.5,6,-1], block=2 e8m0 scales (x2,x1)/(x1,x4).
#[test]
fn dequants_fp4_expert_with_scale() {
    let st = SafeTensors::load(FIXTURE).expect("load fixture");
    let t = st.fp4_tensor("exp", 2, 4, 2, &Device::Cpu).unwrap();
    assert_eq!(t.dims(), &[2, 4]);
    assert_eq!(flat(&t), vec![1.0, 2.0, 2.0, 3.0, 1.5, -0.5, 24.0, -4.0]);
}

/// `auto_tensor` reads a non-quantized tensor by the dtype its header declares: `F32` -> the f32
/// path, `BF16` -> the bf16 path. This is how `from_config` loads norms / biases / embed / head /
/// hc params (and `wo_a`), each stored as whatever the converted checkpoint actually wrote.
#[test]
fn auto_tensor_dispatches_on_header_dtype() {
    let st = SafeTensors::load(FIXTURE).expect("load fixture");
    let f = st.auto_tensor("w_f32", &[2, 3], &Device::Cpu).unwrap();
    assert_eq!(flat(&f), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = st.auto_tensor("w_bf16", &[4], &Device::Cpu).unwrap();
    assert_eq!(flat(&b), vec![1.0, 2.0, 0.5, -1.0]);
}

/// `linear` is the dispatcher `from_config` uses per projection: a `{prefix}.scale` sibling means a
/// quantized weight (FP8, or FP4 when `fp4`), its absence means an unquantized `{prefix}.weight`
/// read by header dtype. All three branches against the one fixture:
/// - `lin` has a scale, `fp4=false` -> FP8 path (block 128), the doubled row values.
/// - `exp32` has a scale, `fp4=true`  -> FP4 path (block 32), all-ones.
/// - `wo` has NO scale -> `auto_tensor` (its BF16 bytes), like the pre-dequantized `wo_a`.
#[test]
fn linear_dispatches_on_scale_presence() {
    let st = SafeTensors::load(FIXTURE).expect("load fixture");
    let fp8 = st.linear("lin", 2, 4, false, &Device::Cpu).unwrap();
    assert_eq!(flat(&fp8), vec![2.0, 4.0, 6.0, 8.0, 2.0, 2.0, 4.0, 4.0]);

    let fp4 = st.linear("exp32", 2, 32, true, &Device::Cpu).unwrap();
    assert_eq!(fp4.dims(), &[2, 32]);
    assert_eq!(flat(&fp4), vec![1.0; 64]);

    let auto = st.linear("wo", 2, 2, false, &Device::Cpu).unwrap();
    assert_eq!(flat(&auto), vec![1.0, 2.0, 0.5, -1.0]);
}
