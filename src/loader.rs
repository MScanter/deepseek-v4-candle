//! Load a converted DeepSeek-V4 checkpoint: a minimal safetensors reader (this module) plus typed
//! dequant that turns the stored FP8/FP4/BF16 tensors into f32 (built on [`crate::quant`]).
//!
//! ## Why hand-roll a safetensors reader
//! candle ships one, but a converted V4 checkpoint mixes dtypes candle has no `DType` for — the
//! `e8m0` block scales and packed `float4_e2m1fn_x2` experts arrive as raw bytes. So we parse the
//! (simple, stable) safetensors *container* ourselves, read each tensor's raw bytes, and decode by
//! the dtype the *architecture* dictates at each site (config `dtype`/`expert_dtype` plus the handful
//! of bf16 special-cases) rather than trusting a `DType` round-trip. This also makes the loader
//! robust to whether a given file labels packed FP4 as `F4_E2M1` or just `U8`. The file is `mmap`ed,
//! so a multi-hundred-GB checkpoint is paged lazily instead of read into RAM.
//!
//! ## safetensors layout
//! `[u64 LE header_len][header_len bytes of JSON][data blob]`. The JSON maps each name to
//! `{dtype, shape, data_offsets:[begin,end]}`, the offsets relative to the data blob. An optional
//! `__metadata__` string map is ignored.

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use candle_core::{Device, Result, Tensor};
use memmap2::Mmap;
use serde_json::Value;

use crate::quant::{bf16_decode, fp4_weight_dequant, fp8_weight_dequant, FP4_BLOCK, FP8_BLOCK};

fn err(msg: String) -> candle_core::Error {
    candle_core::Error::Msg(msg)
}

/// One tensor's location in the file: declared dtype + shape, and its absolute byte range in the mmap.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub dtype: String,
    pub shape: Vec<usize>,
    /// Absolute file offset of the first byte (`data_start + data_offsets[0]`).
    pub begin: usize,
    /// Absolute file offset one past the last byte.
    pub end: usize,
}

/// A memory-mapped safetensors file with its header parsed into [`TensorInfo`]s.
pub struct SafeTensors {
    mmap: Mmap,
    tensors: HashMap<String, TensorInfo>,
}

impl SafeTensors {
    /// `mmap` the file and parse its header.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path).map_err(|e| err(e.to_string()))?;
        // SAFETY: a read-only checkpoint. We treat the mapping as immutable bytes and never assume
        // the backing file is mutated underneath us — the standard safetensors-loading idiom.
        let mmap = unsafe { Mmap::map(&file).map_err(|e| err(e.to_string()))? };
        let tensors = parse_header(&mmap)?;
        Ok(Self { mmap, tensors })
    }

    /// All tensor names, sorted (for inspection and tests).
    pub fn tensor_names(&self) -> Vec<&str> {
        let mut v: Vec<&str> = self.tensors.keys().map(|s| s.as_str()).collect();
        v.sort_unstable();
        v
    }

    /// Metadata for `name`, if present.
    pub fn info(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }

    /// Raw little-endian bytes for `name`.
    pub fn raw(&self, name: &str) -> Result<&[u8]> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| err(format!("tensor not found: {name}")))?;
        Ok(&self.mmap[info.begin..info.end])
    }

    /// Read an `F32` tensor (little-endian) into a `[shape]` f32 tensor.
    pub fn f32_tensor(&self, name: &str, shape: &[usize], dev: &Device) -> Result<Tensor> {
        let bytes = self.raw(name)?;
        let n: usize = shape.iter().product();
        if bytes.len() != n * 4 {
            return Err(err(format!(
                "{name}: expected {} F32 bytes for shape {shape:?}, found {}",
                n * 4,
                bytes.len()
            )));
        }
        let vals: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        Tensor::from_vec(vals, shape.to_vec(), dev)
    }

    /// Read a `BF16` tensor (little-endian `u16`s) and widen each to f32 into a `[shape]` tensor.
    pub fn bf16_tensor(&self, name: &str, shape: &[usize], dev: &Device) -> Result<Tensor> {
        let bytes = self.raw(name)?;
        let n: usize = shape.iter().product();
        if bytes.len() != n * 2 {
            return Err(err(format!(
                "{name}: expected {} BF16 bytes for shape {shape:?}, found {}",
                n * 2,
                bytes.len()
            )));
        }
        let vals: Vec<f32> = bytes
            .chunks_exact(2)
            .map(|c| bf16_decode(u16::from_le_bytes([c[0], c[1]])))
            .collect();
        Tensor::from_vec(vals, shape.to_vec(), dev)
    }

    /// Read a non-quantized tensor by the dtype its header declares (`F32` -> [`Self::f32_tensor`],
    /// `BF16` -> [`Self::bf16_tensor`]). Anything else is an error: this path is only for tensors a
    /// converted V4 checkpoint stores unquantized — every norm/bias, `attn_sink`, the hc params,
    /// `embed`, `head`, and `wo_a`.
    pub fn auto_tensor(&self, name: &str, shape: &[usize], dev: &Device) -> Result<Tensor> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| err(format!("tensor not found: {name}")))?;
        match info.dtype.as_str() {
            "F32" => self.f32_tensor(name, shape, dev),
            "BF16" => self.bf16_tensor(name, shape, dev),
            other => Err(err(format!(
                "{name}: auto_tensor handles F32/BF16, not {other} (quantized weights go through fp8_tensor/fp4_tensor)"
            ))),
        }
    }

    /// Dequant an FP8 (`e4m3`) linear stored as `{prefix}.weight` + `{prefix}.scale` (`e8m0`, one per
    /// `block`x`block` tile) into an f32 `[rows, cols]` tensor.
    pub fn fp8_tensor(
        &self,
        prefix: &str,
        rows: usize,
        cols: usize,
        block: usize,
        dev: &Device,
    ) -> Result<Tensor> {
        let weight = self.raw(&format!("{prefix}.weight"))?;
        let scale = self.raw(&format!("{prefix}.scale"))?;
        let want_scale = rows.div_ceil(block) * cols.div_ceil(block);
        if weight.len() != rows * cols || scale.len() != want_scale {
            return Err(err(format!(
                "{prefix}: FP8 [{rows},{cols}] block {block} wants {} weight + {want_scale} scale bytes, found {} + {}",
                rows * cols,
                weight.len(),
                scale.len()
            )));
        }
        fp8_weight_dequant(weight, scale, rows, cols, block, dev)
    }

    /// Dequant an FP4 (`e2m1`) expert stored as `{prefix}.weight` (packed, 2 nibbles/byte) +
    /// `{prefix}.scale` (`e8m0`, one per `block` along the input dim) into an f32 `[rows, cols]` tensor.
    pub fn fp4_tensor(
        &self,
        prefix: &str,
        rows: usize,
        cols: usize,
        block: usize,
        dev: &Device,
    ) -> Result<Tensor> {
        let weight = self.raw(&format!("{prefix}.weight"))?;
        let scale = self.raw(&format!("{prefix}.scale"))?;
        let want_scale = rows * (cols / block);
        if weight.len() != rows * cols / 2 || scale.len() != want_scale {
            return Err(err(format!(
                "{prefix}: FP4 [{rows},{cols}] block {block} wants {} packed + {want_scale} scale bytes, found {} + {}",
                rows * cols / 2,
                weight.len(),
                scale.len()
            )));
        }
        fp4_weight_dequant(weight, scale, rows, cols, block, dev)
    }

    /// Load one projection weight `[rows, cols]` to f32, dispatching on how it's stored: a
    /// `{prefix}.scale` sibling marks a quantized weight (FP4 when `fp4`, else FP8, each at its
    /// fixed block size); its absence means an unquantized `{prefix}.weight` read by header dtype
    /// (e.g. the pre-dequantized bf16 `wo_a`). This is the single entry point `from_config` uses
    /// for every linear, so the same call site handles fp8 / fp4 / bf16 / f32 storage uniformly.
    pub fn linear(&self, prefix: &str, rows: usize, cols: usize, fp4: bool, dev: &Device) -> Result<Tensor> {
        if self.info(&format!("{prefix}.scale")).is_some() {
            if fp4 {
                self.fp4_tensor(prefix, rows, cols, FP4_BLOCK, dev)
            } else {
                self.fp8_tensor(prefix, rows, cols, FP8_BLOCK, dev)
            }
        } else {
            self.auto_tensor(&format!("{prefix}.weight"), &[rows, cols], dev)
        }
    }
}

/// Parse the 8-byte length-prefixed JSON header into absolute-offset [`TensorInfo`]s.
fn parse_header(buf: &[u8]) -> Result<HashMap<String, TensorInfo>> {
    if buf.len() < 8 {
        return Err(err("file too small for a safetensors header".into()));
    }
    let header_len = u64::from_le_bytes(buf[0..8].try_into().unwrap()) as usize;
    let data_start = 8 + header_len;
    if buf.len() < data_start {
        return Err(err("header length exceeds file size".into()));
    }
    let json: Value = serde_json::from_slice(&buf[8..data_start]).map_err(|e| err(e.to_string()))?;
    let obj = json
        .as_object()
        .ok_or_else(|| err("header is not a JSON object".into()))?;

    let mut tensors = HashMap::new();
    for (name, v) in obj {
        if name == "__metadata__" {
            continue; // optional free-form string map, not a tensor
        }
        let dtype = v
            .get("dtype")
            .and_then(Value::as_str)
            .ok_or_else(|| err(format!("{name}: missing dtype")))?
            .to_string();
        let shape = v
            .get("shape")
            .and_then(Value::as_array)
            .ok_or_else(|| err(format!("{name}: missing shape")))?
            .iter()
            .map(|d| {
                d.as_u64()
                    .map(|x| x as usize)
                    .ok_or_else(|| err(format!("{name}: non-integer shape dim")))
            })
            .collect::<Result<Vec<usize>>>()?;
        let offs = v
            .get("data_offsets")
            .and_then(Value::as_array)
            .ok_or_else(|| err(format!("{name}: missing data_offsets")))?;
        let [rb, re] = offs.as_slice() else {
            return Err(err(format!("{name}: data_offsets must have 2 entries")));
        };
        let begin = data_start
            + rb.as_u64()
                .ok_or_else(|| err(format!("{name}: bad data_offset")))? as usize;
        let end = data_start
            + re.as_u64()
                .ok_or_else(|| err(format!("{name}: bad data_offset")))? as usize;
        if begin > end || end > buf.len() {
            return Err(err(format!("{name}: data_offsets out of range")));
        }
        tensors.insert(name.clone(), TensorInfo { dtype, shape, begin, end });
    }
    Ok(tensors)
}
