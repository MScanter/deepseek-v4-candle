//! DeepSeek-V4 architecture in Rust + candle (reference implementation).
//!
//! Status: work in progress. The novel pieces of V4 — the CSA/HCA hybrid
//! attention, the manifold-constrained Hyper-Connections (mHC), and FP4/FP8
//! weight handling — have no equivalent in `candle-transformers`, so they are
//! built here from scratch and validated against the official reference
//! (`inference/model.py` + `inference/kernel.py`) at toy scale.
//!
//! Module map (mirrors the official inference code):
//! - [`config`]  — hyper-parameters, deserialized from the official `config.json`.
//! - [`mhc`]     — Manifold-Constrained Hyper-Connections (replaces the residual).
//! - [`rope`]    — YaRN rotary position embeddings.
//! - [`attention`] — sink-softmax attention (dense MLA core + index-gathered `sparse_attn`).
//! - [`sparse`]   — per-query KV-selection (sliding window, HCA `Compressor`, CSA `Indexer`).
//! - [`moe`]      — Mixture-of-Experts gate (`sqrtsoftplus` routing), SwiGLU experts, combine.
//! - [`block`]    — one decoder layer: mHC-wrapped attention + MoE.
//! - [`model`]    — embedding + stacked blocks + parallel LM head (full forward).
//! - [`quant`]    — FP4/FP8/BF16 → f32 weight dequant.
//! - [`loader`]   — minimal safetensors reader + typed dequant to load a converted checkpoint.

pub mod attention;
pub mod block;
pub mod config;
pub mod loader;
pub mod mhc;
pub mod model;
pub mod moe;
pub mod quant;
pub mod rope;
pub mod sparse;

pub use config::Config;
