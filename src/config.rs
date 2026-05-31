//! Model hyper-parameters.
//!
//! Field names match the official `inference/config.json` (the `ModelArgs`
//! dataclass in `inference/model.py`) so the JSON deserializes directly.

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub dim: usize,
    pub moe_inter_dim: usize,
    pub n_layers: usize,
    #[serde(default)]
    pub n_hash_layers: usize,
    #[serde(default = "default_mtp_layers")]
    pub n_mtp_layers: usize,
    pub n_heads: usize,

    // MoE
    pub n_routed_experts: usize,
    pub n_shared_experts: usize,
    pub n_activated_experts: usize,
    pub score_func: String,
    pub route_scale: f64,
    pub swiglu_limit: f64,

    // MLA
    pub q_lora_rank: usize,
    pub head_dim: usize,
    pub rope_head_dim: usize,
    pub o_groups: usize,
    pub o_lora_rank: usize,
    pub window_size: usize,
    /// Maximum sequence length — sizes the RoPE precompute table. The reference keeps this as a
    /// `ModelArgs` field (default 4096) rather than in `config.json`, so it defaults here too.
    #[serde(default = "default_max_seq_len")]
    pub max_seq_len: usize,

    // YaRN rope
    pub original_seq_len: usize,
    pub rope_theta: f64,
    pub rope_factor: f64,
    pub beta_fast: f64,
    pub beta_slow: f64,
    pub compress_rope_theta: f64,

    // CSA lightning indexer
    pub index_n_heads: usize,
    pub index_head_dim: usize,
    pub index_topk: usize,

    // Hyper-Connections (mHC)
    pub hc_mult: usize,
    pub hc_sinkhorn_iters: usize,
    #[serde(default = "default_eps")]
    pub hc_eps: f64,

    #[serde(default = "default_eps")]
    pub norm_eps: f64,

    /// Per-layer KV compression ratio, selecting the attention variant for each layer:
    /// `0` → sliding-window, `4` → Compressed Sparse Attention (CSA, with the learned indexer),
    /// otherwise (e.g. `128`) → Heavily Compressed Attention (HCA). Mirrors the reference's
    /// `if compress_ratio:` / `== 4` dispatch in `Attention.__init__`.
    pub compress_ratios: Vec<usize>,

    // Quantization (informational at the config level)
    #[serde(default)]
    pub dtype: Option<String>,
    #[serde(default)]
    pub scale_fmt: Option<String>,
    #[serde(default)]
    pub expert_dtype: Option<String>,
}

fn default_mtp_layers() -> usize {
    1
}
fn default_max_seq_len() -> usize {
    4096
}
fn default_eps() -> f64 {
    1e-6
}

impl Config {
    /// Width of the per-token mixing vector produced for mHC: `(2 + hc) * hc`
    /// (`hc` pre-weights + `hc` post-weights + an `hc × hc` combination matrix).
    pub fn mix_hc(&self) -> usize {
        (2 + self.hc_mult) * self.hc_mult
    }
}
