# deepseek-v4-candle

A from-scratch, educational **reference implementation of the DeepSeek-V4 architecture** in
**Rust + [candle](https://github.com/huggingface/candle)**, validated at toy scale against the
official Python reference.

> This is **not** a checkpoint runner or a fork of anyone's project — it is an *independent
> reimplementation* of V4's novel architectural components, written to understand them by rebuilding
> them from the math up. In the spirit of nanoGPT and llama.cpp.

## Why

DeepSeek-V4 introduces several components with no equivalent in `candle-transformers`, so each is
built here from scratch and pinned to a golden:

- **Manifold-Constrained Hyper-Connections (mHC)** — replaces the plain residual with `hc_mult`
  parallel streams, mixed per-token by a *doubly-stochastic* (Sinkhorn / Birkhoff-polytope) matrix.
- **Hybrid attention** — every layer is sliding-window, **HCA** (compressed KV), or **CSA** (learned
  sparse key selection), dispatched by a per-layer compression ratio, over an **MLA** latent-KV core
  with sink-softmax.
- **MoE** — a `sqrt(softplus)` routing gate with a *selection-only* bias, SwiGLU experts, and an
  always-on shared expert.
- **FP4 / FP8 / UE8M0** quantized weight dequant.
- **YaRN** rotary position embeddings.

## How it's validated

- **Strict TDD** — every unit gets a failing test first, watched fail for the right reason, then a
  minimal port from the reference.
- **Independent goldens** — expected values come from a pure-Python (`math`-only) reimplementation
  of the reference formula, *not* from this Rust code, so a green test is genuine cross-checking.
  (torch / CUDA are unavailable in the dev environment, which forces this independence.)
- **End-to-end parity** — the assembled forward pass (embed → mHC block → parallel head) matches a
  pure-Python golden to `max |Δ| ≈ 1e-5`, for both a 1-layer model and a 3-layer hybrid stack
  exercising all three attention regimes (sliding-window + HCA + CSA) with per-layer RoPE.
- **59 tests**, `clippy` clean.

## Quick start

```bash
cargo test                       # 59 tests (unit goldens + end-to-end parity)
cargo run --example toy_forward  # tiny end-to-end forward pass, prints next-token logits
```

## Module layout

| module | what |
| --- | --- |
| `mhc` | Sinkhorn doubly-stochastic hyper-connections (replaces the residual) |
| `rope` | YaRN rotary embeddings |
| `attention` | MLA core + sink-softmax + index-gathered `sparse_attn` |
| `sparse` | per-layer KV selection: sliding-window / HCA `Compressor` / CSA `Indexer` |
| `moe` | `sqrtsoftplus` gate, SwiGLU experts, shared expert |
| `block` | one decoder layer: mHC-wrapped attention + MoE |
| `model` | embedding + stacked blocks + parallel LM head (full forward) |
| `quant` | FP4 / FP8 / UE8M0 → f32 weight dequant |
| `loader` | `safetensors` mmap reader → typed weight load (FP8/FP4 via `quant`, BF16/F32 direct); `Transformer::from_config` assembles the model |

## Scope & limitations

An *architecture* reference, validated at toy scale — and honest about what it is and isn't:

- ✅ Numerically faithful forward pass for every novel component, parity-pinned per unit and
  end-to-end.
- ✅ **Checkpoint loader** — `Transformer::from_config` assembles the model straight from a converted
  `safetensors` checkpoint, reading every weight by the name `convert.py` emits and dequantizing the
  FP8/FP4 projections as it goes. Pinned on synthetic fixtures (one per dtype branch) and two toy
  converted checkpoints — a 1-layer sliding-window model and a 3-layer hybrid (window + HCA + CSA) —
  each reproducing its end-to-end golden, proof every name lands in the right field, down to the
  per-layer `Compressor` and CSA `Indexer` (and the indexer's own nested compressor). Hash-routing
  layers are the one deferred path — it errors *explicitly* there (`tid2eid` loading not yet wired) —
  so it does not yet assemble the full 284B model, which needs ~150–300 GB regardless, past dev hardware.
- ⛔ No tokenizer yet — inputs are raw token ids.
- ⛔ Prefill only — no decode-phase KV cache (`start_pos > 0`).
- ⛔ Single-rank — tensor-parallel sharding collapses to identity.
- ⛔ Weights are decoded to f32 and fed the standard f32 `linear`. The reference's fused
  `act_quant` + `fp8_gemm` / `fp4_gemm` (CUDA tilelang) are omitted: they quantize *activations* for
  speed and footprint, which only *loses* precision relative to f32 weights. The decoded weight
  values themselves are bit-exact.

## Attribution & license

The architecture and numerical formulas are DeepSeek-V4's, ported from its official reference
inference code. The Rust implementation here is independent and original work. Released under the
[MIT License](LICENSE).
