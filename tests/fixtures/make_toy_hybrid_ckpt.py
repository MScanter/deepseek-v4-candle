#!/usr/bin/env python3
"""Generate `toy_hybrid_ckpt.safetensors` — the 3-layer HYBRID toy *converted* checkpoint for the
`from_config_hybrid_matches_reference` test (sliding-window L0 + HCA L1 + CSA L2).

Pure standard library (`struct`, `json`, `math`) — no torch. Same two contracts as `make_toy_ckpt.py`,
extended to the compressed-attention layers:

1. **Names** are exactly what `inference/convert.py` emits. Beyond the sliding-window names (see
   `make_toy_ckpt.py`), the HCA/CSA layers add the compressor and (CSA only) the learned indexer,
   whose *module attribute paths* convert.py preserves verbatim (it only renames the leaf `key`, and
   none of these leaves are in its `mapping`, except `wq_b`/`weights_proj` which map to themselves):
       layers.{l}.attn.compressor.{wkv,wgate}.weight
       layers.{l}.attn.compressor.ape                         # nn.Parameter -> NO `.weight`
       layers.{l}.attn.compressor.norm.weight
       layers.{l}.attn.indexer.{wq_b,weights_proj}.weight
       layers.{l}.attn.indexer.compressor.{wkv,wgate}.weight
       layers.{l}.attn.indexer.compressor.ape                 # NO `.weight`
       layers.{l}.attn.indexer.compressor.norm.weight
   (`ape` lands in convert.py's "no `.weight`" branch via the `"ape" in name` test, exactly like the
   `hc`/`attn_sink` params.) So `from_config` is exercised against the real hybrid naming.

2. **Values** are the deterministic `det(o,i,start) = 0.3*sin(0.7*(start+k)+1)` of the Rust hybrid
   builders in `tests/common/mod.rs` (module `hybrid`) and `hybrid_golden.py`, with the SAME per-layer
   offset `L = layer*1000`. So a correctly-wired `from_config` reproduces the SAME end-to-end golden
   the hand-built `hybrid::toy_transformer` is pinned to.

Every weight is stored **F32** with NO `.scale` sibling, so `from_config` takes the unquantized
`auto_tensor` path (FP8/FP4 dequant is covered by the loader unit tests). Regenerate with:

    python3 tests/fixtures/make_toy_hybrid_ckpt.py
"""
import json
import math
import struct

OUT = __file__.rsplit("/", 1)[0] + "/toy_hybrid_ckpt.safetensors"

# Toy dims — must match tests/common/mod.rs `hybrid` module + toy_hybrid_config.json + hybrid_golden.py.
DIM, HC, H, HD = 8, 2, 2, 8  # NB: HD = 8 here (not 4), so the per-layer rope theta is observable.
QLR, G, OLR = 4, 2, 2
NROUTED, INTER, VOCAB = 4, 4, 6
MIXHC = (2 + HC) * HC  # 8
IDXH, IDXHD = 1, 8  # indexer heads / head_dim
COMPRESS_RATIOS = [0, 2, 4]


def f32(vals):
    return struct.pack("<%df" % len(vals), *vals)


def det(o, i, start):
    """Row-major [o, i] of 0.3*sin(0.7*(start+k)+1), packed F32 — mirrors `det_at` in common/mod.rs."""
    return f32([0.3 * math.sin(0.7 * (start + k) + 1.0) for k in range(o * i)])


def ones(n):
    return f32([1.0] * n)


def layer_tensors(layer):
    """Every checkpoint tensor for one decoder layer (offset L = layer*1000), keyed by convert.py name."""
    L = layer * 1000
    ratio = COMPRESS_RATIOS[layer]
    p = f"layers.{layer}"
    ts = [
        # --- attention (sliding-window core, present in every layer) ---
        (f"{p}.attn.wq_a.weight", [QLR, DIM], det(QLR, DIM, L + 1)),
        (f"{p}.attn.q_norm.weight", [QLR], ones(QLR)),
        (f"{p}.attn.wq_b.weight", [H * HD, QLR], det(H * HD, QLR, L + 11)),
        (f"{p}.attn.wkv.weight", [HD, DIM], det(HD, DIM, L + 23)),
        (f"{p}.attn.kv_norm.weight", [HD], ones(HD)),
        (f"{p}.attn.wo_a.weight", [G * OLR, (H * HD) // G], det(G * OLR, (H * HD) // G, L + 31)),
        (f"{p}.attn.wo_b.weight", [DIM, G * OLR], det(DIM, G * OLR, L + 43)),
        (f"{p}.attn.attn_sink", [H], det(1, H, L + 7)),
        # --- block norms ---
        (f"{p}.attn_norm.weight", [DIM], ones(DIM)),
        (f"{p}.ffn_norm.weight", [DIM], ones(DIM)),
        # --- MoE gate + shared expert ---
        (f"{p}.ffn.gate.weight", [NROUTED, DIM], det(NROUTED, DIM, L + 300)),
        (f"{p}.ffn.gate.bias", [NROUTED], det(1, NROUTED, L + 311)),
        (f"{p}.ffn.shared_experts.w1.weight", [INTER, DIM], det(INTER, DIM, L + 400)),
        (f"{p}.ffn.shared_experts.w3.weight", [INTER, DIM], det(INTER, DIM, L + 420)),
        (f"{p}.ffn.shared_experts.w2.weight", [DIM, INTER], det(DIM, INTER, L + 440)),
        # --- mHC mixers (attn / ffn sites) ---
        (f"{p}.hc_attn_fn", [MIXHC, HC * DIM], det(MIXHC, HC * DIM, L + 600)),
        (f"{p}.hc_attn_base", [MIXHC], det(1, MIXHC, L + 697)),
        (f"{p}.hc_attn_scale", [3], ones(3)),
        (f"{p}.hc_ffn_fn", [MIXHC, HC * DIM], det(MIXHC, HC * DIM, L + 700)),
        (f"{p}.hc_ffn_base", [MIXHC], det(1, MIXHC, L + 797)),
        (f"{p}.hc_ffn_scale", [3], ones(3)),
    ]
    # Routed experts j=0..3, starts o=L+100+j*50 (w1=o, w3=o+20, w2=o+40) — exactly common/mod.rs.
    for j in range(NROUTED):
        o = L + 100 + j * 50
        ts += [
            (f"{p}.ffn.experts.{j}.w1.weight", [INTER, DIM], det(INTER, DIM, o)),
            (f"{p}.ffn.experts.{j}.w3.weight", [INTER, DIM], det(INTER, DIM, o + 20)),
            (f"{p}.ffn.experts.{j}.w2.weight", [DIM, INTER], det(DIM, INTER, o + 40)),
        ]
    # HCA/CSA compressor (coff = 2 overlap for CSA ratio 4, else 1).
    if ratio > 0:
        coff = 2 if ratio == 4 else 1
        ts += [
            (f"{p}.attn.compressor.wkv.weight", [coff * HD, DIM], det(coff * HD, DIM, L + 50)),
            (f"{p}.attn.compressor.wgate.weight", [coff * HD, DIM], det(coff * HD, DIM, L + 60)),
            (f"{p}.attn.compressor.ape", [ratio, coff * HD], det(ratio, coff * HD, L + 70)),
            (f"{p}.attn.compressor.norm.weight", [HD], ones(HD)),
        ]
    # CSA learned indexer (+ its own overlapping compressor at IDXHD).
    if ratio == 4:
        ts += [
            (f"{p}.attn.indexer.wq_b.weight", [IDXH * IDXHD, QLR], det(IDXH * IDXHD, QLR, L + 80)),
            (f"{p}.attn.indexer.weights_proj.weight", [IDXH, DIM], det(IDXH, DIM, L + 85)),
            (f"{p}.attn.indexer.compressor.wkv.weight", [2 * IDXHD, DIM], det(2 * IDXHD, DIM, L + 90)),
            (f"{p}.attn.indexer.compressor.wgate.weight", [2 * IDXHD, DIM], det(2 * IDXHD, DIM, L + 110)),
            (f"{p}.attn.indexer.compressor.ape", [4, 2 * IDXHD], det(4, 2 * IDXHD, L + 130)),
            (f"{p}.attn.indexer.compressor.norm.weight", [IDXHD], ones(IDXHD)),
        ]
    return ts


def main():
    tensors = []
    for layer in range(len(COMPRESS_RATIOS)):
        tensors += layer_tensors(layer)
    # Global (no layer offset): embedding, final norm, head, head mHC.
    tensors += [
        ("embed.weight", [VOCAB, DIM], det(VOCAB, DIM, 800)),
        ("norm.weight", [DIM], ones(DIM)),
        ("head.weight", [VOCAB, DIM], det(VOCAB, DIM, 500)),
        ("hc_head_fn", [HC, HC * DIM], det(HC, HC * DIM, 520)),
        ("hc_head_base", [HC], det(1, HC, 560)),
        ("hc_head_scale", [1], ones(1)),
    ]

    header, blob = {}, bytearray()
    for name, shape, data in tensors:
        start = len(blob)
        header[name] = {"dtype": "F32", "shape": shape, "data_offsets": [start, start + len(data)]}
        blob += data

    hjson = json.dumps(header, separators=(",", ":")).encode("utf-8")
    hjson += b" " * ((-len(hjson)) % 8)  # pad header to an 8-byte boundary
    with open(OUT, "wb") as f:
        f.write(struct.pack("<Q", len(hjson)))
        f.write(hjson)
        f.write(blob)
    print("wrote", OUT, "(", 8 + len(hjson) + len(blob), "bytes,", len(tensors), "tensors )")


if __name__ == "__main__":
    main()
