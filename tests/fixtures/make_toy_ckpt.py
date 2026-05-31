#!/usr/bin/env python3
"""Generate `toy_ckpt.safetensors` — a one-layer toy *converted* checkpoint for the `from_config` test.

Pure standard library (`struct`, `json`, `math`) — no torch. Two contracts are encoded here:

1. **Names** are exactly what `inference/convert.py` emits for a real V4 checkpoint (the inference
   model's attribute paths): `embed.weight`, `layers.0.attn.{wq_a,wq_b,wkv,wo_a,wo_b}.weight`,
   `layers.0.attn.{q_norm,kv_norm}.weight`, `layers.0.attn.attn_sink`,
   `layers.0.ffn.gate.{weight,bias}`, `layers.0.ffn.experts.{j}.{w1,w2,w3}.weight`,
   `layers.0.ffn.shared_experts.{w1,w2,w3}.weight`, `layers.0.{attn_norm,ffn_norm}.weight`,
   `layers.0.{hc_attn,hc_ffn}_{fn,base,scale}`, `norm.weight`, `head.weight`,
   `hc_head_{fn,base,scale}`. So `Transformer::from_config` (which reads these names) is exercised
   against the real naming, and a real converted checkpoint would load the same way.

2. **Values** are the same deterministic `det_at(o,i,start) = 0.3*sin(0.7*(start+k)+1)` (row-major)
   used by the Rust toy builders in `tests/common/mod.rs`. So a correctly-wired `from_config` must
   reproduce the *same* end-to-end golden the hand-built `toy_transformer` is pinned to.

Every weight is stored **F32** with NO `.scale` sibling: `from_config` then takes the unquantized
`auto_tensor` path for each tensor (the FP8/FP4 dequant paths are covered by the loader unit tests).
This keeps the E2E test a pure *plumbing* check — does each name land in the right field — with only
f32-rounding noise (Python computes `det_at` in f64, rounds once to f32). Regenerate with:

    python3 tests/fixtures/make_toy_ckpt.py
"""
import json
import math
import struct

OUT = __file__.rsplit("/", 1)[0] + "/toy_ckpt.safetensors"

# Toy dims — must match tests/common/mod.rs and tests/fixtures/toy_config.json.
DIM, HC, H, HD, RD = 8, 2, 2, 4, 2
QLR, G, OLR = 4, 2, 2
NROUTED, INTER, VOCAB = 4, 4, 6
MIXHC = (2 + HC) * HC  # 8


def f32(vals):
    return struct.pack("<%df" % len(vals), *vals)


def det(o, i, start):
    """Row-major [o, i] of 0.3*sin(0.7*(start+k)+1), packed F32 — mirrors `det_at` in common/mod.rs."""
    return f32([0.3 * math.sin(0.7 * (start + k) + 1.0) for k in range(o * i)])


def ones(n):
    return f32([1.0] * n)


# (name, shape, F32 bytes). Starts copy the `det_at(...)` starts in tests/common/mod.rs exactly.
TENSORS = [
    ("embed.weight", [VOCAB, DIM], det(VOCAB, DIM, 800)),
    # --- attention ---
    ("layers.0.attn.wq_a.weight", [QLR, DIM], det(QLR, DIM, 1)),
    ("layers.0.attn.q_norm.weight", [QLR], ones(QLR)),
    ("layers.0.attn.wq_b.weight", [H * HD, QLR], det(H * HD, QLR, 11)),
    ("layers.0.attn.wkv.weight", [HD, DIM], det(HD, DIM, 23)),
    ("layers.0.attn.kv_norm.weight", [HD], ones(HD)),
    ("layers.0.attn.wo_a.weight", [G * OLR, (H * HD) // G], det(G * OLR, (H * HD) // G, 31)),
    ("layers.0.attn.wo_b.weight", [DIM, G * OLR], det(DIM, G * OLR, 43)),
    ("layers.0.attn.attn_sink", [H], det(1, H, 7)),
    # --- block norms ---
    ("layers.0.attn_norm.weight", [DIM], ones(DIM)),
    ("layers.0.ffn_norm.weight", [DIM], ones(DIM)),
    # --- MoE gate ---
    ("layers.0.ffn.gate.weight", [NROUTED, DIM], det(NROUTED, DIM, 300)),
    ("layers.0.ffn.gate.bias", [NROUTED], det(1, NROUTED, 311)),
    # --- shared expert ---
    ("layers.0.ffn.shared_experts.w1.weight", [INTER, DIM], det(INTER, DIM, 400)),
    ("layers.0.ffn.shared_experts.w3.weight", [INTER, DIM], det(INTER, DIM, 420)),
    ("layers.0.ffn.shared_experts.w2.weight", [DIM, INTER], det(DIM, INTER, 440)),
    # --- mHC mixers (attn / ffn sites) ---
    ("layers.0.hc_attn_fn", [MIXHC, HC * DIM], det(MIXHC, HC * DIM, 600)),
    ("layers.0.hc_attn_base", [MIXHC], det(1, MIXHC, 697)),
    ("layers.0.hc_attn_scale", [3], ones(3)),
    ("layers.0.hc_ffn_fn", [MIXHC, HC * DIM], det(MIXHC, HC * DIM, 700)),
    ("layers.0.hc_ffn_base", [MIXHC], det(1, MIXHC, 797)),
    ("layers.0.hc_ffn_scale", [3], ones(3)),
    # --- final norm + head + head mHC ---
    ("norm.weight", [DIM], ones(DIM)),
    ("head.weight", [VOCAB, DIM], det(VOCAB, DIM, 500)),
    ("hc_head_fn", [HC, HC * DIM], det(HC, HC * DIM, 520)),
    ("hc_head_base", [HC], det(1, HC, 560)),
    ("hc_head_scale", [1], ones(1)),
]

# Routed experts j=0..3, starts o=100+j*50 (w1=o, w3=o+20, w2=o+40) — exactly common/mod.rs.
for j in range(NROUTED):
    o = 100 + j * 50
    TENSORS += [
        (f"layers.0.ffn.experts.{j}.w1.weight", [INTER, DIM], det(INTER, DIM, o)),
        (f"layers.0.ffn.experts.{j}.w3.weight", [INTER, DIM], det(INTER, DIM, o + 20)),
        (f"layers.0.ffn.experts.{j}.w2.weight", [DIM, INTER], det(DIM, INTER, o + 40)),
    ]


def main():
    header, blob = {}, bytearray()
    for name, shape, data in TENSORS:
        start = len(blob)
        header[name] = {"dtype": "F32", "shape": shape, "data_offsets": [start, start + len(data)]}
        blob += data

    hjson = json.dumps(header, separators=(",", ":")).encode("utf-8")
    hjson += b" " * ((-len(hjson)) % 8)  # pad header to an 8-byte boundary
    with open(OUT, "wb") as f:
        f.write(struct.pack("<Q", len(hjson)))
        f.write(hjson)
        f.write(blob)
    print("wrote", OUT, "(", 8 + len(hjson) + len(blob), "bytes,", len(TENSORS), "tensors )")


if __name__ == "__main__":
    main()
