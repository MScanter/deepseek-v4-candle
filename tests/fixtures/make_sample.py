#!/usr/bin/env python3
"""Generate `sample.safetensors` — a tiny fixture for the Rust loader tests.

Pure standard library (`struct`, `json`) — **no torch, no numpy, no safetensors**. This is the
*independent encoder* (f32 -> bits) that the Rust loader (bits -> f32) is cross-checked against, so a
green loader test is genuine cross-validation, not a tautology. Regenerate with:

    python3 tests/fixtures/make_sample.py

The safetensors binary layout we emit (and the Rust side parses):
    [8 bytes: little-endian u64 = N, the JSON header length]
    [N bytes: UTF-8 JSON header: {name: {dtype, shape, data_offsets:[start,end]}, ...}]
    [the raw tensor data buffer; each tensor occupies data[start:end]]
`data_offsets` are relative to the start of the data buffer (i.e. byte 8 + N of the file).

The tensors below mirror the dtypes a converted V4 checkpoint carries — F32, BF16, FP8 (e4m3) weight
with a U8 (e8m0) scale, and packed FP4 (e2m1) weight with its U8 scale — at toy dims, with bytes
hand-chosen to match the goldens already pinned in `tests/quant.rs`.
"""
import json
import struct

OUT = __file__.rsplit("/", 1)[0] + "/sample.safetensors"


def f32_bytes(vals):
    return struct.pack("<%df" % len(vals), *vals)


def bf16_bytes(vals):
    # bf16 is the high 16 bits of an IEEE-754 f32. The chosen values are all exactly representable
    # in bf16 (their low 16 f32 bits are zero), so truncation == round-to-nearest here.
    out = bytearray()
    for v in vals:
        bits = struct.unpack("<I", struct.pack("<f", v))[0]
        out += struct.pack("<H", (bits >> 16) & 0xFFFF)
    return bytes(out)


# (name, dtype string, shape, raw little-endian bytes)
TENSORS = [
    ("w_f32", "F32", [2, 3], f32_bytes([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])),
    ("w_bf16", "BF16", [4], bf16_bytes([1.0, 2.0, 0.5, -1.0])),
    # FP8 weight [2,4] (e4m3): row0 [1,2,3,4], row1 [1,1,2,2]; one e8m0 scale 0x80 (=2.0).
    ("lin.weight", "F8_E4M3", [2, 4], bytes([0x38, 0x40, 0x44, 0x48, 0x38, 0x38, 0x40, 0x40])),
    ("lin.scale", "U8", [1, 1], bytes([0x80])),
    # FP4 weight packed [2,2] -> logical [2,4]: row0 [0.5,1,2,3], row1 [1.5,-0.5,6,-1].
    ("exp.weight", "U8", [2, 2], bytes([0x21, 0x54, 0x93, 0xA7])),
    # FP4 e8m0 scales, block=2 -> [2,2]: row0 (x2,x1), row1 (x1,x4).
    ("exp.scale", "U8", [2, 2], bytes([0x80, 0x7F, 0x7F, 0x81])),
    # A 32-wide FP4 weight at the *real* block size (32): every nibble 0x2 (=1.0), every e8m0 scale
    # 0x7F (=x1) -> all 64 logical values are 1.0. Exercises `linear`'s fp4 branch (FP4_BLOCK=32),
    # which the 4-wide `exp` above can't (block 32 > 4 cols). weight [2,16] packed, scale [2,1].
    ("exp32.weight", "U8", [2, 16], bytes([0x22] * 32)),
    ("exp32.scale", "U8", [2, 1], bytes([0x7F, 0x7F])),
    # A `.weight` with NO `.scale` sibling — mirrors the converted `wo_a` (bf16, pre-dequantized).
    # `linear("wo", ...)` must take the unquantized `auto_tensor` path for it.
    ("wo.weight", "BF16", [2, 2], bf16_bytes([1.0, 2.0, 0.5, -1.0])),
]


def main():
    header = {}
    blob = bytearray()
    for name, dtype, shape, data in TENSORS:
        start = len(blob)
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [start, start + len(data)]}
        blob += data

    hjson = json.dumps(header, separators=(",", ":")).encode("utf-8")
    hjson += b" " * ((-len(hjson)) % 8)  # pad header to an 8-byte boundary (safetensors convention)

    with open(OUT, "wb") as f:
        f.write(struct.pack("<Q", len(hjson)))
        f.write(hjson)
        f.write(blob)
    print("wrote", OUT, "(", 8 + len(hjson) + len(blob), "bytes )")


if __name__ == "__main__":
    main()
