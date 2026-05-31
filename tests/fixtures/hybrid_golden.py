#!/usr/bin/env python3
"""Independent pure-Python (math-only, f64) golden for a 3-layer HYBRID DeepSeek-V4 toy:
layer 0 = sliding-window (ratio 0), layer 1 = HCA (ratio 2), layer 2 = CSA (ratio 4).

Transcribed from the REFERENCE inference/model.py formulas (NOT from the Rust crate), so a
green test is a genuine cross-check of the assembled HCA/CSA forward path, the per-layer rope
selection (window rope_theta vs compress_rope_theta), and the multi-layer composition.

Same documented omissions as the Rust port (all no-ops for this toy): no Hadamard rotation,
no FP8/FP4 act-quant QAT sim, prefill only (start_pos == 0), seqlen % ratio == 0 (no remainder).
YaRN is off everywhere (original_seq_len == 0, rope_factor == 1); the per-layer ropes differ only
in theta, which is observable because rope_head_dim == 4 (>1 frequency index).

Weights are the deterministic det(o,i,start) = 0.3*sin(0.7*(start+k)+1) used by the Rust toy
builders, with a per-layer offset L = layer*1000 so a misrouted layer diverges from this golden.
"""
import math

# ---- toy dims (must match tests/common/mod.rs hybrid builders + toy_hybrid_config.json) ----
DIM, HC, H, HD, RD = 8, 2, 2, 8, 4
QLR, G, OLR = 4, 2, 2
NROUTED, TOPK, INTER, VOCAB = 4, 2, 4, 6
MIXHC = (2 + HC) * HC  # 8
IDXH, IDXHD, IDXTOPK = 1, 8, 1
ITERS = 20
WINDOW = 128
EPS = 1e-6
ROUTE_SCALE = 1.5
ROPE_THETA, COMPRESS_ROPE_THETA = 10000.0, 160000.0
COMPRESS_RATIOS = [0, 2, 4]
INPUT_IDS = [1, 3, 0, 2, 4, 1, 5, 2]


# ===== basic linear algebra on nested python lists (f64) =====
def det(o, i, start):
    """Row-major [o][i] of 0.3*sin(0.7*(start+k)+1) — mirrors det_at in tests/common/mod.rs."""
    flat = [0.3 * math.sin(0.7 * (start + k) + 1.0) for k in range(o * i)]
    return [flat[r * i:(r + 1) * i] for r in range(o)]


def ones(n):
    return [1.0] * n


def linear(x, W):
    """x: [n][in], W: [out][in] -> [n][out]  (y = x @ W^T)."""
    out = len(W)
    return [[sum(xr[k] * W[o][k] for k in range(len(xr))) for o in range(out)] for xr in x]


def rms_norm(rows, gamma, eps=EPS):
    """RMSNorm over last dim for each row. gamma None -> weightless."""
    res = []
    for r in rows:
        ms = sum(v * v for v in r) / len(r)
        inv = 1.0 / math.sqrt(ms + eps)
        if gamma is None:
            res.append([v * inv for v in r])
        else:
            res.append([v * inv * gamma[j] for j, v in enumerate(r)])
    return res


def softmax(xs):
    m = max(xs)
    e = [math.exp(v - m) for v in xs]
    s = sum(e)
    return [v / s for v in e]


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def silu(x):
    return x * sigmoid(x)


def softplus(x):
    return max(x, 0.0) + math.log1p(math.exp(-abs(x)))


def sqrtsoftplus(x):
    return math.sqrt(softplus(x))


# ===== YaRN RoPE (interleaved). original_seq_len==0 -> plain rope at `base`. =====
def rope_freqs(dim, base, original_seq_len, factor, beta_fast, beta_slow):
    half = dim // 2
    freqs = [1.0 / base ** ((2 * i) / dim) for i in range(half)]
    if original_seq_len > 0:  # YaRN (unused in this toy: original_seq_len==0)
        def corr_dim(nr):
            return dim * math.log(original_seq_len / (nr * 2 * math.pi)) / (2 * math.log(base))
        low = max(math.floor(corr_dim(beta_fast)), 0.0)
        high = min(math.ceil(corr_dim(beta_slow)), dim - 1)
        denom = 0.001 if low == high else (high - low)
        for i in range(half):
            ramp = min(max((i - low) / denom, 0.0), 1.0)
            smooth = 1.0 - ramp
            freqs[i] = freqs[i] / factor * (1.0 - smooth) + freqs[i] * smooth
    return freqs


def make_rope(theta):
    # toy: original_seq_len=0, factor=1 -> YaRN off; only theta varies per layer.
    return rope_freqs(RD, theta, 0, 1.0, 32.0, 1.0)


def apply_rope_tail(vec, pos, freqs, inverse=False):
    """Rotate the last RD dims of `vec` (len HD) at absolute position `pos`. Interleaved pairs."""
    d = len(vec)
    nope = d - RD
    out = vec[:nope] + [0.0] * RD
    for j in range(RD // 2):
        ang = pos * freqs[j]
        c, sn = math.cos(ang), math.sin(ang)
        if inverse:
            sn = -sn
        e = vec[nope + 2 * j]
        o = vec[nope + 2 * j + 1]
        out[nope + 2 * j] = e * c - o * sn
        out[nope + 2 * j + 1] = e * sn + o * c
    return out


# ===== KV compressor (gated pool + RMSNorm + strided rope), HCA non-overlap / CSA overlap =====
def compressor_compress(x, wkv, wgate, ape, norm, ratio, head_dim, freqs):
    """x: [s][DIM] -> compressed kv [nb][head_dim]. coff=2 (overlap) when ratio==4 else 1."""
    s = len(x)
    nb = s // ratio
    overlap = (ratio == 4)
    proj_d = 2 * head_dim if overlap else head_dim
    kv_p = linear(x, wkv)      # [s][proj_d]
    sc_p = linear(x, wgate)    # [s][proj_d]
    # fold into blocks of `ratio` tokens, add within-block APE to the gate.
    kv_b = [[kv_p[blk * ratio + t] for t in range(ratio)] for blk in range(nb)]            # [nb][ratio][proj_d]
    sc_b = [[[sc_p[blk * ratio + t][c] + ape[t][c] for c in range(proj_d)]
             for t in range(ratio)] for blk in range(nb)]                                  # [nb][ratio][proj_d]

    if overlap:
        kv_w = _overlap(kv_b, head_dim, nb, ratio, 0.0)
        sc_w = _overlap(sc_b, head_dim, nb, ratio, float("-inf"))
        ntok = 2 * ratio
    else:
        kv_w, sc_w, ntok = kv_b, sc_b, ratio

    # gated softmax pool over the token axis, per dimension.
    pooled = []
    for blk in range(nb):
        outd = []
        for d in range(head_dim):
            w = softmax([sc_w[blk][t][d] for t in range(ntok)])
            outd.append(sum(kv_w[blk][t][d] * w[t] for t in range(ntok)))
        pooled.append(outd)

    pooled = rms_norm(pooled, norm)                       # [nb][head_dim]
    # RoPE each block at its leading token's position blk*ratio.
    return [apply_rope_tail(pooled[blk], blk * ratio, freqs) for blk in range(nb)]


def _overlap(blocks, d, nb, ratio, fill):
    """blocks: [nb][ratio][2d] -> [nb][2*ratio][d]. tokens [ratio:] = current 2nd-half;
    tokens [:ratio] of block i = block i-1's 1st-half (block 0 -> fill)."""
    res = []
    for blk in range(nb):
        toks = []
        for t in range(ratio):  # overlap (previous block's first half)
            if blk == 0:
                toks.append([fill] * d)
            else:
                toks.append([blocks[blk - 1][t][c] for c in range(d)])
        for t in range(ratio):  # current block's second half
            toks.append([blocks[blk][t][d + c] for c in range(d)])
        res.append(toks)
    return res


# ===== learned block selector (CSA indexer) =====
def indexer_select(x, qr, lw, freqs):
    """Returns topk_idxs [s][k] (cache indices block+offset or -1). offset = s."""
    s = len(x)
    ratio = 4
    nb = s // ratio
    offset = s
    k = min(IDXTOPK, nb)
    # per-head queries from qr (no weightless RMS), rope tail at position i.
    q = linear(qr, lw["idx_wq_b"])  # [s][IDXH*IDXHD]
    q = [[apply_rope_tail(q[i][hh * IDXHD:(hh + 1) * IDXHD], i, freqs) for hh in range(IDXH)] for i in range(s)]
    # indexer's own (overlapping) compressed kv.
    kv = compressor_compress(x, lw["idx_c_wkv"], lw["idx_c_wgate"], lw["idx_c_ape"], lw["idx_c_norm"],
                             ratio, IDXHD, freqs)  # [nb][IDXHD]
    # per-head scoring weights: weights_proj(x) * (head_dim^-0.5 * n_heads^-0.5).
    wsc = (IDXHD ** -0.5) * (IDXH ** -0.5)
    weights = [[v * wsc for v in row] for row in linear(x, lw["idx_weights_proj"])]  # [s][IDXH]
    # index_score[i][t] = sum_h relu(q[i][h]·kv[t]) * weight[i][h]
    idxs = []
    for i in range(s):
        score = []
        for t in range(nb):
            tot = 0.0
            for hh in range(IDXH):
                dot = sum(q[i][hh][c] * kv[t][c] for c in range(IDXHD))
                tot += max(dot, 0.0) * weights[i][hh]
            score.append(tot)
        visible = (i + 1) // ratio
        cand = sorted([(score[t], t) for t in range(nb) if t < visible], key=lambda p: -p[0])
        row = [-1] * k
        for j, (_, t) in enumerate(cand[:k]):
            row[j] = t + offset
        idxs.append(row)
    return idxs


def window_topk_idxs(s):
    k = min(WINDOW, s)
    idxs = []
    for i in range(s):
        lo = max(i - WINDOW + 1, 0)
        idxs.append([(lo + j) if (lo + j) <= i else -1 for j in range(k)])
    return idxs


def compress_topk_idxs(ratio, s, offset):
    cols = s // ratio
    idxs = []
    for i in range(s):
        vis = min((i + 1) // ratio, cols)
        idxs.append([(c + offset) if c < vis else -1 for c in range(cols)])
    return idxs


# ===== sink attention over a gathered key set =====
def sparse_attn(q, kv, sink, idxs, scale):
    """q: [s][H][HD], kv: [n][HD] (single latent head), idxs: [s][k]. -> o: [s][H][HD]."""
    s = len(q)
    n = len(kv)
    o = [[[0.0] * HD for _ in range(H)] for _ in range(s)]
    for i in range(s):
        keep = [j for j in idxs[i] if 0 <= j < n]
        for h in range(H):
            sc = [scale * sum(q[i][h][d] * kv[j][d] for d in range(HD)) for j in keep]
            m = max(sc) if sc else 0.0
            ex = [math.exp(v - m) for v in sc]
            denom = sum(ex) + math.exp(sink[h] - m)
            for jj, j in enumerate(keep):
                w = ex[jj] / denom
                for d in range(HD):
                    o[i][h][d] += w * kv[j][d]
    return o


# ===== MLA forward (one layer) =====
def mla_forward(x, lw, ratio, freqs):
    """x: [s][DIM] -> [s][DIM]."""
    s = len(x)
    qr = rms_norm(linear(x, lw["wq_a"]), lw["q_norm"])             # [s][QLR]
    qb = linear(qr, lw["wq_b"])                                    # [s][H*HD]
    q = []
    for i in range(s):
        heads = []
        for h in range(H):
            hv = rms_norm([qb[i][h * HD:(h + 1) * HD]], None)[0]   # weightless per-head RMS
            heads.append(apply_rope_tail(hv, i, freqs))            # rope tail at pos i
        q.append(heads)                                            # [s][H][HD]

    kv0 = rms_norm(linear(x, lw["wkv"]), lw["kv_norm"])            # [s][HD]
    kv = [apply_rope_tail(kv0[i], i, freqs) for i in range(s)]     # latent kv, roped

    idxs = window_topk_idxs(s)                                     # window keys (full causal here)
    if ratio > 0:
        kvc = compressor_compress(x, lw["c_wkv"], lw["c_wgate"], lw["c_ape"], lw["c_norm"], ratio, HD, freqs)
        offset = s
        if ratio == 4:
            cidx = indexer_select(x, qr, lw, freqs)
        else:
            cidx = compress_topk_idxs(ratio, s, offset)
        kv = kv + kvc                                              # concat compressed blocks
        idxs = [idxs[i] + cidx[i] for i in range(s)]               # union window + compressed

    scale = HD ** -0.5
    o = sparse_attn(q, kv, lw["attn_sink"], idxs, scale)           # [s][H][HD]
    # inverse rope on the output's rope dims, at pos i.
    o = [[apply_rope_tail(o[i][h], i, freqs, inverse=True) for h in range(H)] for i in range(s)]

    # grouped low-rank output projection: einsum bsgd,grd->bsgr, then wo_b.
    din = H * HD // G
    wo_a = lw["wo_a"]   # [G*OLR][din]
    wo_b = lw["wo_b"]   # [DIM][G*OLR]
    out = []
    for i in range(s):
        flat = [o[i][h][d] for h in range(H) for d in range(HD)]   # [H*HD]
        og = [0.0] * (G * OLR)
        for g in range(G):
            grp = flat[g * din:(g + 1) * din]                      # [din]
            for r in range(OLR):
                row = wo_a[g * OLR + r]                            # [din]
                og[g * OLR + r] = sum(grp[k] * row[k] for k in range(din))
        out.append([sum(og[k] * wo_b[o2][k] for k in range(G * OLR)) for o2 in range(DIM)])
    return out


# ===== mHC (Sinkhorn doubly-stochastic hyper-connections) =====
def hc_split_sinkhorn(mixes, hc_scale, hc_base):
    """mixes: [n][MIXHC] -> (pre [n][HC], post [n][HC], comb [n][HC][HC])."""
    s0, s1, s2 = hc_scale
    pre, post, comb = [], [], []
    for mx in mixes:
        pr = [sigmoid(mx[k] * s0 + hc_base[k]) + EPS for k in range(HC)]
        po = [2.0 * sigmoid(mx[HC + k] * s1 + hc_base[HC + k]) for k in range(HC)]
        logits = [[mx[2 * HC + a * HC + b] * s2 + hc_base[2 * HC + a * HC + b] for b in range(HC)] for a in range(HC)]
        c = [softmax(row) for row in logits]
        c = [[v + EPS for v in row] for row in c]
        # one column normalization, then ITERS-1 (row, col) iterations.
        col = [sum(c[a][b] for a in range(HC)) + EPS for b in range(HC)]
        c = [[c[a][b] / col[b] for b in range(HC)] for a in range(HC)]
        for _ in range(ITERS - 1):
            for a in range(HC):
                rs = sum(c[a]) + EPS
                c[a] = [v / rs for v in c[a]]
            col = [sum(c[a][b] for a in range(HC)) + EPS for b in range(HC)]
            c = [[c[a][b] / col[b] for b in range(HC)] for a in range(HC)]
        pre.append(pr); post.append(po); comb.append(c)
    return pre, post, comb


def hc_pre(x, hc_fn, hc_base, hc_scale):
    """x: [s][HC][DIM] -> (collapsed [s][DIM], post [s][HC], comb [s][HC][HC])."""
    s = len(x)
    flat = [[x[i][k][d] for k in range(HC) for d in range(DIM)] for i in range(s)]  # [s][HC*DIM]
    mixes = []
    for i in range(s):
        ms = sum(v * v for v in flat[i]) / (HC * DIM)
        rms = 1.0 / math.sqrt(ms + EPS)
        proj = linear([flat[i]], hc_fn)[0]
        mixes.append([v * rms for v in proj])
    pre, post, comb = hc_split_sinkhorn(mixes, hc_scale, hc_base)
    collapsed = [[sum(pre[i][k] * x[i][k][d] for k in range(HC)) for d in range(DIM)] for i in range(s)]
    return collapsed, post, comb


def hc_post(sub, residual, post, comb):
    """sub: [s][DIM], residual: [s][HC][DIM] -> [s][HC][DIM].
    y[i][k][:] = post[i][k]*sub[i][:] + sum_j comb[i][j][k]*residual[i][j][:]."""
    s = len(sub)
    out = []
    for i in range(s):
        streams = []
        for k in range(HC):
            row = []
            for d in range(DIM):
                v = post[i][k] * sub[i][d]
                v += sum(comb[i][j][k] * residual[i][j][d] for j in range(HC))
                row.append(v)
            streams.append(row)
        out.append(streams)
    return out


# ===== MoE =====
def moe(x, lw):
    """x: [s][DIM] -> [s][DIM]."""
    s = len(x)
    raw = linear(x, lw["gate_w"])  # [s][NROUTED]
    out = [[0.0] * DIM for _ in range(s)]
    for i in range(s):
        scores = [sqrtsoftplus(v) for v in raw[i]]
        biased = [scores[e] + lw["gate_b"][e] for e in range(NROUTED)]
        order = sorted(range(NROUTED), key=lambda e: -biased[e])[:TOPK]
        wsel = [scores[e] for e in order]
        denom = sum(wsel)
        wsel = [w / denom * ROUTE_SCALE for w in wsel]
        for sel, e in enumerate(order):
            w1, w3, w2 = lw["exp"][e]
            gate = linear([x[i]], w1)[0]
            up = linear([x[i]], w3)[0]
            h = [silu(gate[j]) * up[j] * wsel[sel] for j in range(INTER)]
            ye = linear([h], w2)[0]
            for d in range(DIM):
                out[i][d] += ye[d]
        # shared expert (no routing weight).
        w1, w3, w2 = lw["shared"]
        gate = linear([x[i]], w1)[0]
        up = linear([x[i]], w3)[0]
        h = [silu(gate[j]) * up[j] for j in range(INTER)]
        ys = linear([h], w2)[0]
        for d in range(DIM):
            out[i][d] += ys[d]
    return out


# ===== block =====
def block(x, lw, ratio, freqs):
    """x: [s][HC][DIM] -> [s][HC][DIM]."""
    collapsed, post, comb = hc_pre(x, lw["hca_fn"], lw["hca_base"], lw["hca_scale"])
    normed = rms_norm(collapsed, lw["attn_norm"])
    attended = mla_forward(normed, lw, ratio, freqs)
    x = hc_post(attended, x, post, comb)

    collapsed, post, comb = hc_pre(x, lw["hcf_fn"], lw["hcf_base"], lw["hcf_scale"])
    normed = rms_norm(collapsed, lw["ffn_norm"])
    ff = moe(normed, lw)
    return hc_post(ff, x, post, comb)


# ===== head =====
def head_forward(x, hw):
    """x: [s][HC][DIM] -> logits [VOCAB] (last position). Simplified hc_pre collapse (pre gate only)."""
    s = len(x)
    flat = [[x[i][k][d] for k in range(HC) for d in range(DIM)] for i in range(s)]
    collapsed = []
    for i in range(s):
        ms = sum(v * v for v in flat[i]) / (HC * DIM)
        rms = 1.0 / math.sqrt(ms + EPS)
        mixes = [v * rms for v in linear([flat[i]], hw["hc_fn"])[0]]
        pre = [sigmoid(mixes[k] * hw["hc_scale"][0] + hw["hc_base"][k]) + EPS for k in range(HC)]
        collapsed.append([sum(pre[k] * x[i][k][d] for k in range(HC)) for d in range(DIM)])
    normed = rms_norm(collapsed, hw["norm"])
    last = normed[s - 1]
    return linear([last], hw["weight"])[0]


# ===== assemble the toy weights (per-layer offset L = layer*1000) =====
def layer_weights(layer):
    L = layer * 1000
    ratio = COMPRESS_RATIOS[layer]
    lw = {
        "wq_a": det(QLR, DIM, L + 1), "q_norm": ones(QLR),
        "wq_b": det(H * HD, QLR, L + 11), "wkv": det(HD, DIM, L + 23), "kv_norm": ones(HD),
        "wo_a": det(G * OLR, (H * HD) // G, L + 31), "wo_b": det(DIM, G * OLR, L + 43),
        "attn_sink": det(1, H, L + 7)[0],
        "attn_norm": ones(DIM), "ffn_norm": ones(DIM),
        "gate_w": det(NROUTED, DIM, L + 300), "gate_b": det(1, NROUTED, L + 311)[0],
        "exp": [(det(INTER, DIM, L + 100 + j * 50), det(INTER, DIM, L + 120 + j * 50),
                 det(DIM, INTER, L + 140 + j * 50)) for j in range(NROUTED)],
        "shared": (det(INTER, DIM, L + 400), det(INTER, DIM, L + 420), det(DIM, INTER, L + 440)),
        "hca_fn": det(MIXHC, HC * DIM, L + 600), "hca_base": det(1, MIXHC, L + 697)[0], "hca_scale": ones(3),
        "hcf_fn": det(MIXHC, HC * DIM, L + 700), "hcf_base": det(1, MIXHC, L + 797)[0], "hcf_scale": ones(3),
    }
    if ratio > 0:  # HCA/CSA compressor (coff = 2 for ratio 4, else 1)
        coff = 2 if ratio == 4 else 1
        lw["c_wkv"] = det(coff * HD, DIM, L + 50)
        lw["c_wgate"] = det(coff * HD, DIM, L + 60)
        lw["c_ape"] = det(ratio, coff * HD, L + 70)
        lw["c_norm"] = ones(HD)
    if ratio == 4:  # CSA indexer (+ its own overlapping compressor at IDXHD)
        lw["idx_wq_b"] = det(IDXH * IDXHD, QLR, L + 80)
        lw["idx_weights_proj"] = det(IDXH, DIM, L + 85)
        lw["idx_c_wkv"] = det(2 * IDXHD, DIM, L + 90)
        lw["idx_c_wgate"] = det(2 * IDXHD, DIM, L + 110)
        lw["idx_c_ape"] = det(4, 2 * IDXHD, L + 130)
        lw["idx_c_norm"] = ones(IDXHD)
    return lw


def head_weights():
    return {
        "weight": det(VOCAB, DIM, 500), "norm": ones(DIM),
        "hc_fn": det(HC, HC * DIM, 520), "hc_base": det(1, HC, 560)[0], "hc_scale": ones(1),
    }


def main():
    embed = det(VOCAB, DIM, 800)
    # embed lookup -> [s][DIM], expand to HC identical streams -> [s][HC][DIM].
    h = [[embed[tid][:] for _ in range(HC)] for tid in INPUT_IDS]
    for layer in range(len(COMPRESS_RATIOS)):
        ratio = COMPRESS_RATIOS[layer]
        theta = COMPRESS_ROPE_THETA if ratio > 0 else ROPE_THETA
        freqs = make_rope(theta)
        h = block(h, layer_weights(layer), ratio, freqs)
    logits = head_forward(h, head_weights())
    print("input_ids:", INPUT_IDS)
    print("hybrid logits:", [round(v, 6) for v in logits])
    print("rust-literal:", "[" + ", ".join("%.6f" % v for v in logits) + "]")


if __name__ == "__main__":
    main()
