//! Sparse KV-selection tests: the per-query key-index sets that feed `sparse_attn`.
//!
//! These are deterministic integer index math, golden-verified by hand against the
//! prefill (`start_pos == 0`) branch of `get_window_topk_idxs` (inference/model.py). The
//! sliding window is the causal key set truncated to the last `window` positions; `-1`
//! marks an empty slot (dropped by `sparse_attn`).

use candle_core::{DType, Device, Tensor};
use deepseek_v4_candle::rope::Rope;
use deepseek_v4_candle::sparse::{compress_topk_idxs, window_topk_idxs, Compressor, Indexer};

/// Window 3 over 5 positions: row `i` keeps keys `max(i-2,0)..=i`, right-padded with `-1`.
#[test]
fn window_idxs_sliding() -> candle_core::Result<()> {
    let dev = Device::Cpu;
    let idxs = window_topk_idxs(3, 5, &dev)?;
    assert_eq!(idxs.dims(), &[5, 3]);

    let got = idxs.flatten_all()?.to_vec1::<i64>()?;
    #[rustfmt::skip]
    let golden = [
        0, -1, -1,
        0,  1, -1,
        0,  1,  2,
        1,  2,  3,
        2,  3,  4,
    ];
    assert_eq!(got, golden);
    Ok(())
}

/// Sequence shorter than the window: every query sees all its past — plain lower-triangular
/// causal indices, truncated to `k = min(seqlen, window)` columns (no window cropping).
#[test]
fn window_idxs_shorter_than_window_is_full_causal() -> candle_core::Result<()> {
    let dev = Device::Cpu;
    let idxs = window_topk_idxs(8, 3, &dev)?;
    assert_eq!(idxs.dims(), &[3, 3]);

    let got = idxs.flatten_all()?.to_vec1::<i64>()?;
    #[rustfmt::skip]
    let golden = [
        0, -1, -1,
        0,  1, -1,
        0,  1,  2,
    ];
    assert_eq!(got, golden);
    Ok(())
}

/// Compression ratio 2 over 6 positions, compressed KVs offset to cache index 10: query `i`
/// sees compressed block `c` once it is fully in the past (`c < (i+1)/2`), as index `c + 10`.
/// Each block pools 2 consecutive tokens, so a new block becomes visible every 2 positions.
#[test]
fn compress_idxs_match_reference() -> candle_core::Result<()> {
    let dev = Device::Cpu;
    let idxs = compress_topk_idxs(2, 6, 10, &dev)?;
    assert_eq!(idxs.dims(), &[6, 3]);

    let got = idxs.flatten_all()?.to_vec1::<i64>()?;
    #[rustfmt::skip]
    let golden = [
        -1, -1, -1,
        10, -1, -1,
        10, -1, -1,
        10, 11, -1,
        10, 11, -1,
        10, 11, 12,
    ];
    assert_eq!(got, golden);
    Ok(())
}

/// Non-overlap prefill compression with identity projections and no RoPE (`rd = 0`): one
/// block of 2 tokens is gated-pooled (softmax over the block) then RMS-normed. Golden from
/// the pure-Python `compressor_golden.py` (`[1,2]`,`[3,4]` -> normed `[0.83692, 1.13998]`).
#[test]
fn compressor_gated_pool_norm() -> candle_core::Result<()> {
    let dev = Device::Cpu;
    let id = Tensor::from_vec(vec![1f32, 0., 0., 1.], (2, 2), &dev)?; // 2x2 identity wkv = wgate
    let comp = Compressor {
        wkv: id.clone(),
        wgate: id,
        ape: Tensor::zeros((2, 2), DType::F32, &dev)?,
        norm: Tensor::ones((2,), DType::F32, &dev)?,
        compress_ratio: 2,
        head_dim: 2,
        rope_head_dim: 0,
        eps: 1e-6,
    };
    let x = Tensor::from_vec(vec![1f32, 2., 3., 4.], (1, 2, 2), &dev)?; // tokens [1,2], [3,4]
    let rope = Rope::new(2, 4, 0, 10000.0, 1.0, 32.0, 1.0, &dev)?; // unused at rd = 0

    let out = comp.compress(&x, &rope)?;
    assert_eq!(out.dims(), &[1, 1, 2]);

    let got = out.flatten_all()?.to_vec1::<f32>()?;
    let golden = [0.836_923_7_f32, 1.139_981_8];
    for (k, (a, b)) in got.iter().zip(golden.iter()).enumerate() {
        assert!((a - b).abs() < 1e-5, "out[{k}] = {a}, expected {b}");
    }
    Ok(())
}

/// Overlapping prefill compression (CSA path, `compress_ratio == 4`): `wkv`/`wgate` project to
/// `2*head_dim`, and each block pools `2*ratio` tokens — its own (2nd-half dims) plus the
/// previous block's (1st-half dims), block 0's overlap half masked by `-inf`. Golden from the
/// pure-Python `overlap_golden.py` (d=2, ratio=4, seqlen=8 -> 2 blocks). Exercises the
/// cross-block overlap (block 1 must see block 0's tokens).
#[test]
fn compressor_overlap_pool_norm() -> candle_core::Result<()> {
    let dev = Device::Cpu;
    // wkv,wgate: [2d=4, dim=2]; ape: [ratio=4, 2d=4]; norm: [d=2].
    let wkv = Tensor::from_vec(
        vec![1f32, 0.5, 0.2, 1.0, -0.3, 0.4, 0.6, -0.2],
        (4, 2),
        &dev,
    )?;
    let wgate = Tensor::from_vec(
        vec![0.4f32, -0.1, 0.1, 0.3, -0.2, 0.5, 0.7, 0.0],
        (4, 2),
        &dev,
    )?;
    let ape = Tensor::from_vec(
        vec![
            0.10f32, -0.20, 0.30, 0.05, -0.15, 0.25, 0.00, 0.40, 0.20, 0.10, -0.30, 0.15, 0.05,
            -0.05, 0.35, -0.10,
        ],
        (4, 4),
        &dev,
    )?;
    let comp = Compressor {
        wkv,
        wgate,
        ape,
        norm: Tensor::ones((2,), DType::F32, &dev)?,
        compress_ratio: 4,
        head_dim: 2,
        rope_head_dim: 0,
        eps: 1e-6,
    };
    let x = Tensor::from_vec(
        vec![
            0.5f32, -0.3, 0.2, 0.8, -0.6, 0.1, 0.9, -0.4, 0.3, 0.7, -0.2, -0.5, 0.4, 0.6, -0.7, 0.2,
        ],
        (1, 8, 2),
        &dev,
    )?;
    let rope = Rope::new(2, 8, 0, 10000.0, 1.0, 32.0, 1.0, &dev)?; // unused at rd = 0

    let out = comp.compress(&x, &rope)?;
    assert_eq!(out.dims(), &[1, 2, 2]);

    let got = out.flatten_all()?.to_vec1::<f32>()?;
    let golden = [-0.201_197_2_f32, 1.399_798_8, 1.293_915, 0.570_724_5];
    for (k, (a, b)) in got.iter().zip(golden.iter()).enumerate() {
        assert!((a - b).abs() < 1e-5, "out[{k}] = {a}, expected {b}");
    }
    Ok(())
}

/// CSA Indexer learned top-k block selection (`index_topk = 1`, rd = 0 to isolate scoring from
/// RoPE). For each query the indexer scores every compressed block via a small per-head
/// attention (own overlap compressor for the KV, `wq_b` for Q), `relu`s, head-weights, sums,
/// causally masks future blocks, and keeps the top block — returned as a cache index
/// `block + offset` (offset = seqlen = 12) or `-1` when no block is yet visible. Golden (with a
/// searched, well-separated input so no f32 tie can flip the pick) from `indexer_golden.py`:
/// queries 0-2 see no block (`-1`); 3-6 see only block 0 (`12`); 7-11 pick block 1 (`13`).
#[test]
fn indexer_selects_topk_blocks() -> candle_core::Result<()> {
    let dev = Device::Cpu;
    // Indexer's own overlap compressor (head_dim = 2, ratio = 4, rd = 0).
    let compressor = Compressor {
        wkv: Tensor::from_vec(vec![1f32, 0.5, 0.2, 1.0, -0.3, 0.4, 0.6, -0.2], (4, 2), &dev)?,
        wgate: Tensor::from_vec(vec![0.4f32, -0.1, 0.1, 0.3, -0.2, 0.5, 0.7, 0.0], (4, 2), &dev)?,
        ape: Tensor::from_vec(
            vec![
                0.10f32, -0.20, 0.30, 0.05, -0.15, 0.25, 0.00, 0.40, 0.20, 0.10, -0.30, 0.15, 0.05,
                -0.05, 0.35, -0.10,
            ],
            (4, 4),
            &dev,
        )?,
        norm: Tensor::ones((2,), DType::F32, &dev)?,
        compress_ratio: 4,
        head_dim: 2,
        rope_head_dim: 0,
        eps: 1e-6,
    };
    let indexer = Indexer {
        wq_b: Tensor::from_vec(vec![0.6f32, -0.3, 0.2, 0.5, -0.4, 0.1, 0.3, 0.7], (4, 2), &dev)?,
        weights_proj: Tensor::from_vec(vec![0.5f32, -0.2, 0.1, 0.4], (2, 2), &dev)?,
        compressor,
        n_heads: 2,
        head_dim: 2,
        rope_head_dim: 0,
        index_topk: 1,
        compress_ratio: 4,
        scale: (2f64).powf(-0.5),
    };

    #[rustfmt::skip]
    let x = Tensor::from_vec(vec![
        0.841471f32, 0.745705, -0.44252, -0.982453, -0.083089, 0.938, 0.584917, -0.625071,
        -0.919329, 0.133232, 0.990607, 0.396741, -0.778352, -0.813157, 0.343315, 0.99683,
        0.189987, -0.895187, -0.66891, 0.537322, 0.956376, -0.025663, -0.970106, -0.493341,
    ], (1, 12, 2), &dev)?;
    #[rustfmt::skip]
    let qr = Tensor::from_vec(vec![
        0.354624f32, 1.091157, -0.635803, -0.927317, 0.874763, 0.701901, -1.055635, -0.429875,
        1.166409, 0.129304, -1.199729, 0.179853, 1.153383, -0.477067, -1.030448, 0.742602,
        0.839088, -0.958826, -0.592009, 1.11138, 0.305619, -1.190135, 0.001066, 1.18986,
    ], (1, 12, 2), &dev)?;
    let rope = Rope::new(2, 16, 0, 10000.0, 1.0, 32.0, 1.0, &dev)?; // unused at rd = 0

    let idxs = indexer.select(&x, &qr, &rope)?;
    assert_eq!(idxs.dims(), &[1, 12, 1]);

    let got = idxs.flatten_all()?.to_vec1::<i64>()?;
    let golden = [-1i64, -1, -1, 12, 12, 12, 12, 13, 13, 13, 13, 13];
    assert_eq!(got, golden);
    Ok(())
}
