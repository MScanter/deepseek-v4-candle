//! mHC tests.
//!
//! `sinkhorn_matches_reference` pins `hc_split_sinkhorn` to golden values from the independent
//! pure-Python port (`mhc_sinkhorn_golden.py`) — the Sinkhorn projection is the risky math, so it
//! is verified to the value. `comb_is_doubly_stochastic` then checks the defining invariant (after
//! Sinkhorn every row and column of `comb` sums to 1, i.e. it lives on the Birkhoff polytope) and
//! the gate ranges. `pre_post_shapes_round_trip` checks the `hc_pre` → `hc_post` shape contract.
//!
//! All inputs are deterministic (`det_at`, sin-of-arange): candle's CPU `randn` draws from a shared
//! global RNG that parallel test threads consume in nondeterministic order, which made a tolerance
//! assertion on random Sinkhorn output flaky. Deterministic inputs match the repo convention and
//! make every assertion reproducible.

use candle_core::{Device, Tensor};
use deepseek_v4_candle::mhc::{hc_split_sinkhorn, Hc};

const HC: usize = 4; // hc_mult
const ITERS: usize = 20; // hc_sinkhorn_iters
const EPS: f64 = 1e-6;
const MIX_HC: usize = (2 + HC) * HC; // 24

/// Deterministic `[o, i]` tensor from `sin` of a linear index (no RNG): `0.3 * sin(0.7*(start+k) + 1)`.
/// Matches the `det_at` helper used across the other test suites so goldens are reproducible.
fn det_at(o: usize, i: usize, start: i64, dev: &Device) -> candle_core::Result<Tensor> {
    let n = (o * i) as i64;
    Tensor::arange(start, start + n, dev)?
        .to_dtype(candle_core::DType::F32)?
        .affine(0.7, 1.0)?
        .sin()?
        .affine(0.3, 0.0)?
        .reshape((o, i))
}

/// The shared golden inputs: `mixes` `[2, 24]`, `hc_scale` `[3]` (amplified to 4 so the comb
/// softmax is non-trivially peaked), `hc_base` `[24]`. Identical to `mhc_sinkhorn_golden.py`.
fn golden_inputs(dev: &Device) -> candle_core::Result<(Tensor, Tensor, Tensor)> {
    let mixes = det_at(2, MIX_HC, 10, dev)?;
    let scale = Tensor::new(&[4.0f32, 4.0, 4.0], dev)?;
    let base = det_at(1, MIX_HC, 50, dev)?.reshape((MIX_HC,))?;
    Ok((mixes, scale, base))
}

/// Golden parity: `hc_split_sinkhorn` must reproduce the pure-Python `pre` / `post` / `comb`
/// to the value (math-only reference, derived independently of the Rust code).
#[test]
fn sinkhorn_matches_reference() -> candle_core::Result<()> {
    let dev = Device::Cpu;
    let (mixes, scale, base) = golden_inputs(&dev)?;

    let (pre, post, comb) = hc_split_sinkhorn(&mixes, &scale, &base, HC, ITERS, EPS)?;
    assert_eq!(pre.dims(), &[2, HC]);
    assert_eq!(post.dims(), &[2, HC]);
    assert_eq!(comb.dims(), &[2, HC, HC]);

    let pre_golden = [
        0.708_828_f32, 0.632_573, 0.485_337, 0.346_828, 0.334_146, 0.544_526, 0.723_654, 0.784_822,
    ];
    let post_golden = [
        0.574_147_f32, 0.637_966, 0.875_378, 1.185_425, 1.468_672, 1.129_911, 0.700_913, 0.461_116,
    ];
    let comb_golden = [
        0.277_498_f32, 0.325_558, 0.254_334, 0.142_609, 0.206_221, 0.174_172, 0.228_088, 0.391_519,
        0.162_219, 0.278_386, 0.326_458, 0.232_936, 0.354_061, 0.221_883, 0.191_119, 0.232_935,
        0.082_716, 0.138_254, 0.319_754, 0.459_275, 0.469_726, 0.336_611, 0.151_888, 0.041_773,
        0.162_536, 0.154_884, 0.264_314, 0.418_265, 0.285_021, 0.370_25, 0.264_043, 0.080_685,
    ];

    let check = |got: &Tensor, golden: &[f32], name: &str| -> candle_core::Result<()> {
        for (k, (a, b)) in got.flatten_all()?.to_vec1::<f32>()?.iter().zip(golden.iter()).enumerate()
        {
            assert!((a - b).abs() < 1e-4, "{name}[{k}] = {a}, expected {b}");
        }
        Ok(())
    };
    check(&pre, &pre_golden, "pre")?;
    check(&post, &post_golden, "post")?;
    check(&comb, &comb_golden, "comb")?;
    Ok(())
}

/// The defining mHC invariant: after Sinkhorn, `comb` is doubly-stochastic (every row and column
/// sums to 1), and the gates lie in their ranges (`post` in (0,2), `pre` in (0,1+eps]).
#[test]
fn comb_is_doubly_stochastic() -> candle_core::Result<()> {
    let dev = Device::Cpu;
    let (mixes, scale, base) = golden_inputs(&dev)?;

    let (pre, post, comb) = hc_split_sinkhorn(&mixes, &scale, &base, HC, ITERS, EPS)?;

    let row_sums = comb.sum(2)?.flatten_all()?.to_vec1::<f32>()?; // [n*HC]
    let col_sums = comb.sum(1)?.flatten_all()?.to_vec1::<f32>()?; // [n*HC]
    for (label, sums) in [("row", &row_sums), ("col", &col_sums)] {
        for &v in sums {
            assert!((v - 1.0).abs() < 1e-3, "{label} sum {v} deviates from 1 by more than 1e-3");
        }
    }

    for &v in &post.flatten_all()?.to_vec1::<f32>()? {
        assert!(v > 0.0 && v < 2.0, "post {v} out of (0,2)");
    }
    for &v in &pre.flatten_all()?.to_vec1::<f32>()? {
        assert!(v > 0.0 && v < 1.0 + 1e-3, "pre {v} out of (0,1]");
    }
    Ok(())
}

/// The `hc_pre` → `hc_post` shape contract: streams `[b,s,hc,dim]` collapse to `[b,s,dim]` for the
/// sublayer and re-expand back to `[b,s,hc,dim]`, with `post`/`comb` carrying the mixing weights.
#[test]
fn pre_post_shapes_round_trip() -> candle_core::Result<()> {
    let dev = Device::Cpu;
    let (b, s, dim) = (2usize, 3usize, 8usize);

    let hc = Hc {
        hc_fn: det_at(MIX_HC, HC * dim, 3, &dev)?,
        hc_base: det_at(1, MIX_HC, 71, &dev)?.reshape((MIX_HC,))?,
        hc_scale: Tensor::new(&[1.0f32, 1.0, 1.0], &dev)?,
        hc: HC,
        sinkhorn_iters: ITERS,
        eps: EPS,
        norm_eps: 1e-6,
    };

    // Streams in: [b, s, hc, dim]
    let x = det_at(b * s * HC, dim, 200, &dev)?.reshape((b, s, HC, dim))?;
    let (collapsed, post, comb) = hc.pre(&x)?;
    assert_eq!(collapsed.dims(), &[b, s, dim]);
    assert_eq!(post.dims(), &[b, s, HC]);
    assert_eq!(comb.dims(), &[b, s, HC, HC]);

    // Pretend the sublayer returned `collapsed`; re-expand.
    let out = hc.post(&collapsed, &x, &post, &comb)?;
    assert_eq!(out.dims(), &[b, s, HC, dim]);
    Ok(())
}
