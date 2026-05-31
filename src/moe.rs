//! Mixture-of-Experts: routing gate, SwiGLU experts, and the routed + shared combine.
//!
//! Ports `Gate` / `Expert` / `MoE` (inference/model.py). V4 routes each token to
//! `n_activated_experts` of `n_routed_experts` via a learned gate, plus one always-on shared
//! expert. The gate's novelty is `sqrtsoftplus` scoring with a selection-only bias (see [`Gate`]).

use crate::attention::linear;
use candle_core::{DType, Result, Tensor, D};

/// `sqrt(softplus(x))`, V4's routing nonlinearity. `softplus = relu(x) + ln(1 + e^{-|x|})` is the
/// numerically stable form (no overflow for large `x`), matching `F.softplus(x).sqrt()`.
fn sqrtsoftplus(x: &Tensor) -> Result<Tensor> {
    let ln_term = x.abs()?.neg()?.exp()?.affine(1.0, 1.0)?.log()?; // ln(1 + e^{-|x|})
    x.relu()?.broadcast_add(&ln_term)?.sqrt()
}

/// Numerically stable softmax over the last dim.
fn softmax_last(x: &Tensor) -> Result<Tensor> {
    let e = x.broadcast_sub(&x.max_keepdim(D::Minus1)?)?.exp()?;
    e.broadcast_div(&e.sum_keepdim(D::Minus1)?)
}

/// Logistic sigmoid `1 / (1 + e^{-x})`.
fn sigmoid(x: &Tensor) -> Result<Tensor> {
    let denom = x.neg()?.exp()?.affine(1.0, 1.0)?; // 1 + e^{-x}
    Tensor::ones_like(&denom)?.broadcast_div(&denom)
}

/// MoE gate scoring function. V4-Flash uses [`ScoreFunc::SqrtSoftplus`]; the others mirror the
/// reference's `softmax`/`sigmoid` options. Only `Softmax` skips the top-k weight renormalization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScoreFunc {
    SqrtSoftplus,
    Softmax,
    Sigmoid,
}

/// MoE routing gate (the score-based path of `Gate.forward`, inference/model.py 564-585).
///
/// Scores each token against every routed expert, selects the top-k, and returns their routing
/// weights. The V4 subtlety: a per-expert `bias` is added **only for the top-k selection**, while
/// the returned weights are gathered from the *pre-bias* scores — then (for non-softmax score
/// functions) renormalized to sum 1 and scaled by `route_scale`.
///
/// The hash-routing path (first `n_hash_layers`, `bias = None`, indices from a precomputed
/// `tid2eid[input_ids]` table) is deferred — those layers select via [`Gate::route_hashed`].
pub struct Gate {
    /// `[n_routed_experts, dim]` — expert scoring projection.
    pub weight: Tensor,
    /// `[n_routed_experts]` — selection bias, `None` for hash-routed layers.
    pub bias: Option<Tensor>,
    /// `n_activated_experts` — experts kept per token.
    pub topk: usize,
    pub route_scale: f64,
    pub score_func: ScoreFunc,
}

impl Gate {
    /// `x`: `[n_tokens, dim]` → `(weights [n_tokens, topk] f32, indices [n_tokens, topk] i64)`.
    /// Indices are ordered by descending (biased) score; weights are the renormalized, scaled
    /// pre-bias scores at those indices.
    pub fn route(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        let raw = linear(x, &self.weight)?; // [n, n_routed]
        let scores = match self.score_func {
            ScoreFunc::SqrtSoftplus => sqrtsoftplus(&raw)?,
            ScoreFunc::Softmax => softmax_last(&raw)?,
            ScoreFunc::Sigmoid => sigmoid(&raw)?,
        };
        // Bias shifts the top-k *selection* only; weights are read from the pre-bias scores.
        let biased = match &self.bias {
            Some(b) => scores.broadcast_add(b)?,
            None => scores.clone(),
        };

        let (n, e) = scores.dims2()?;
        let k = self.topk.min(e);
        let sc = scores.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let bi = biased.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let mut wv = vec![0f32; n * k];
        let mut iv = vec![0i64; n * k];
        for r in 0..n {
            let base = r * e;
            let mut cand: Vec<(f32, usize)> = (0..e).map(|j| (bi[base + j], j)).collect();
            cand.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            for (t, &(_, j)) in cand.iter().take(k).enumerate() {
                iv[r * k + t] = j as i64;
                wv[r * k + t] = sc[base + j]; // gather PRE-bias score at the selected index
            }
        }
        let weights = Tensor::from_vec(wv, (n, k), x.device())?;
        let indices = Tensor::from_vec(iv, (n, k), x.device())?;

        // Renormalize the selected weights to sum 1 (non-softmax only), then apply route_scale.
        let weights = if matches!(self.score_func, ScoreFunc::Softmax) {
            weights.affine(self.route_scale, 0.0)?
        } else {
            let denom = weights.sum_keepdim(D::Minus1)?;
            weights.broadcast_div(&denom)?.affine(self.route_scale, 0.0)?
        };
        Ok((weights, indices))
    }
}

/// A single MoE expert: a SwiGLU feed-forward network (`Expert.forward`, inference/model.py
/// 596-608). `h = silu(w1·x) * (w3·x)`, then `w2·h`; an optional per-token routing `weights`
/// (`[n, 1]`) scales `h` before the down-projection. `swiglu_limit > 0` clamps the activations:
/// the up path to `[-limit, limit]`, the gate path on the max side only (matching the reference).
pub struct Expert {
    /// `[inter_dim, dim]` — gate (SwiGLU "value") projection.
    pub w1: Tensor,
    /// `[dim, inter_dim]` — down projection.
    pub w2: Tensor,
    /// `[inter_dim, dim]` — up projection.
    pub w3: Tensor,
    /// SwiGLU clamp; `0.0` disables clamping.
    pub swiglu_limit: f64,
}

impl Expert {
    /// `x`: `[n_tokens, dim]` → `[n_tokens, dim]`. `weights`, if given, is `[n_tokens, 1]` and
    /// scales each token's hidden activations before the down-projection.
    pub fn forward(&self, x: &Tensor, weights: Option<&Tensor>) -> Result<Tensor> {
        let mut gate = linear(x, &self.w1)?; // [n, inter]
        let mut up = linear(x, &self.w3)?; // [n, inter]
        if self.swiglu_limit > 0.0 {
            let l = self.swiglu_limit;
            up = up.clamp(-l, l)?; // up: both sides
            gate = gate.minimum(l)?; // gate: max side only
        }
        let mut h = gate.silu()?.mul(&up)?; // silu(gate) * up
        if let Some(w) = weights {
            h = h.broadcast_mul(w)?; // per-token routing weight, pre-down-projection
        }
        linear(&h, &self.w2)
    }
}

/// A full Mixture-of-Experts layer (`MoE.forward`, inference/model.py 609-660): a routing [`Gate`],
/// `n_routed_experts` routed [`Expert`]s, and one always-on shared expert.
pub struct Moe {
    pub gate: Gate,
    /// The routed experts, indexed by the gate's expert ids.
    pub experts: Vec<Expert>,
    /// The shared expert, applied to every token with no routing weight.
    pub shared: Expert,
}

impl Moe {
    /// `x`: `[n_tokens, dim]` → `[n_tokens, dim]`.
    ///
    /// Routes each token to its top-k experts, runs every routed expert on only its own tokens
    /// (weighted by the gate), scatter-adds the results back, then adds the shared expert over all
    /// tokens. Mirrors the reference's `y[idx] += expert(x[idx], w[idx]); y += shared(x)`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (n, dim) = x.dims2()?;
        let (weights, indices) = self.gate.route(x)?;
        let topk = self.gate.topk;
        let wv = weights.flatten_all()?.to_vec1::<f32>()?; // [n*topk]
        let iv = indices.flatten_all()?.to_vec1::<i64>()?; // [n*topk]

        let mut y = Tensor::zeros((n, dim), DType::F32, x.device())?;
        for (e, expert) in self.experts.iter().enumerate() {
            // Gather the (token, weight) pairs routed to expert `e`.
            let mut rows = Vec::new();
            let mut ws = Vec::new();
            for t in 0..n {
                for s in 0..topk {
                    if iv[t * topk + s] == e as i64 {
                        rows.push(t as u32);
                        ws.push(wv[t * topk + s]);
                    }
                }
            }
            if rows.is_empty() {
                continue; // no token chose this expert
            }
            let sel = Tensor::from_vec(rows.clone(), (rows.len(),), x.device())?;
            let xe = x.index_select(&sel, 0)?; // [r, dim]
            let we = Tensor::from_vec(ws, (rows.len(), 1), x.device())?; // [r, 1]
            let ye = expert.forward(&xe, Some(&we))?; // [r, dim]
            y = y.index_add(&sel, &ye, 0)?; // scatter-add back into the routed rows
        }

        // Shared expert: every token, no routing weight.
        let ys = self.shared.forward(x, None)?;
        y.broadcast_add(&ys)
    }
}
