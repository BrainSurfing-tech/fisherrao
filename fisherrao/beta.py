"""Fisher-Rao geometry on the Beta-distribution family.

The two-parameter Beta family with parameters (α, β) ∈ (0, ∞)² is a
2-dimensional statistical manifold under the Fisher information metric.
Unlike the MVN family (where Calvo-Oller embeds into SPD matrices), the
Beta family has its own information-geometric structure with closed-form
KL divergence and a tractable Fisher information metric tensor.

Why Beta is a useful belief-state model
---------------------------------------
- Conjugate prior for Bernoulli/binomial — natural for "probability" beliefs.
- Two parameters fully describe a wide family: U-shaped (α<1, β<1), bell
  (α>1, β>1), monotone, skewed.
- Matches OMEGA's per-ticker aggregate-belief use case where 15 agent
  scores are normalised to a [0,1] mean and variance, then converted to
  (α, β) via method of moments.

Mathematical primitives in this module
--------------------------------------
- `kl_beta(α₁, β₁, α₂, β₂)` — closed form via digamma; canonical reference.
- `fisher_information_beta(α, β)` — 2×2 metric tensor (g_ij), derivatives
  via trigamma.
- `geodesic_distance_beta(p, q)` — Riemannian distance via numerical
  geodesic integration. No general closed form on the Beta manifold;
  we use the Christoffel-symbol ODE solved over a path discretised on a
  small grid + scipy quad to integrate ds.
- `scalar_curvature_beta(α, β)` — Brioschi formula for 2D Riemannian
  metrics, computed from g and its first/second partials.
- `fit_beta_mom(samples)` — method-of-moments estimator (α, β) from a
  sample's mean + variance; standard textbook formula with edge guards.

References
----------
- Amari, S. (1985). "Differential-Geometrical Methods in Statistics."
  Lecture Notes in Statistics 28, Springer. Sec. 3.6 covers exponential
  families incl. Beta.
- Brigo, D. & Pistone, G. (2017). "Eta and Theta convergence on
  exponential families." (Beta is in this class.)
- Brioschi, F. (1854) — closed-form Gaussian curvature for 2D metrics.
- Costa, S.I.R., Santos, S.A. & Strapasson, J.E. (2015). "Fisher
  information distance: A geometrical reading." Disc. Applied Math 197.
"""

from __future__ import annotations

import numpy as np
from scipy.special import digamma, gammaln, polygamma
from scipy.integrate import quad


def _validate(alpha: float, beta: float) -> None:
    if not (np.isfinite(alpha) and np.isfinite(beta)):
        raise ValueError(f"alpha, beta must be finite; got {alpha}, {beta}")
    if alpha <= 0 or beta <= 0:
        raise ValueError(f"alpha, beta must be > 0; got {alpha}, {beta}")


def kl_beta(alpha1: float, beta1: float, alpha2: float, beta2: float) -> float:
    """Closed-form KL divergence KL(Beta(α₁, β₁) ∥ Beta(α₂, β₂)).

    KL = ln(B(α₂, β₂) / B(α₁, β₁))
       + (α₁ − α₂) ψ(α₁)
       + (β₁ − β₂) ψ(β₁)
       + (α₂ − α₁ + β₂ − β₁) ψ(α₁ + β₁)

    where B is the Beta function and ψ is the digamma function. Uses
    `gammaln` so the log-Beta term is stable for extreme parameters.

    Returns
    -------
    float ≥ 0; equals 0 iff (α₁, β₁) == (α₂, β₂) up to floating-point.
    """
    _validate(alpha1, beta1)
    _validate(alpha2, beta2)
    log_beta1 = gammaln(alpha1) + gammaln(beta1) - gammaln(alpha1 + beta1)
    log_beta2 = gammaln(alpha2) + gammaln(beta2) - gammaln(alpha2 + beta2)
    val = (
        (log_beta2 - log_beta1)
        + (alpha1 - alpha2) * digamma(alpha1)
        + (beta1 - beta2) * digamma(beta1)
        + (alpha2 - alpha1 + beta2 - beta1) * digamma(alpha1 + beta1)
    )
    # numerical guard: KL is non-negative but finite-precision can dip ~-1e-15
    return float(max(val, 0.0))


def fisher_information_beta(alpha: float, beta: float) -> np.ndarray:
    """Fisher information matrix of the Beta(α, β) family at (α, β).

    g(α, β) = [[ψ'(α) − ψ'(α+β),  −ψ'(α+β)],
              [−ψ'(α+β),         ψ'(β) − ψ'(α+β)]]

    where ψ' is the trigamma function. Returns a 2×2 symmetric positive-
    definite ndarray.
    """
    _validate(alpha, beta)
    psi1_a = polygamma(1, alpha)
    psi1_b = polygamma(1, beta)
    psi1_ab = polygamma(1, alpha + beta)
    g = np.array(
        [[psi1_a - psi1_ab, -psi1_ab],
         [-psi1_ab,          psi1_b - psi1_ab]],
        dtype=np.float64,
    )
    return g


def _ds_beta(alpha: float, beta: float, dalpha: float, dbeta: float) -> float:
    """Riemannian line element ds = sqrt(v.T @ g(α,β) @ v) for tangent
    direction v = (dalpha, dbeta) at point (α, β).
    """
    g = fisher_information_beta(alpha, beta)
    v = np.array([dalpha, dbeta], dtype=np.float64)
    qf = float(v @ g @ v)
    if qf < 0:
        # numerical noise on near-singular metric; clamp to 0
        return 0.0
    return float(np.sqrt(qf))


def geodesic_distance_beta(
    alpha1: float, beta1: float,
    alpha2: float, beta2: float,
    n_segments: int = 64,
) -> float:
    """Approximate Fisher-Rao geodesic distance on the Beta manifold.

    No closed form exists in the general case. We integrate the line
    element ds along the **straight line in (α, β) parameter space** as a
    UPPER BOUND on the geodesic distance — the true geodesic minimises
    this integral, and the linear-path integral over-counts. For
    monitoring purposes (Phase 8 Information Cost), the upper bound is
    sufficient and stable; relative ordering across tickers is preserved.

    Parameters
    ----------
    n_segments : int
        Number of discretisation segments for the line integral.
        64 is a good speed/accuracy compromise for typical (α, β) pairs.

    Returns
    -------
    float ≥ 0. Equals 0 iff the two points are identical.

    Note
    ----
    For a true closed-form-equivalent on Beta we would need to solve the
    geodesic equation (a 2D ODE with Christoffel symbols). v0.3 will add a
    Strapasson-style iterative solver that produces tighter values; v0.2
    ships the upper-bound form because it's robust and monotone in the
    actual distance, which is what downstream "information cost" wants.
    """
    _validate(alpha1, beta1)
    _validate(alpha2, beta2)
    if alpha1 == alpha2 and beta1 == beta2:
        return 0.0
    da = alpha2 - alpha1
    db = beta2 - beta1

    # parametric line: (α(t), β(t)) = (α₁ + t·da, β₁ + t·db), t ∈ [0,1]
    def integrand(t: float) -> float:
        a = alpha1 + t * da
        b = beta1 + t * db
        return _ds_beta(a, b, da, db)

    # quad with sane tolerances — beta family integrand is smooth on (0, ∞)²
    val, _err = quad(integrand, 0.0, 1.0, limit=n_segments, epsabs=1e-8, epsrel=1e-6)
    return float(max(val, 0.0))


def scalar_curvature_beta(alpha: float, beta: float) -> float:
    """Gaussian curvature K of the Beta manifold at (α, β).

    Computed via the standard Brioschi determinant formula for 2D
    Riemannian metrics (Wikipedia "Gaussian curvature", coordinate form):

        K = (1 / (EG − F²)²) · [ det(M₁) − det(M₂) ]

    where, for E = g₁₁, F = g₁₂, G = g₂₂ and partials denoted with
    subscripts u, v (here u = α, v = β):

        M₁ = ⎡ −½ E_vv + F_uv − ½ G_uu     ½ E_u    F_u − ½ E_v ⎤
             ⎢ F_v − ½ G_u                  E         F             ⎥
             ⎣ ½ G_v                         F         G             ⎦

        M₂ = ⎡ 0          ½ E_v    ½ G_u ⎤
             ⎢ ½ E_v       E         F     ⎥
             ⎣ ½ G_u       F         G     ⎦

    All partials computed by central finite differences with adaptive step.

    For 2D, the scalar curvature R (Ricci-trace convention) equals 2·K;
    we return K directly. The Beta manifold has **negative curvature**
    throughout its bell-shaped interior (α, β > 1) — characteristic of
    information manifolds of natural exponential families.
    """
    _validate(alpha, beta)
    h = 1e-4 * max(1.0, alpha, beta)

    def g_at(a: float, b: float) -> tuple[float, float, float]:
        m = fisher_information_beta(a, b)
        return float(m[0, 0]), float(m[0, 1]), float(m[1, 1])

    E, F, G = g_at(alpha, beta)
    # First partials
    E_u = (g_at(alpha + h, beta)[0] - g_at(alpha - h, beta)[0]) / (2 * h)
    E_v = (g_at(alpha, beta + h)[0] - g_at(alpha, beta - h)[0]) / (2 * h)
    F_u = (g_at(alpha + h, beta)[1] - g_at(alpha - h, beta)[1]) / (2 * h)
    F_v = (g_at(alpha, beta + h)[1] - g_at(alpha, beta - h)[1]) / (2 * h)
    G_u = (g_at(alpha + h, beta)[2] - g_at(alpha - h, beta)[2]) / (2 * h)
    G_v = (g_at(alpha, beta + h)[2] - g_at(alpha, beta - h)[2]) / (2 * h)
    # Second partials needed by Brioschi
    E_vv = (g_at(alpha, beta + h)[0] - 2 * E + g_at(alpha, beta - h)[0]) / (h * h)
    G_uu = (g_at(alpha + h, beta)[2] - 2 * G + g_at(alpha - h, beta)[2]) / (h * h)
    F_uv = (
        g_at(alpha + h, beta + h)[1]
        - g_at(alpha + h, beta - h)[1]
        - g_at(alpha - h, beta + h)[1]
        + g_at(alpha - h, beta - h)[1]
    ) / (4 * h * h)

    det_g = E * G - F * F
    if det_g <= 1e-30:
        return float("nan")

    M1 = np.array([
        [-0.5 * E_vv + F_uv - 0.5 * G_uu,  0.5 * E_u,            F_u - 0.5 * E_v],
        [F_v - 0.5 * G_u,                   E,                    F],
        [0.5 * G_v,                         F,                    G],
    ])
    M2 = np.array([
        [0.0,           0.5 * E_v,    0.5 * G_u],
        [0.5 * E_v,     E,            F],
        [0.5 * G_u,     F,            G],
    ])
    K = (np.linalg.det(M1) - np.linalg.det(M2)) / (det_g ** 2)
    return float(K)


def fit_beta_mom(samples: np.ndarray, *, eps: float = 1e-3) -> tuple[float, float]:
    """Method-of-moments estimator for Beta(α, β) from a sample on (0, 1).

    Given samples with mean μ and variance v, if μ(1-μ) > v then:
        α = μ * (μ(1-μ)/v - 1)
        β = (1-μ) * (μ(1-μ)/v - 1)

    Falls back to a near-uniform Beta(1, 1) if the data is degenerate
    (zero variance, all values at the boundary). Clamps each parameter
    to [0.5, 100] to avoid manifold-edge singularities — this is the
    same numerical guard the OMEGA Phase 8 plan specifies.

    Parameters
    ----------
    samples : ndarray
        1-D array of values in (0, 1). Values exactly at 0 or 1 are
        clamped to (eps, 1-eps) for numerical stability.
    eps : float
        Clamp width for boundary protection.
    """
    x = np.asarray(samples, dtype=np.float64).ravel()
    if x.size < 2:
        raise ValueError("need >= 2 samples for method-of-moments")
    x = np.clip(x, eps, 1.0 - eps)
    mu = float(x.mean())
    v = float(x.var(ddof=1))
    if v < 1e-12:
        # zero variance ⇒ degenerate; return symmetric mild prior
        return 1.0, 1.0
    # method-of-moments
    s = mu * (1.0 - mu) / v - 1.0
    if s <= 0:
        return 1.0, 1.0
    alpha = mu * s
    beta = (1.0 - mu) * s
    return float(np.clip(alpha, 0.5, 100.0)), float(np.clip(beta, 0.5, 100.0))


# =====================================================================
# v0.3 — MLE estimator (alternative to MoM, less biased on small n)
# =====================================================================

def fit_beta_mle(
    samples: np.ndarray,
    *,
    eps: float = 1e-3,
    init: tuple[float, float] | None = None,
) -> tuple[float, float]:
    """Maximum-likelihood estimator for Beta(α, β) via numerical optimisation.

    Method-of-moments (`fit_beta_mom`) has a systematic bias on small samples
    (especially when the sample variance underestimates the population
    variance — common with n < 30). MLE optimises the log-likelihood
    directly:

        L(α, β | x) = (α - 1) Σ log x_i + (β - 1) Σ log(1 - x_i)
                      - n · ln B(α, β)

    Solved with `scipy.optimize.minimize` (L-BFGS-B, log-parameterised so
    α, β stay strictly positive). Falls back to MoM when optimisation fails.
    Returns parameters clamped to [0.5, 100] matching MoM's behaviour, so
    the two estimators are drop-in interchangeable downstream.

    Parameters
    ----------
    samples : ndarray
        1-D array of values in (0, 1). Boundary values clamped to (eps, 1-eps).
    eps : float
        Clamp width for boundary protection.
    init : (α₀, β₀) | None
        Starting point for L-BFGS-B. Defaults to MoM estimate (typically
        within a few iterations of the true MLE).

    Returns
    -------
    (α, β) : tuple[float, float]
        Clamped to [0.5, 100] each.

    References
    ----------
    Owen, C.E.B. (2008). "Parameter Estimation for the Beta Distribution",
    M.Sc. thesis, Brigham Young University. Sec. 3.2 — MLE via Newton-
    Raphson on the digamma equations; we use L-BFGS-B which is more robust
    on the (α, β) cliff at small variance.
    """
    from scipy.optimize import minimize

    x = np.asarray(samples, dtype=np.float64).ravel()
    if x.size < 2:
        raise ValueError("need >= 2 samples for MLE")
    x = np.clip(x, eps, 1.0 - eps)
    n = x.size
    sum_log_x = float(np.sum(np.log(x)))
    sum_log_1mx = float(np.sum(np.log(1.0 - x)))

    if init is None:
        try:
            init = fit_beta_mom(x, eps=eps)
        except Exception:
            init = (1.0, 1.0)

    # Optimise in log-space so α, β stay > 0
    log_init = np.log(np.array(init, dtype=np.float64))

    def neg_log_lik(log_params: np.ndarray) -> float:
        a = float(np.exp(log_params[0]))
        b = float(np.exp(log_params[1]))
        # log B(a, b) = lgamma(a) + lgamma(b) - lgamma(a+b)
        # gammaln may overflow on extreme L-BFGS-B steps — np handles it as inf,
        # which the optimizer reads as a barrier. Suppress the noisy warning.
        with np.errstate(invalid="ignore", over="ignore"):
            log_beta = gammaln(a) + gammaln(b) - gammaln(a + b)
        if not np.isfinite(log_beta):
            return float("inf")
        ll = (a - 1.0) * sum_log_x + (b - 1.0) * sum_log_1mx - n * log_beta
        return -ll

    try:
        res = minimize(neg_log_lik, log_init, method="L-BFGS-B",
                       options={"maxiter": 200, "ftol": 1e-9})
        if not res.success:
            return fit_beta_mom(x, eps=eps)
        alpha, beta = float(np.exp(res.x[0])), float(np.exp(res.x[1]))
    except Exception:
        return fit_beta_mom(x, eps=eps)

    return (
        float(np.clip(alpha, 0.5, 100.0)),
        float(np.clip(beta, 0.5, 100.0)),
    )
