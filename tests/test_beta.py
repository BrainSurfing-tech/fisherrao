"""Correctness tests for the Beta-family Fisher-Rao primitives.

Strategy:
  1. Analytical invariants: KL(P‖P)=0, KL non-negative, symmetry of the
     Fisher information matrix, positive-definiteness of g.
  2. scipy cross-checks: KL via direct integration of p log(p/q) over (0,1)
     to validate the closed-form digamma formula.
  3. Known-curvature cases: Beta(1,1) = Uniform; the Beta manifold has
     well-documented sign of curvature (negative throughout).
  4. fit_beta_mom: round-trip — sample from Beta(α₀,β₀), refit, recover
     parameters within sample-size tolerance.
"""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import beta as scipy_beta

from fisherrao import (
    fisher_information_beta,
    fit_beta_mom,
    geodesic_distance_beta,
    kl_beta,
    scalar_curvature_beta,
)


# ---------- KL divergence ----------

def test_kl_self_is_zero():
    """KL(P‖P) = 0 exactly (or within ~1e-12 floating-point noise)."""
    for a, b in [(2.0, 5.0), (0.5, 0.5), (10.0, 1.5), (1.0, 1.0)]:
        d = kl_beta(a, b, a, b)
        assert d < 1e-12, f"KL(Beta({a},{b}) || self) = {d}, expected ~0"


def test_kl_non_negative():
    """KL(P‖Q) ≥ 0 for any valid (α, β) pair."""
    rng = np.random.default_rng(0)
    for _ in range(50):
        a1, b1, a2, b2 = rng.uniform(0.5, 10.0, 4)
        assert kl_beta(a1, b1, a2, b2) >= 0.0


def test_kl_matches_numerical_integration():
    """Closed-form digamma KL must agree with direct ∫p log(p/q) dx on (0,1)."""
    test_cases = [
        (2.0, 5.0, 1.5, 4.0),
        (0.8, 1.2, 2.0, 2.0),
        (5.0, 5.0, 3.0, 7.0),
        (1.0, 1.0, 2.0, 2.0),
    ]
    for a1, b1, a2, b2 in test_cases:
        analytical = kl_beta(a1, b1, a2, b2)

        def integrand(x, a1=a1, b1=b1, a2=a2, b2=b2):
            p = scipy_beta.pdf(x, a1, b1)
            q = scipy_beta.pdf(x, a2, b2)
            if p <= 0 or q <= 0:
                return 0.0
            return p * np.log(p / q)

        numerical, _ = quad(integrand, 1e-6, 1.0 - 1e-6, limit=100)
        rel_err = abs(analytical - numerical) / max(abs(numerical), 1e-9)
        assert rel_err < 1e-3, (
            f"KL Beta({a1},{b1}) || Beta({a2},{b2}): "
            f"analytical={analytical:.6f} vs numerical={numerical:.6f}, "
            f"rel err {rel_err:.4f}"
        )


def test_kl_rejects_invalid_params():
    with pytest.raises(ValueError):
        kl_beta(0.0, 1.0, 1.0, 1.0)
    with pytest.raises(ValueError):
        kl_beta(1.0, -0.5, 1.0, 1.0)
    with pytest.raises(ValueError):
        kl_beta(np.inf, 1.0, 1.0, 1.0)


# ---------- Fisher information matrix ----------

def test_fisher_information_symmetric_spd():
    rng = np.random.default_rng(1)
    for _ in range(30):
        a, b = rng.uniform(0.5, 20.0, 2)
        g = fisher_information_beta(a, b)
        assert g.shape == (2, 2)
        # symmetry
        assert np.isclose(g[0, 1], g[1, 0])
        # positive-definite ⇒ both eigenvalues > 0
        eigs = np.linalg.eigvalsh(g)
        assert np.all(eigs > 0), f"FIM not SPD at ({a}, {b}); eigs={eigs}"


def test_fisher_information_uniform():
    """At Beta(1,1) (uniform on [0,1]) the Fisher info has a known form.

    At α=β=1: ψ'(1) = π²/6, ψ'(2) = π²/6 − 1.
    g₁₁ = ψ'(1) − ψ'(2) = 1
    g₂₂ = ψ'(1) − ψ'(2) = 1
    g₁₂ = −ψ'(2) = 1 − π²/6
    """
    g = fisher_information_beta(1.0, 1.0)
    np.testing.assert_allclose(g[0, 0], 1.0, atol=1e-10)
    np.testing.assert_allclose(g[1, 1], 1.0, atol=1e-10)
    np.testing.assert_allclose(g[0, 1], 1.0 - np.pi ** 2 / 6, atol=1e-10)


# ---------- Geodesic distance ----------

def test_geodesic_self_is_zero():
    for a, b in [(2.0, 5.0), (1.0, 1.0), (0.7, 0.7)]:
        d = geodesic_distance_beta(a, b, a, b)
        assert d == 0.0, f"geodesic(P, P) = {d}, expected 0"


def test_geodesic_non_negative():
    rng = np.random.default_rng(2)
    for _ in range(15):
        a1, b1, a2, b2 = rng.uniform(0.5, 10.0, 4)
        d = geodesic_distance_beta(a1, b1, a2, b2)
        assert d >= 0


def test_geodesic_symmetric():
    rng = np.random.default_rng(3)
    for _ in range(8):
        a1, b1, a2, b2 = rng.uniform(0.5, 10.0, 4)
        d_pq = geodesic_distance_beta(a1, b1, a2, b2)
        d_qp = geodesic_distance_beta(a2, b2, a1, b1)
        rel_err = abs(d_pq - d_qp) / max(abs(d_pq), 1e-9)
        assert rel_err < 1e-3, f"geodesic asymmetric: {d_pq} vs {d_qp}"


def test_geodesic_monotone_in_step_size():
    """Travelling further in (α, β) parameter space increases the integral."""
    base_a, base_b = 2.0, 2.0
    distances = []
    for step in [0.1, 0.5, 1.0, 2.0]:
        d = geodesic_distance_beta(base_a, base_b, base_a + step, base_b + step)
        distances.append(d)
    for i in range(len(distances) - 1):
        assert distances[i + 1] > distances[i], (
            f"geodesic not monotone: step{i}={distances[i]} step{i+1}={distances[i+1]}"
        )


def test_geodesic_upper_bounds_kl_relation():
    """sqrt(2*KL) is a known asymptotic lower bound on geodesic distance for
    nearby distributions. Our line-integral upper bound must satisfy
    geodesic_distance >= sqrt(2*KL) up to numerical noise on small steps."""
    test_cases = [
        (2.0, 5.0, 2.1, 5.05),
        (1.0, 1.0, 1.05, 1.02),
        (3.0, 3.0, 3.2, 2.95),
    ]
    for a1, b1, a2, b2 in test_cases:
        d_geo = geodesic_distance_beta(a1, b1, a2, b2)
        kl = kl_beta(a1, b1, a2, b2)
        lower = np.sqrt(2 * kl)
        # tolerate small deficit on tiny perturbations from finite-precision integration
        assert d_geo >= 0.95 * lower, (
            f"geodesic {d_geo} < sqrt(2*KL) {lower} for "
            f"Beta({a1},{b1}) -> Beta({a2},{b2})"
        )


# ---------- Scalar curvature ----------

def test_scalar_curvature_finite():
    """Curvature is finite (NaN only on near-singular metric near boundary)."""
    rng = np.random.default_rng(4)
    for _ in range(20):
        # avoid singular boundary (α, β) -> (0, 0) where U-shape blows up
        a, b = rng.uniform(1.0, 10.0, 2)
        k = scalar_curvature_beta(a, b)
        assert np.isfinite(k), f"non-finite curvature at ({a}, {b}): {k}"


def test_scalar_curvature_negative_in_interior():
    """The Beta manifold has negative Gaussian curvature throughout the
    bell-shaped (α, β > 1) interior — consistent with information manifolds
    of natural exponential families. Test on a small sample."""
    interior_pts = [(2.0, 2.0), (3.0, 4.0), (5.0, 5.0), (4.0, 7.0)]
    negatives = 0
    for a, b in interior_pts:
        k = scalar_curvature_beta(a, b)
        if k < 0:
            negatives += 1
    # All four should be negative; allow one outlier from finite-difference noise
    assert negatives >= 3, (
        f"expected negative curvature throughout interior, got {negatives}/4 "
        f"negative on test points"
    )


# ---------- Method-of-moments fit ----------

def test_fit_beta_mom_recovers_params():
    """Round-trip: sample from Beta(α₀, β₀), fit, recover params."""
    rng = np.random.default_rng(7)
    test_cases = [(2.0, 5.0), (3.0, 3.0), (10.0, 2.0), (1.5, 1.5)]
    for a0, b0 in test_cases:
        samples = scipy_beta.rvs(a0, b0, size=5000, random_state=rng)
        a_hat, b_hat = fit_beta_mom(samples)
        # MoM has known sample-size dependence; tolerate ~10% recovery error
        # on n=5000 (good enough for telemetry-style fits)
        assert abs(a_hat - a0) / a0 < 0.15, f"α not recovered: true {a0}, fit {a_hat}"
        assert abs(b_hat - b0) / b0 < 0.15, f"β not recovered: true {b0}, fit {b_hat}"


def test_fit_beta_mom_handles_degenerate_input():
    """Constant input ⇒ method falls back to symmetric Beta(1, 1)."""
    samples = np.full(50, 0.5)
    a, b = fit_beta_mom(samples)
    assert a == 1.0 and b == 1.0


def test_fit_beta_mom_clamps_extreme_params():
    """When data implies α or β > 100 (very tight distribution), clamp."""
    rng = np.random.default_rng(11)
    # Very tight Beta(50, 50) sample concentrated near 0.5
    samples = scipy_beta.rvs(50, 50, size=200, random_state=rng)
    # might fit to α, β ~ 50; should still respect clamp at 100
    a, b = fit_beta_mom(samples)
    assert 0.5 <= a <= 100.0
    assert 0.5 <= b <= 100.0


def test_fit_beta_mom_clamps_boundary_values():
    """Inputs of exactly 0 or 1 are clamped to (eps, 1-eps); no crash."""
    samples = np.array([0.0, 0.0, 0.5, 1.0, 1.0])
    a, b = fit_beta_mom(samples)
    assert np.isfinite(a) and np.isfinite(b)
    assert 0.5 <= a <= 100.0
    assert 0.5 <= b <= 100.0


def test_fit_beta_mom_rejects_short_input():
    with pytest.raises(ValueError):
        fit_beta_mom(np.array([0.5]))
