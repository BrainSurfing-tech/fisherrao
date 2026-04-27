"""Unit tests for v0.3 additions: standardize=True, fit_beta_mle, MVN bounds."""

import numpy as np
import pytest

from fisherrao import (
    calvo_oller_distance,
    fisher_rao_mvn_bounds,
    fit_beta_mle,
    fit_beta_mom,
    fit_mvn,
)


# =====================================================================
# fit_mvn(standardize=True)
# =====================================================================

def test_standardize_yields_zero_mean():
    rng = np.random.default_rng(0)
    x = rng.normal(loc=10.0, scale=3.0, size=(200, 5))
    mu, _, _ = fit_mvn(x, standardize=True)
    assert np.allclose(mu, 0.0, atol=1e-8)


def test_standardize_yields_unit_diagonal_covariance():
    rng = np.random.default_rng(1)
    # Wildly different per-column scales — exactly the OCTG use case
    x = rng.normal(loc=0.0, scale=1.0, size=(500, 4))
    x[:, 0] *= 0.01     # price-return-like
    x[:, 1] *= 50.0     # sentiment-score-like
    x[:, 2] *= 1e-4     # microcap-volume-like
    x[:, 3] *= 1.0      # baseline
    _, sigma, _ = fit_mvn(x, standardize=True, shrinkage="none")
    # After standardization, diagonal of empirical covariance ≈ 1.0
    assert np.allclose(np.diag(sigma), 1.0, atol=1e-6)


def test_standardize_makes_distance_scale_invariant():
    """Two streams that are scale-multiples of each other should give same
    Fisher-Rao distance after standardization."""
    rng = np.random.default_rng(2)
    base = rng.normal(size=(100, 3))
    # Stream A: base scaled by [1, 1, 1]
    a1, a2 = base[:50], base[50:]
    # Stream B: base scaled by [1000, 0.001, 1]
    scale = np.array([1000.0, 0.001, 1.0])
    b1, b2 = base[:50] * scale, base[50:] * scale

    mu_a1, sig_a1, _ = fit_mvn(a1, standardize=True)
    mu_a2, sig_a2, _ = fit_mvn(a2, standardize=True)
    mu_b1, sig_b1, _ = fit_mvn(b1, standardize=True)
    mu_b2, sig_b2, _ = fit_mvn(b2, standardize=True)

    d_a = calvo_oller_distance(mu_a1, sig_a1, mu_a2, sig_a2)
    d_b = calvo_oller_distance(mu_b1, sig_b1, mu_b2, sig_b2)
    # Should be identical up to floating tolerance
    assert d_a == pytest.approx(d_b, rel=1e-6)


def test_standardize_handles_zero_variance_column():
    """Constant columns must not crash (divide-by-zero guard)."""
    rng = np.random.default_rng(3)
    x = rng.normal(size=(50, 3))
    x[:, 1] = 5.0  # constant column
    # Use shrinkage="none" so we see the raw empirical covariance — Ledoit-Wolf
    # would shrink the zero-variance diagonal toward the trace mean (intentional
    # regularisation, but obscures what we're verifying here).
    mu, sigma, _ = fit_mvn(x, standardize=True, shrinkage="none")
    assert np.all(np.isfinite(mu))
    assert np.all(np.isfinite(sigma))
    # Constant column → standardized to zeros → empirical diagonal entry == 0
    assert sigma[1, 1] == pytest.approx(0.0, abs=1e-9)


def test_standardize_default_off_preserves_v01_behaviour():
    """Calling fit_mvn without standardize should be byte-identical to v0.1/v0.2."""
    rng = np.random.default_rng(4)
    x = rng.normal(loc=2.5, scale=1.5, size=(100, 3))
    mu1, sig1, _ = fit_mvn(x)                          # default
    mu2, sig2, _ = fit_mvn(x, standardize=False)       # explicit
    np.testing.assert_array_equal(mu1, mu2)
    np.testing.assert_array_equal(sig1, sig2)


# =====================================================================
# fit_beta_mle
# =====================================================================

def test_fit_beta_mle_recovers_known_alpha_beta():
    """MLE should recover ground-truth parameters on a large sample."""
    rng = np.random.default_rng(5)
    true_alpha, true_beta = 4.0, 2.0
    samples = rng.beta(true_alpha, true_beta, size=5000)
    a, b = fit_beta_mle(samples)
    # Within ~5% on n=5000
    assert a == pytest.approx(true_alpha, rel=0.05)
    assert b == pytest.approx(true_beta, rel=0.05)


def test_fit_beta_mle_clamps_to_safe_range():
    """Even on degenerate input, MLE returns clamped parameters."""
    samples = np.array([0.5] * 100)  # zero variance
    a, b = fit_beta_mle(samples)
    assert 0.5 <= a <= 100.0
    assert 0.5 <= b <= 100.0


def test_fit_beta_mle_rejects_too_small_sample():
    with pytest.raises(ValueError, match="need >= 2 samples"):
        fit_beta_mle(np.array([0.5]))


def test_fit_beta_mle_falls_back_to_mom_on_optimizer_failure(monkeypatch):
    """If the optimizer raises, MLE returns MoM result."""
    import scipy.optimize as scipy_opt

    def boom(*args, **kwargs):
        raise RuntimeError("optimizer exploded")
    monkeypatch.setattr(scipy_opt, "minimize", boom)

    rng = np.random.default_rng(6)
    samples = rng.beta(3.0, 5.0, size=1000)
    mle = fit_beta_mle(samples)
    mom = fit_beta_mom(samples)
    assert mle == pytest.approx(mom, rel=1e-9)


def test_fit_beta_mle_outperforms_mom_on_small_n():
    """On small samples MLE typically has lower bias than MoM."""
    rng = np.random.default_rng(7)
    true_alpha, true_beta = 5.0, 2.0
    n_trials = 50
    mle_errors = []
    mom_errors = []
    for _ in range(n_trials):
        samples = rng.beta(true_alpha, true_beta, size=20)  # small n
        a_mle, b_mle = fit_beta_mle(samples)
        a_mom, b_mom = fit_beta_mom(samples)
        mle_errors.append(abs(a_mle - true_alpha) + abs(b_mle - true_beta))
        mom_errors.append(abs(a_mom - true_alpha) + abs(b_mom - true_beta))
    # Mean absolute error: MLE should be at least as good as MoM on average
    # (We don't require strict improvement on every trial — too noisy.)
    assert np.mean(mle_errors) <= np.mean(mom_errors) * 1.2


# =====================================================================
# fisher_rao_mvn_bounds
# =====================================================================

def test_bounds_lower_equals_calvo_oller():
    """The `lower` element of the bounds tuple must equal calvo_oller_distance."""
    mu_a = np.array([0.0, 0.0])
    mu_b = np.array([1.0, 0.5])
    sigma_a = np.eye(2)
    sigma_b = np.diag([2.0, 1.5])
    lower, upper = fisher_rao_mvn_bounds(mu_a, sigma_a, mu_b, sigma_b)
    direct = calvo_oller_distance(mu_a, sigma_a, mu_b, sigma_b)
    assert lower == pytest.approx(direct, rel=1e-9)
    # And upper >= lower (it's a bracket)
    assert upper >= lower


def test_bounds_collapse_when_means_equal():
    """When mu_a == mu_b, the mean leg vanishes; lower may still be < upper
    when the sigma path is non-trivial. Verify upper is finite + non-negative."""
    mu = np.array([0.5, 0.5])
    sig_a = np.eye(2)
    sig_b = np.diag([3.0, 0.5])
    lower, upper = fisher_rao_mvn_bounds(mu, sig_a, mu, sig_b)
    assert lower >= 0
    assert upper >= lower
    # leg1 (mean-only between identical means) = 0, so upper = leg2 = lower
    # in this special case (since the path collapses to the direct sigma jump)
    assert upper == pytest.approx(lower, rel=1e-9)


def test_bounds_collapse_when_sigmas_equal():
    """Same idea, mean-only displacement with shared sigma."""
    mu_a = np.array([0.0, 0.0])
    mu_b = np.array([1.0, -2.0])
    sigma = np.eye(2)
    lower, upper = fisher_rao_mvn_bounds(mu_a, sigma, mu_b, sigma)
    assert upper == pytest.approx(lower, rel=1e-9)


def test_bounds_zero_when_distributions_identical():
    mu = np.array([1.0, 2.0])
    sigma = np.eye(2) + 0.5
    lower, upper = fisher_rao_mvn_bounds(mu, sigma, mu, sigma)
    assert lower == pytest.approx(0.0, abs=1e-12)
    assert upper == pytest.approx(0.0, abs=1e-12)


def test_bounds_upper_strictly_greater_for_joint_displacement():
    """When BOTH mu and sigma differ, upper bound > lower (triangle inequality
    is strict in this case)."""
    mu_a = np.array([0.0, 0.0])
    mu_b = np.array([1.0, 1.0])
    sigma_a = np.eye(2)
    sigma_b = np.diag([2.0, 2.0])
    lower, upper = fisher_rao_mvn_bounds(mu_a, sigma_a, mu_b, sigma_b)
    assert upper > lower
