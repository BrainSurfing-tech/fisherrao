"""Tests for the MVN-trajectory velocity/acceleration helpers."""

import numpy as np
import pytest

from fisherrao import calvo_oller_distance, fit_mvn, trajectory_metrics


def test_single_step_has_no_velocity():
    steps = trajectory_metrics([(np.zeros(2), np.eye(2))])
    assert len(steps) == 1
    assert steps[0].velocity is None
    assert steps[0].acceleration is None


def test_two_step_trajectory():
    mu0 = np.zeros(3)
    mu1 = np.array([1.0, 0.0, 0.0])
    sigma = np.eye(3)
    steps = trajectory_metrics([(mu0, sigma), (mu1, sigma)], times=[0.0, 1.0])
    assert len(steps) == 2
    assert steps[0].velocity is None
    expected = calvo_oller_distance(mu0, sigma, mu1, sigma)
    assert np.isclose(steps[1].velocity, expected, atol=1e-9)
    # acceleration needs 3 points
    assert steps[1].acceleration is None


def test_constant_velocity_zero_acceleration():
    """Move along a straight line in the MVN mean space at constant rate."""
    sigma = np.eye(2)
    mvns = [(np.array([float(i), 0.0]), sigma) for i in range(5)]
    steps = trajectory_metrics(mvns, times=[0.0, 1.0, 2.0, 3.0, 4.0])
    # Not literally constant because Fisher-Rao is curved, but pairs with same
    # (μ_prev, μ_next, Σ) give equal velocity
    assert steps[1].velocity is not None
    for i in range(2, len(steps)):
        assert steps[i].velocity is not None
        assert np.isclose(steps[i].velocity, steps[1].velocity, atol=1e-9), (
            f"velocity drift at step {i}: {steps[i].velocity} vs {steps[1].velocity}"
        )
        # therefore acceleration should be ~0
        assert abs(steps[i].acceleration) < 1e-9


def test_times_must_be_strictly_increasing():
    with pytest.raises(ValueError, match="strictly increasing"):
        trajectory_metrics(
            [(np.zeros(2), np.eye(2)), (np.ones(2), np.eye(2))],
            times=[1.0, 1.0],
        )


def test_times_length_mismatch():
    with pytest.raises(ValueError, match="length"):
        trajectory_metrics(
            [(np.zeros(2), np.eye(2)), (np.ones(2), np.eye(2))],
            times=[0.0],
        )


def test_fit_mvn_recovers_ground_truth():
    rng = np.random.default_rng(11)
    d = 4
    mu_true = rng.standard_normal(d)
    a = rng.standard_normal((d, d))
    sigma_true = a @ a.T + np.eye(d)
    # 1000 samples — shrinkage should be small
    x = rng.multivariate_normal(mu_true, sigma_true, size=1000)
    mu_fit, sigma_fit, alpha = fit_mvn(x, shrinkage="ledoit-wolf")
    # mean within 0.2 RMS
    assert np.linalg.norm(mu_fit - mu_true) < 0.3
    # covariance within 15% relative Frobenius
    rel_err = np.linalg.norm(sigma_fit - sigma_true, ord="fro") / np.linalg.norm(
        sigma_true, ord="fro"
    )
    assert rel_err < 0.15, f"rel_err={rel_err}"
    # large n ⇒ small shrinkage
    assert alpha < 0.1, f"alpha={alpha} too large for n=1000"


def test_fit_mvn_small_sample_still_spd():
    """Tiny n where empirical cov would be rank-deficient ⇒ shrinkage must save it."""
    rng = np.random.default_rng(13)
    d = 8
    mu_true = np.zeros(d)
    sigma_true = np.eye(d)
    # only 5 samples, d=8 ⇒ empirical cov is rank 4 (after centering)
    x = rng.multivariate_normal(mu_true, sigma_true, size=5)
    mu_fit, sigma_fit, alpha = fit_mvn(x, shrinkage="ledoit-wolf")
    eigvals = np.linalg.eigvalsh(sigma_fit)
    assert np.all(eigvals > 0), f"sigma_fit not SPD: eigvals={eigvals}"
    # shrinkage should be strong when n < d
    assert alpha > 0.3, f"expected strong shrinkage, got alpha={alpha}"


def test_fit_mvn_no_shrinkage_mode():
    rng = np.random.default_rng(17)
    x = rng.standard_normal((50, 3))
    mu, sigma, alpha = fit_mvn(x, shrinkage="none")
    assert alpha == 0.0
    expected = (x - x.mean(0)).T @ (x - x.mean(0)) / 50
    np.testing.assert_allclose(sigma, expected, atol=1e-9)
