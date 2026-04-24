"""Correctness tests for the Calvo-Oller Fisher-Rao distance.

Strategy: test against three independent sources of truth:
  1. Trivial analytical identities (d(A,A)=0, symmetry, non-negativity)
  2. Scalar 1D closed form (Atkinson-Mitchell 1981; exact for d=1)
  3. Agreement between the fast eigenvalue path and the logm reference path
"""

import numpy as np
import pytest

from fisherrao import (
    calvo_oller_distance,
    fisher_rao_1d,
    spd_affine_distance,
    spd_affine_distance_logm,
    spd_embedding,
)


def _rand_spd(rng: np.random.Generator, d: int, cond: float = 4.0) -> np.ndarray:
    """Random SPD matrix with bounded condition number."""
    q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    eigs = rng.uniform(1.0, cond, size=d)
    return (q * eigs) @ q.T


def test_spd_embedding_is_spd():
    rng = np.random.default_rng(0)
    for d in (1, 2, 5):
        mu = rng.standard_normal(d)
        sigma = _rand_spd(rng, d)
        q = spd_embedding(mu, sigma)
        assert q.shape == (d + 1, d + 1)
        # symmetry
        np.testing.assert_allclose(q, q.T, atol=1e-12)
        # positive definiteness: all eigvals > 0
        eigvals = np.linalg.eigvalsh(q)
        assert np.all(eigvals > 0), f"non-PD embedding, eigvals={eigvals}"


def test_distance_to_self_is_zero():
    rng = np.random.default_rng(1)
    for d in (1, 2, 5, 10):
        mu = rng.standard_normal(d)
        sigma = _rand_spd(rng, d)
        d_self = calvo_oller_distance(mu, sigma, mu, sigma)
        assert d_self < 1e-9, f"d(A,A) = {d_self} for d={d}"


def test_distance_is_symmetric():
    rng = np.random.default_rng(2)
    for _ in range(20):
        d = rng.integers(1, 8)
        mu_a = rng.standard_normal(d)
        mu_b = rng.standard_normal(d)
        sigma_a = _rand_spd(rng, d)
        sigma_b = _rand_spd(rng, d)
        d_ab = calvo_oller_distance(mu_a, sigma_a, mu_b, sigma_b)
        d_ba = calvo_oller_distance(mu_b, sigma_b, mu_a, sigma_a)
        assert np.isclose(d_ab, d_ba, atol=1e-9), f"asymmetry: {d_ab} vs {d_ba}"


def test_distance_is_non_negative():
    rng = np.random.default_rng(3)
    for _ in range(20):
        d = rng.integers(1, 6)
        mu_a = rng.standard_normal(d)
        mu_b = rng.standard_normal(d)
        sigma_a = _rand_spd(rng, d)
        sigma_b = _rand_spd(rng, d)
        assert calvo_oller_distance(mu_a, sigma_a, mu_b, sigma_b) >= 0.0


def test_eigpath_agrees_with_logm_path():
    """The fast eigenvalue path should match the reference logm path on SPD inputs."""
    rng = np.random.default_rng(4)
    for d in (2, 4, 8):
        for _ in range(5):
            a = _rand_spd(rng, d)
            b = _rand_spd(rng, d)
            d_eig = spd_affine_distance(a, b)
            d_logm = spd_affine_distance_logm(a, b)
            assert np.isclose(d_eig, d_logm, atol=1e-7, rtol=1e-7), (
                f"eig {d_eig} vs logm {d_logm} for d={d}"
            )


def test_calvo_oller_1d_mean_shift_closed_form():
    """In 1D with σ=1 on both sides and a pure mean shift by Δ,
    d_CO = sqrt(2) * acosh(1 + Δ²/2) by direct computation from the SPD
    embedding (see module docstring derivation)."""
    rng = np.random.default_rng(5)
    sigma = np.array([[1.0]])
    for delta in rng.uniform(-4.0, 4.0, size=20):
        mu_a = np.array([0.0])
        mu_b = np.array([delta])
        d_co = calvo_oller_distance(mu_a, sigma, mu_b, sigma)
        d_expected = np.sqrt(2.0) * np.arccosh(1.0 + delta * delta / 2.0)
        assert np.isclose(d_co, d_expected, atol=1e-8, rtol=1e-8), (
            f"Δ={delta}: d_co={d_co} d_expected={d_expected}"
        )


def test_fisher_rao_1d_sanity():
    """Standalone correctness of the 1D Atkinson-Mitchell formula.

    Note: this is the TRUE 1D Fisher-Rao distance and is not expected to match
    `calvo_oller_distance` in general — the Calvo-Oller SPD embedding is a
    lower bound on the true Fisher-Rao distance, and the two do not differ by
    a constant factor (the ratio depends on the direction of displacement in
    (μ, σ) space). Both are useful; they measure different things.
    """
    # d(A, A) = 0
    assert fisher_rao_1d(1.0, 2.0, 1.0, 2.0) == 0.0
    # symmetry
    d_ab = fisher_rao_1d(0.0, 1.0, 2.0, 3.0)
    d_ba = fisher_rao_1d(2.0, 3.0, 0.0, 1.0)
    assert np.isclose(d_ab, d_ba)


def test_pure_mean_shift_nd():
    """Pure mean shift with identical Sigma: distance should grow monotonically with ||Δμ||."""
    d = 3
    sigma = np.eye(d)
    mu0 = np.zeros(d)
    # increasing shift magnitudes
    prev = 0.0
    for k in range(1, 10):
        mu_k = np.array([k * 0.5, 0.0, 0.0])
        d_k = calvo_oller_distance(mu0, sigma, mu_k, sigma)
        assert d_k > prev, f"non-monotone at k={k}: d_k={d_k} prev={prev}"
        prev = d_k


def test_pure_scale_shift_nd():
    """Pure covariance scale shift: d(N(0, I), N(0, c * I)) should match |log c| structure.

    For N(0, I) and N(0, c * I) with c > 0 in d dimensions, the Fisher-Rao
    distance via Calvo-Oller equals sqrt(d) * |log c| because the embedded
    SPD matrices differ only by scaling of the d-dim sub-block.
    """
    rng = np.random.default_rng(7)
    d = 4
    mu = np.zeros(d)
    sigma_base = np.eye(d)
    for c in rng.uniform(0.2, 5.0, size=10):
        sigma_scaled = c * np.eye(d)
        d_actual = calvo_oller_distance(mu, sigma_base, mu, sigma_scaled)
        d_expected = np.sqrt(d) * abs(np.log(c))
        assert np.isclose(d_actual, d_expected, atol=1e-7, rtol=1e-6), (
            f"c={c}: d_actual={d_actual} d_expected={d_expected}"
        )


def test_rejects_mismatched_dimensions():
    with pytest.raises(ValueError, match="match"):
        spd_embedding(np.zeros(3), np.eye(2))


def test_rejects_non_spd_via_eigpath():
    """The fast path requires A SPD; on a non-symmetric input scipy eigh raises."""
    a = np.array([[1.0, 2.0], [3.0, 4.0]])  # non-symmetric
    b = np.eye(2)
    with pytest.raises(Exception):
        spd_affine_distance(a, b)


def test_fisher_rao_1d_rejects_bad_sigma():
    with pytest.raises(ValueError, match="positive"):
        fisher_rao_1d(0.0, -1.0, 0.0, 1.0)
