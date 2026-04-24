"""Covariance estimation helpers for small-sample MVN fitting.

When n_samples <= d+1, the empirical covariance is singular and Fisher-Rao
distances blow up. Ledoit-Wolf shrinkage regularises toward a scaled identity
target with an analytically-derived shrinkage intensity that minimises MSE.

Reference
---------
Ledoit, O. & Wolf, M. (2004). "A well-conditioned estimator for large-
dimensional covariance matrices". J. Multivariate Analysis 88(2):365-411.
"""

from __future__ import annotations

import numpy as np


def ledoit_wolf_shrinkage(
    x: np.ndarray,
    assume_centered: bool = False,
) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf shrinkage of the sample covariance toward scaled identity.

    Pure-numpy implementation of the closed-form Ledoit-Wolf 2004 estimator.
    We avoid a hard dependency on scikit-learn while matching its result.

    Parameters
    ----------
    x : ndarray of shape (n_samples, d)
        Sample matrix. Each row is one observation.
    assume_centered : bool
        If True, skip mean-centering (assumes x already has zero mean).

    Returns
    -------
    sigma_shrunk : ndarray of shape (d, d)
        Shrunk covariance estimate.
    shrinkage : float
        Shrinkage intensity alpha in [0, 1]. sigma_shrunk = (1 - a) * S + a * F
        where S is the empirical covariance and F = mu * I with mu = tr(S) / d.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"x must be 2D (n_samples, d), got shape {x.shape}")
    n, d = x.shape
    if n < 2:
        raise ValueError(f"need >= 2 samples, got {n}")

    if not assume_centered:
        x = x - x.mean(axis=0, keepdims=True)

    # empirical covariance (MLE, 1/n not 1/(n-1) — matches Ledoit-Wolf 2004)
    s = (x.T @ x) / n

    # scaled-identity target F = mu * I where mu = tr(S) / d
    mu = float(np.trace(s) / d)
    f = mu * np.eye(d)

    # squared Frobenius norm of (S - F)
    diff = s - f
    d2 = float(np.sum(diff * diff))

    # pi-hat: sum over i,j of Var(s_ij)
    # Ledoit-Wolf 2004 eq. (9): pi_hat = (1/n) * sum_ij [ (1/n) sum_k (x_ki x_kj - s_ij)^2 ]
    x_sq = x * x
    pi_mat = (x_sq.T @ x_sq) / n - s * s
    pi_hat = float(np.sum(pi_mat))

    # gamma-hat: squared Frobenius distance from S to F (already = d2)
    gamma_hat = d2

    # kappa-hat: pi_hat / gamma_hat; guard denominator
    if gamma_hat <= 0:
        # S already equals F; no shrinkage needed
        return s, 0.0

    kappa = pi_hat / gamma_hat
    shrinkage = float(np.clip(kappa / n, 0.0, 1.0))

    sigma_shrunk = (1.0 - shrinkage) * s + shrinkage * f
    return sigma_shrunk, shrinkage


def fit_mvn(
    x: np.ndarray,
    shrinkage: str = "ledoit-wolf",
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit a multivariate Gaussian to a sample matrix with optional shrinkage.

    Parameters
    ----------
    x : ndarray of shape (n_samples, d)
    shrinkage : {"ledoit-wolf", "none"}
        "ledoit-wolf" (default) applies Ledoit-Wolf 2004; safer for small n.
        "none" returns the empirical MLE (may be singular).

    Returns
    -------
    mu : ndarray of shape (d,)
        Sample mean.
    sigma : ndarray of shape (d, d)
        Fitted covariance (shrunk if requested).
    shrinkage_intensity : float
        Applied shrinkage intensity in [0, 1]. 0 means no shrinkage.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"x must be 2D (n_samples, d), got shape {x.shape}")

    mu = x.mean(axis=0)
    centered = x - mu

    if shrinkage == "none":
        n = x.shape[0]
        sigma = (centered.T @ centered) / n
        return mu, sigma, 0.0
    if shrinkage == "ledoit-wolf":
        sigma, alpha = ledoit_wolf_shrinkage(centered, assume_centered=True)
        return mu, sigma, alpha
    raise ValueError(f"unknown shrinkage mode {shrinkage!r}")
