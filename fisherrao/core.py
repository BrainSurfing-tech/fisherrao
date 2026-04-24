"""Fisher-Rao geodesic distance on the multivariate Gaussian (MVN) manifold.

The statistical manifold of d-variate Gaussians N(mu, Sigma) with mu in R^d
and Sigma symmetric positive definite is a (d + d(d+1)/2)-dimensional Riemannian
manifold when equipped with the Fisher information metric.

There is no universally-accepted closed-form Fisher-Rao geodesic distance on
this joint (mu, Sigma) manifold; most production work uses the **Calvo-Oller
(1991) embedding** which maps each MVN into an SPD matrix of dimension
(d+1)x(d+1) and measures the affine-invariant SPD distance there:

    Q(mu, Sigma) = [[ Sigma + mu mu^T, mu ],
                    [      mu^T,        1  ]]

    d_FR_CO(A, B) = d_SPD(Q_A, Q_B)
                  = || log( Q_A^{-1/2} Q_B Q_A^{-1/2} ) ||_F

This is a **lower bound** on the true Fisher-Rao distance (strict equality
holds when the two MVNs share a mean). In practice the bound is tight enough
for applications like trajectory velocity / acceleration on statistical
manifolds — what matters is that the metric respects Chentsov's axioms and
reacts monotonically to changes in the underlying parameters.

References
----------
- Calvo, M. & Oller, J.M. (1991). "A distance between multivariate normal
  distributions based in an embedding into the Siegel group". J. Multivariate
  Analysis 35(2).
- Amari, S. (1985). "Differential-Geometrical Methods in Statistics".
  Lecture Notes in Statistics 28, Springer.
- Chentsov, N.N. (1982). "Statistical Decision Rules and Optimal Inference".
  AMS Translations 53.
- Pennec, X., Fillard, P. & Ayache, N. (2006). "A Riemannian framework for
  tensor computing". Int. J. Computer Vision 66(1) — source for the
  affine-invariant SPD log-Euclidean distance used here.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import logm, sqrtm


def spd_embedding(mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Calvo-Oller embedding of N(mu, Sigma) into SPD((d+1) x (d+1)).

    Q(mu, Sigma) = [[Sigma + mu mu^T, mu],
                    [mu^T,            1 ]]

    The embedded matrix is always SPD when Sigma is SPD (verified via block
    matrix positivity of the Schur complement).
    """
    mu = np.asarray(mu, dtype=np.float64).reshape(-1)
    sigma = np.asarray(sigma, dtype=np.float64)
    d = mu.shape[0]
    if sigma.shape != (d, d):
        raise ValueError(
            f"sigma must be ({d}, {d}) to match mu of length {d}, got {sigma.shape}"
        )

    outer = np.outer(mu, mu)
    top_left = sigma + outer
    q = np.zeros((d + 1, d + 1), dtype=np.float64)
    q[:d, :d] = top_left
    q[:d, d] = mu
    q[d, :d] = mu
    q[d, d] = 1.0
    return q


def spd_affine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Affine-invariant Riemannian distance between two SPD matrices.

    d(A, B) = || log( A^{-1/2} B A^{-1/2} ) ||_F

    Equivalent (and numerically preferred via eigendecomposition when available):
    d(A, B)^2 = sum_i log^2(lambda_i)
    where lambda_i are the generalized eigenvalues of B relative to A
    (i.e. eigenvalues of A^{-1} B).

    We use the generalized eigenvalue path because it avoids matrix square
    roots and is more numerically stable on ill-conditioned inputs.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(
            f"inputs must be square matrices of equal shape, got {a.shape} and {b.shape}"
        )
    # Generalized eigenvalue problem B v = lambda A v. Equivalent to eig(A^{-1} B)
    # on real SPD inputs.
    # scipy.linalg.eigh with b=a gives the generalized eigenvalues of (b, a)
    # properly for symmetric-definite pairs.
    from scipy.linalg import eigh

    eigvals = eigh(b, a, eigvals_only=True)
    # numerical guard: SPD ⇒ all eigvals > 0; clip tiny negatives from roundoff
    eigvals = np.clip(eigvals, 1e-300, None)
    log_eigs = np.log(eigvals)
    return float(np.sqrt(np.sum(log_eigs * log_eigs)))


def spd_affine_distance_logm(a: np.ndarray, b: np.ndarray) -> float:
    """Reference implementation via matrix logarithm. Used for unit tests.

    Slower and less stable than the eigenvalue path; exposed so tests can
    verify the two paths agree.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a_inv_half = np.linalg.inv(sqrtm(a))
    m = a_inv_half @ b @ a_inv_half
    logm_m = logm(m)
    # logm can return complex for near-symmetric real matrices; imaginary part
    # is numerical noise at roundoff level when inputs are SPD.
    return float(np.linalg.norm(np.real(logm_m), ord="fro"))


def calvo_oller_distance(
    mu_a: np.ndarray,
    sigma_a: np.ndarray,
    mu_b: np.ndarray,
    sigma_b: np.ndarray,
) -> float:
    """Fisher-Rao geodesic distance between N(mu_a, Sigma_a) and N(mu_b, Sigma_b)
    via the Calvo-Oller 1991 SPD embedding.

    Returns a non-negative float. Zero iff the two MVNs are identical.
    Symmetric in A and B.
    """
    q_a = spd_embedding(mu_a, sigma_a)
    q_b = spd_embedding(mu_b, sigma_b)
    return spd_affine_distance(q_a, q_b)


def fisher_rao_1d(
    mu_a: float,
    sigma_a: float,
    mu_b: float,
    sigma_b: float,
) -> float:
    """Exact TRUE Fisher-Rao geodesic distance between two 1D Gaussians.

    Closed form (Atkinson & Mitchell 1981; see also Amari 1985 sec. 3.5):

        d_FR( (mu_A, sigma_A), (mu_B, sigma_B) )
            = sqrt(2) * acosh( 1 + ( (mu_A - mu_B)^2 + 2 (sigma_A - sigma_B)^2 )
                                 / ( 4 * sigma_A * sigma_B ) )

    **This does NOT match `calvo_oller_distance` even in 1D.** The Calvo-Oller
    SPD embedding is a *lower bound* on the true Fisher-Rao distance; the gap
    depends on the direction of displacement in (mu, sigma) space (mean-only,
    scale-only, and joint shifts each pick up different sub-unity factors).

    This function is exposed mainly for users who want the genuine 1D Fisher-Rao
    for standalone scalar problems, and as a reference/sanity utility.
    """
    if sigma_a <= 0 or sigma_b <= 0:
        raise ValueError(f"sigmas must be positive, got {sigma_a}, {sigma_b}")
    num = (mu_a - mu_b) ** 2 + 2.0 * (sigma_a - sigma_b) ** 2
    den = 4.0 * sigma_a * sigma_b
    # acosh argument is always >= 1 by non-negativity of num/den
    arg = 1.0 + num / den
    return float(np.sqrt(2.0) * np.arccosh(arg))
