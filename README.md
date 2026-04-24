# fisherrao

Fisher-Rao geodesic distance on the multivariate Gaussian manifold — pure numpy + scipy. Production-grade implementation of the **Calvo-Oller 1991 SPD embedding**, closed-form, no C++ deps, no iterative solvers, no GPU.

## What this computes

For two d-variate Gaussians N(μ_A, Σ_A) and N(μ_B, Σ_B), this library computes

  d_CO(A, B) = || log( Q_A^{-1/2} Q_B Q_A^{-1/2} ) ||_F

where

  Q(μ, Σ) = [[ Σ + μμᵀ, μ ], [ μᵀ, 1 ]]    (the (d+1)x(d+1) Calvo-Oller SPD embedding)

This is a **closed-form lower bound** on the true Fisher-Rao geodesic distance between the two MVNs. It respects Chentsov's axioms (invariant under reparametrization), is symmetric and non-negative, vanishes iff the two MVNs are identical, and composes as a proper metric. For OMEGA-style applications (detecting regime shifts via velocity/acceleration of a rolling MVN trajectory) it is the right tool — what matters is a consistent, tractable Riemannian distance, not the exact Fisher-Rao value.

The true Fisher-Rao distance between arbitrary MVN pairs has no universally-accepted closed form; most production implementations (including this one) use either the Calvo-Oller lower bound or the Strapasson 2016 iterative solver. We ship the closed-form path because it's ~100× faster and plenty accurate for geodesic-velocity work.

## Install

```bash
pip install fisherrao
```

Requires Python ≥ 3.9, numpy ≥ 1.23, scipy ≥ 1.10.

## Quickstart

```python
import numpy as np
from fisherrao import calvo_oller_distance, fit_mvn, trajectory_metrics

# two MVNs
mu_a = np.array([0.0, 0.0])
sigma_a = np.eye(2)
mu_b = np.array([1.0, 0.0])
sigma_b = np.array([[2.0, 0.3], [0.3, 1.0]])

d = calvo_oller_distance(mu_a, sigma_a, mu_b, sigma_b)   # => ~1.2

# fit an MVN from a sample matrix (shape n_samples x d) with automatic
# Ledoit-Wolf shrinkage — safe for small n
samples = np.random.default_rng(0).multivariate_normal(mu_a, sigma_a, size=40)
mu_hat, sigma_hat, shrinkage = fit_mvn(samples)

# trajectory velocity + acceleration from a sequence of MVNs
steps = trajectory_metrics(
    [(mu_a, sigma_a), (mu_b, sigma_b), (mu_b, 2 * sigma_b)],
    times=[0.0, 1.0, 2.0],
)
steps[1].velocity         # distance/time from step 0 to step 1
steps[2].acceleration     # velocity[2] - velocity[1], per unit time
```

## Regime-shift detection (the motivating use case)

The `examples/quickstart.py` script generates 60 time steps of 10-dim samples with a mean shift at step 30, fits an MVN per step, and measures Fisher-Rao velocity between consecutive MVNs. On 50 samples per step, the velocity peaks exactly at step 30 at 2.4× the pre-shift baseline — a clean regime-detection signal.

## Small-sample covariance

When the number of samples is smaller than the dimension, the empirical covariance is singular and Fisher-Rao distances blow up. `fit_mvn` defaults to **Ledoit-Wolf 2004** shrinkage, a pure-numpy implementation matching scikit-learn's behaviour without the dependency. Shrinkage is returned alongside the fit so you can audit the regularisation strength.

## API

| Function | Purpose |
|---|---|
| `calvo_oller_distance(mu_a, sigma_a, mu_b, sigma_b)` | Fisher-Rao distance (Calvo-Oller SPD embedding) between two MVNs |
| `fisher_rao_1d(mu_a, sigma_a, mu_b, sigma_b)` | Exact 1D TRUE Fisher-Rao distance (Atkinson-Mitchell 1981 closed form). Note: does NOT match `calvo_oller_distance` in 1D — the two are different metrics |
| `spd_embedding(mu, sigma)` | (d+1)×(d+1) Calvo-Oller SPD embedding of a single MVN |
| `spd_affine_distance(a, b)` | Affine-invariant Riemannian distance on raw SPD matrices (eigenvalue path, fast + stable) |
| `spd_affine_distance_logm(a, b)` | Reference path via scipy matrix logarithm (slow; for tests) |
| `fit_mvn(x, shrinkage="ledoit-wolf")` | Fit (μ, Σ) from samples with optional shrinkage |
| `ledoit_wolf_shrinkage(x)` | Standalone Ledoit-Wolf shrinkage estimator |
| `trajectory_metrics(mvns, times=None)` | Velocity + acceleration along an MVN trajectory |

## Correctness

20 unit tests cover:
- SPD embedding positivity on randomised inputs
- d(A, A) = 0 up to 1e-9 in dimensions 1, 2, 5, 10
- Symmetry d(A, B) = d(B, A)
- Agreement between the fast eigenvalue path and the reference matrix-log path (1e-7 rtol)
- Pure mean-shift monotonicity in ||Δμ||
- Pure scale-shift closed form: d_CO(N(0, I_d), N(0, c·I_d)) = √d · |log c|
- Hand-derived 1D mean-shift formula: d_CO(N(0, 1), N(Δ, 1)) = √2 · acosh(1 + Δ²/2)
- Trajectory helpers: constant velocity ⇒ ~0 acceleration, strictly-increasing-times enforcement
- Ledoit-Wolf recovers ground-truth Σ on large n (< 15% relative Frobenius error) and keeps Σ SPD on n < d+1

## Why this library exists

While building OMEGA's "Consensus Trajectory Geometry" meta-signal (Phase 19 Tier A #5 — Fisher-Rao geodesic velocity of sector-level agent-score MVNs as a pre-regime-transition indicator), we needed a small, fast, tested Fisher-Rao implementation that:

- Installs cleanly on Python 3.9–3.13 (geomstats 2.7 breaks on numpy 2.x as of 2026-04)
- Has zero C++ / GPU dependencies
- Documents the convention choice honestly (Calvo-Oller vs true Fisher-Rao is a real distinction)
- Validates against hand-derived analytical cases, not just library cross-checks

Nothing in the fisherrao API mentions OMEGA, regime shifts, or agent scores — the library is deliberately application-agnostic.

## Roadmap

**v0.2 (planned):**
- Strapasson 2016 iterative solver for the true Fisher-Rao distance (when the lower bound's gap matters)
- Pinele-Costa 2020 sharper lower/upper bounds
- Fisher-Rao for Dirichlet / Beta / Gamma families (exponential family)
- Pre-whitening helper so trajectory distances ignore chosen axes' scales

## Authors

- **Pierre Samson** ([@darw007d](https://github.com/darw007d)) — idea, use-case, design decisions
- **Claude Opus** (Anthropic) — implementation and tests

Originally motivated by the [OMEGA Swarm](https://github.com/darw007d/hedge-fund-mcp) project, Phase 19 Tier A #5 "Consensus Trajectory Geometry". Shipped as a companion to [phawkes](https://pypi.org/project/phawkes/) (same authors, same "small, tested, publishable" ethos).

## Citations

If this library contributes to a published result, please cite:

- Calvo, M. & Oller, J.M. (1991). *A distance between multivariate normal distributions based in an embedding into the Siegel group*. J. Multivariate Analysis 35(2).
- Ledoit, O. & Wolf, M. (2004). *A well-conditioned estimator for large-dimensional covariance matrices*. J. Multivariate Analysis 88(2).
- Pennec, X., Fillard, P. & Ayache, N. (2006). *A Riemannian framework for tensor computing*. Int. J. Computer Vision 66(1).
- Amari, S. (1985). *Differential-Geometrical Methods in Statistics*. Lecture Notes in Statistics 28, Springer.

## License

MIT — see [LICENSE](LICENSE).
