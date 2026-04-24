"""fisherrao — Fisher-Rao geodesic distance on the multivariate Gaussian manifold.

Public API:

    from fisherrao import (
        calvo_oller_distance,  # Fisher-Rao distance between two MVNs
        fisher_rao_1d,          # exact 1D Gaussian distance (for tests)
        fit_mvn,                # fit MVN from sample with Ledoit-Wolf shrinkage
        trajectory_metrics,     # velocity + acceleration along MVN trajectory
    )

    d = calvo_oller_distance(mu_a, sigma_a, mu_b, sigma_b)

    steps = trajectory_metrics([(mu1, S1), (mu2, S2), (mu3, S3)], times=[0, 1, 2])
    steps[-1].velocity       # Fisher-Rao distance per unit time
    steps[-1].acceleration   # second derivative
"""

from fisherrao.core import (
    calvo_oller_distance,
    fisher_rao_1d,
    spd_affine_distance,
    spd_affine_distance_logm,
    spd_embedding,
)
from fisherrao.shrinkage import fit_mvn, ledoit_wolf_shrinkage
from fisherrao.trajectory import TrajectoryStep, trajectory_metrics

__all__ = [
    "calvo_oller_distance",
    "fisher_rao_1d",
    "spd_affine_distance",
    "spd_affine_distance_logm",
    "spd_embedding",
    "fit_mvn",
    "ledoit_wolf_shrinkage",
    "TrajectoryStep",
    "trajectory_metrics",
]
__version__ = "0.1.0"
