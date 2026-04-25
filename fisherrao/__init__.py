"""fisherrao — Fisher-Rao geometry on multiple statistical manifolds.

v0.1: Multivariate Gaussian (Calvo-Oller SPD embedding) — `calvo_oller_distance`,
trajectory metrics over MVN streams, Ledoit-Wolf shrinkage helpers.

v0.2: Beta distribution family — closed-form KL, Fisher information matrix,
geodesic distance (line-integral upper bound), Gaussian curvature via the
Brioschi formula, method-of-moments fit. The Beta module is the foundation
for OMEGA-style "information cost" of belief updates.

Public API:

    from fisherrao import (
        # MVN family (v0.1)
        calvo_oller_distance, fisher_rao_1d, fit_mvn, trajectory_metrics,
        spd_embedding, spd_affine_distance,
        # Beta family (v0.2)
        kl_beta, fisher_information_beta, geodesic_distance_beta,
        scalar_curvature_beta, fit_beta_mom,
    )
"""

from fisherrao.beta import (
    fisher_information_beta,
    fit_beta_mom,
    geodesic_distance_beta,
    kl_beta,
    scalar_curvature_beta,
)
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
    # MVN family
    "calvo_oller_distance",
    "fisher_rao_1d",
    "spd_affine_distance",
    "spd_affine_distance_logm",
    "spd_embedding",
    "fit_mvn",
    "ledoit_wolf_shrinkage",
    "TrajectoryStep",
    "trajectory_metrics",
    # Beta family
    "kl_beta",
    "fisher_information_beta",
    "geodesic_distance_beta",
    "scalar_curvature_beta",
    "fit_beta_mom",
]
__version__ = "0.2.0"
