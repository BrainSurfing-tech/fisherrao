"""Rolling trajectory on the MVN manifold.

Given a sequence of MVN parameters (mu_t, Sigma_t) — e.g. fit from a rolling
window of multivariate samples — compute the geodesic **velocity** between
consecutive points and the **acceleration** as a second-difference of
velocities.

This is what OMEGA consumes: velocity/acceleration spikes of the per-sector
agent-score MVN trajectory become the candidate pre-regime-transition
signal. The library is application-agnostic — velocities are just distances
per unit time; the caller decides what to do with them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fisherrao.core import calvo_oller_distance


@dataclass
class TrajectoryStep:
    """One step of an MVN trajectory with derived velocity and acceleration.

    `velocity` is the Fisher-Rao geodesic distance from the previous MVN
    (None for the first step).
    `acceleration` is the difference of consecutive velocities
    (None for the first two steps).
    """

    mu: np.ndarray
    sigma: np.ndarray
    t: float
    velocity: float | None = None
    acceleration: float | None = None


def trajectory_metrics(
    mvns: list[tuple[np.ndarray, np.ndarray]],
    times: list[float] | None = None,
) -> list[TrajectoryStep]:
    """Compute per-step velocity and acceleration along an MVN trajectory.

    Parameters
    ----------
    mvns : list of (mu, Sigma) tuples
        Ordered sequence of MVN parameters.
    times : optional list of floats, same length as mvns
        Timestamps for each MVN. Used to normalise velocity (distance per unit
        time) and acceleration (velocity delta per unit time). Defaults to
        unit spacing.

    Returns
    -------
    list[TrajectoryStep], same length as mvns. First entry has velocity=None;
    first two entries have acceleration=None.
    """
    n = len(mvns)
    if n == 0:
        return []
    if times is None:
        times = list(range(n))
    if len(times) != n:
        raise ValueError(f"times length {len(times)} != mvns length {n}")

    steps: list[TrajectoryStep] = []
    velocities: list[float | None] = [None] * n

    for i, (mu, sigma) in enumerate(mvns):
        if i == 0:
            steps.append(TrajectoryStep(mu=np.asarray(mu), sigma=np.asarray(sigma), t=times[i]))
            continue
        prev_mu, prev_sigma = mvns[i - 1]
        d = calvo_oller_distance(prev_mu, prev_sigma, mu, sigma)
        dt = times[i] - times[i - 1]
        if dt <= 0:
            raise ValueError(
                f"times must be strictly increasing; step {i}: dt={dt}"
            )
        velocities[i] = d / dt
        steps.append(
            TrajectoryStep(
                mu=np.asarray(mu),
                sigma=np.asarray(sigma),
                t=times[i],
                velocity=velocities[i],
            )
        )

    # acceleration: forward difference of velocity, divided by dt between
    # step i and step i-1 (consistent with first derivative using (t_i - t_{i-1}))
    for i in range(2, n):
        v_now = velocities[i]
        v_prev = velocities[i - 1]
        if v_now is None or v_prev is None:
            continue
        dt = times[i] - times[i - 1]
        steps[i].acceleration = (v_now - v_prev) / dt

    return steps
