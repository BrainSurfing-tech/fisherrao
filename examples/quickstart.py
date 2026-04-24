"""fisherrao quickstart — detect a regime shift via velocity spike.

Synthetic experiment: 60 time steps, each step aggregates 50 observations
of a 10-dim random vector. Mean is zero for steps 0-29, then shifts on
step 30+. Covariance stays fixed.

At each step we fit an MVN directly from that step's 50 samples (no rolling
window needed — 50 samples >> 10 dims). Then we compute the Fisher-Rao
geodesic velocity between consecutive MVNs and watch for a spike at t=30.
"""

import numpy as np

from fisherrao import calvo_oller_distance, fit_mvn

rng = np.random.default_rng(0)
d = 10
n_steps = 60
samples_per_step = 50

true_mu_before = np.zeros(d)
true_mu_after = np.array([0.0, 1.5, -1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
true_sigma = np.eye(d)

# one MVN per step, fit from 50 observations
mvns: list[tuple[np.ndarray, np.ndarray]] = []
for t in range(n_steps):
    mu_true = true_mu_before if t < 30 else true_mu_after
    samples = rng.multivariate_normal(mu_true, true_sigma, size=samples_per_step)
    mu_hat, sigma_hat, _ = fit_mvn(samples, shrinkage="ledoit-wolf")
    mvns.append((mu_hat, sigma_hat))

# velocities between consecutive steps
velocities: list[float] = []
for i in range(1, n_steps):
    mu_prev, s_prev = mvns[i - 1]
    mu_now, s_now = mvns[i]
    velocities.append(calvo_oller_distance(mu_prev, s_prev, mu_now, s_now))

max_idx = int(np.argmax(velocities))
t_at_max = max_idx + 1  # velocity[i] lives at step i+1

baseline = float(np.median(velocities[:28]))  # pre-shift median
peak = velocities[max_idx]
print(f"n_steps={n_steps}, samples_per_step={samples_per_step}, d={d}")
print(f"true regime shift at step 30")
print(f"peak velocity step {t_at_max}, value {peak:.4f}")
print(f"pre-shift baseline median: {baseline:.4f}  (peak is {peak/baseline:.1f}x baseline)")
print(f"lag from ground truth: {t_at_max - 30:+d} steps")
print()
print("step   velocity")
for i, v in enumerate(velocities):
    marker = "  <-- PEAK" if i == max_idx else ""
    print(f"{i+1:4d}   {v:.4f}{marker}")
