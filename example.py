#!/usr/bin/env python
"""
WHAMpy example: 3D umbrella sampling with analytical error estimation.

Generates a 3-dimensional Gaussian unbiased distribution on a 100^3 grid,
creates a chain of biased windows (harmonic potentials) exploring the grid,
solves WHAM with Anderson acceleration, estimates errors, and produces
marginal-probability plots for each dimension.
"""

import os
num_cores = 1
os.environ["OMP_NUM_THREADS"] = f"{num_cores}"
os.environ["OPENBLAS_NUM_THREADS"] = f"{num_cores}"
os.environ["MKL_NUM_THREADS"] = f"{num_cores}"
os.environ["VECLIB_MAXIMUM_THREADS"] = f"{num_cores}"
os.environ["NUMEXPR_NUM_THREADS"] = f"{num_cores}"

import numpy as np
import time

# ── User settings ──────────────────────────────────────────────────────────
ANDERSON = False          # Use Anderson/DIIS acceleration (much faster)
TOL = 1e-10               # Convergence tolerance
N_BINS = 50             # Bins per dimension  (100^3 = 1 000 000 total)
N_SAMPLES = 100_000       # Samples per window
# ───────────────────────────────────────────────────────────────────────────

from whampy import WhamSolver

# ---------------------------------------------------------------------------
# 1. Build the unbiased distribution
# ---------------------------------------------------------------------------
edges_1d = np.linspace(0.0, 10.0, N_BINS + 1)
centers_1d = (edges_1d[:-1] + edges_1d[1:]) / 2.0
bin_edges = [edges_1d.copy() for _ in range(3)]

cx, cy, cz = np.meshgrid(centers_1d, centers_1d, centers_1d, indexing="ij")

sigma_unbiased = 1.0
log_p_unbiased = -(
    (cx - 1.0) ** 2 + (cy - 1.0) ** 2 + (cz - 1.0) ** 2
) / (2 * sigma_unbiased ** 2)
p_unbiased = np.exp(log_p_unbiased)
p_unbiased /= p_unbiased.sum()


# ---------------------------------------------------------------------------
# 2. Helper functions
# ---------------------------------------------------------------------------
def make_harmonic_bias(center, kappa):
    """Return β·U on the grid for a 3D harmonic potential."""
    dx = cx - center[0]
    dy = cy - center[1]
    dz = cz - center[2]
    return 0.5 * kappa * (dx ** 2 + dy ** 2 + dz ** 2)


def sample_biased_histogram(p_unbiased, beta_bias, n_samples, rng):
    """Draw multinomial samples from the biased distribution."""
    log_p_biased = np.log(np.maximum(p_unbiased, 1e-300)) - beta_bias
    log_p_biased -= log_p_biased.max()
    p_biased = np.exp(log_p_biased)
    p_biased /= p_biased.sum()
    idx = rng.choice(p_biased.size, size=n_samples, p=p_biased.ravel())
    hist = np.bincount(idx, minlength=p_biased.size).reshape(p_biased.shape)
    return hist.astype(np.float64)


# ---------------------------------------------------------------------------
# 3. Define umbrella windows
# ---------------------------------------------------------------------------
# The first window is unbiased (kappa=0).  The remaining windows form a
# chain of harmonic restraints exploring the grid along the x-axis,
# then turning along y, then z, and a diagonal shortcut back.
window_centers = [
    (0.0, 0.0, 0.0),    # unbiased
    (1.0, 0.0, 0.0),
    (2.0, 0.0, 0.0),
    (3.0, 0.0, 0.0),
    (4.0, 0.0, 0.0),
    (5.0, 0.0, 0.0),
    (6.0, 0.0, 0.0),
    (7.0, 0.0, 0.0),
    (8.0, 0.0, 0.0),
    (9.0, 0.0, 0.0),
    (9.0, 1.0, 0.0),
    (9.0, 2.0, 0.0),
    (9.0, 3.0, 0.0),
    (9.0, 4.0, 0.0),
    (9.0, 5.0, 0.0),
    (9.0, 6.0, 0.0),
    (9.0, 7.0, 0.0),
    (9.0, 8.0, 0.0),
    (9.0, 9.0, 0.0),
    (9.0, 9.0, 1.0),
    (9.0, 9.0, 2.0),
    (9.0, 9.0, 3.0),
    (9.0, 9.0, 4.0),
    (9.0, 9.0, 5.0),
    (9.0, 9.0, 6.0),
    (9.0, 9.0, 7.0),
    (9.0, 9.0, 8.0),
    (9.0, 9.0, 9.0),
]
# window_centers = window_centers[1::2]

kappas = [np.sqrt(np.linalg.norm(center))*2 for center in window_centers]

# kappas = [0.0] + [2.0] * (len(window_centers) - 1)


print(f"Grid : {N_BINS}^3 = {N_BINS ** 3:,} bins")
print(f"Windows: {len(window_centers)}")

rng = np.random.default_rng(12345)
windows = []
for i, center in enumerate(window_centers):
    bias = make_harmonic_bias(center, kappas[i])
    hist = sample_biased_histogram(p_unbiased, bias, N_SAMPLES, rng)
    windows.append((hist, bias))
    nz = (hist > 0).sum()
    print(f"  [{i:2d}] center={center}, κ={kappas[i]:.1f}, "
          f"nonzero={nz} ({100 * nz / cx.size:.1f} %)")

# ---------------------------------------------------------------------------
# 4. Solve WHAM
# ---------------------------------------------------------------------------

print(f"\n{'='*70}")
print(f"\nSolving (anderson={ANDERSON}) ...")
print("=" * 70)
solver = WhamSolver(bin_edges=bin_edges, tol=TOL, max_iter=100_000, lazy=True)
times_orig = []
iters_orig = []
for i, (hist, bias) in enumerate(windows):
    solver.add_window(hist, bias_array=bias)
    t0 = time.perf_counter()
    r = solver.solve()
    dt = time.perf_counter() - t0
    times_orig.append(dt)
    iters_orig.append(r.n_iterations)
    print(f"  Window {i:2d}: {dt:8.3f} s  ({r.n_iterations:5d} iters, "
          f"{dt/max(r.n_iterations,1)*1000:7.2f} ms/iter)")
print(f"  TOTAL: {sum(times_orig):.3f} s")
print(f"  Converged: {r.converged}  |  "
      f"Iterations: {r.n_iterations}  |  Time: {dt:.3f} s")


# ---------------------------------------------------------------------------
# 5. Estimate errors (analytical — fast)
# ---------------------------------------------------------------------------
print("\nEstimating errors (analytical) ...")
t0 = time.perf_counter()
errors = solver.estimate_errors(method="analytical")
dt = time.perf_counter() - t0
print(f"  Done in {dt:.3f} s")
print(f"  std(f_k) = {errors.std_free_energies}")

# ---------------------------------------------------------------------------
# 6. Plot marginals for each dimension
# ---------------------------------------------------------------------------
os.makedirs("Figs", exist_ok=True)
for d in range(3):
    print(f"  Plotting dimension {d} ...")
    solver.plot_marginal(dim=d, savefn=f"Figs/example_marginal_dimension_{d}")
    import matplotlib.pyplot as plt
    plt.close("all")

print("\nDone — figures saved in Figs/")
