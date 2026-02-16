# WHAMpy

A general-purpose Python implementation of the **Weighted Histogram Analysis Method (WHAM)** for combining umbrella sampling simulations. Supports arbitrary dimensionality, progressive window addition with warm-start convergence, Anderson/DIIS acceleration, analytical and bootstrap error estimation, overlap diagnostics, and full state serialization.

## Installation

### From PyPI (when published)

```bash
pip install whampy
```

### From GitHub

```bash
pip install git+https://github.com/eskoruppa/WHAMpy.git
```

### Local (editable) install

```bash
git clone https://github.com/eskoruppa/WHAMpy.git
cd WHAMpy
pip install -e .
```

## Overview

WHAMpy provides three main objects:

- **`WhamSolver`** — manages simulation windows, runs the WHAM self-consistency iteration, estimates errors, and computes diagnostics.
- **`WhamResult`** — a lightweight, self-contained dataclass holding all outputs of a WHAM computation, including optional error estimates.
- **`WhamErrors`** — a dataclass holding per-bin and per-window uncertainty estimates (variances, standard deviations, covariances, confidence intervals).

All bias potentials are expected in **dimensionless units** (βU, i.e., units of kT). The iteration kernel uses BLAS-accelerated matrix-vector products for high performance.

## Quick-Start Example

The following script creates a 3D umbrella sampling problem on a 100³ grid, solves it with Anderson acceleration, estimates errors, and produces diagnostic plots for each dimension:

```python
import numpy as np
from whampy import WhamSolver

# 1. Grid
N = 100
edges = np.linspace(0.0, 10.0, N + 1)
bin_edges = [edges.copy() for _ in range(3)]
centers = (edges[:-1] + edges[1:]) / 2.0
cx, cy, cz = np.meshgrid(centers, centers, centers, indexing="ij")

# 2. Unbiased distribution (3D Gaussian centred at (1,1,1), σ=1)
log_p = -((cx - 1)**2 + (cy - 1)**2 + (cz - 1)**2) / 2.0
p = np.exp(log_p); p /= p.sum()

# 3. Build biased histograms
def bias(c, kappa):
    return 0.5 * kappa * ((cx-c[0])**2 + (cy-c[1])**2 + (cz-c[2])**2)

rng = np.random.default_rng(42)
window_centers = [(0,0,0), (1,0,0), (2,0,0), (3,0,0),
                  (4,0,0), (5,0,0), (6,0,0), (7,0,0),
                  (8,0,0), (9,0,0)]
kappas = [0.0] + [2.0] * 9   # first window is unbiased

windows = []
for c, k in zip(window_centers, kappas):
    b = bias(c, k)
    lp = np.log(np.maximum(p, 1e-300)) - b
    lp -= lp.max()
    pb = np.exp(lp); pb /= pb.sum()
    idx = rng.choice(pb.size, size=50_000, p=pb.ravel())
    h = np.bincount(idx, minlength=pb.size).reshape(pb.shape)
    windows.append((h.astype(float), b))

# 4. Solve
solver = WhamSolver(bin_edges=bin_edges, tol=1e-7)
for h, b in windows:
    solver.add_window(h, bias_array=b)

result = solver.solve(anderson=True, anderson_depth=8)
print(f"Converged: {result.converged}, iterations: {result.n_iterations}")

# 5. Error estimation (analytical — fast)
errors = solver.estimate_errors(method="analytical")
print(f"std(f_k) = {errors.std_free_energies}")

# 6. Plot marginal probability along each dimension
for d in range(3):
    solver.plot_marginal(dim=d, savefn=f"marginal_dim_{d}")
```

See [`example.py`](example.py) for a complete, runnable version.

## Grid Setup

Before adding windows, define the histogram grid in one of two ways:

- **`set_bin_edges(bin_edges)`** — provide a list of 1-D arrays of bin edges (one per dimension). Bin centers, widths, and default volumes are computed automatically.
- **`set_bin_centers(bin_centers)`** — provide bin centers directly. Widths are inferred assuming uniform spacing.

Optionally, apply a Jacobian correction to convert probability to density:

- **`set_bin_volumes(volumes=..., volume_function=...)`** — supply a pre-computed volume array or a callable that receives meshgrid coordinate arrays and returns volumes.

## Adding Windows

Each umbrella sampling window consists of a histogram of bin counts and a bias potential evaluated on the grid:

- **`add_window(histogram, bias_array=None, bias_function=None)`** — add a single window. The bias can be provided as a pre-computed array or as a callable over meshgrid coordinates. Returns the window index.
- **`add_windows(windows)`** — add multiple windows from a list of `(histogram, bias_array)` tuples.
- **`remove_window(index)`** / **`remove_last_window()`** — remove a window by index or remove the most recently added window.
- **`replace_window(index, histogram, ...)`** — replace a window in-place, preserving ordering.

Properties **`n_windows`** and **`grid_shape`** report the current state.

## Solving

- **`solve()`** — run the WHAM self-consistency iteration and return a `WhamResult`. Automatically warm-starts from previous free energies when windows are added incrementally, to reduce iteration count.

### Anderson mixing acceleration

Pass `anderson=True` to `solve()` to enable Anderson/DIIS mixing, which dramatically reduces the number of iterations needed for convergence:

```python
result = solver.solve(anderson=True, anderson_depth=8)
```

Anderson mixing (also known as Anderson acceleration or DIIS — Direct Inversion in the Iterative Subspace) fits a local linear model from the last *depth* iterates and extrapolates toward the fixed point. This can convert the linear convergence of plain WHAM iteration at spectral radius ρ into approximately ρ^depth convergence, yielding 5–30× overall speedup on typical problems.

| Parameter | Default | Description |
|---|---|---|
| `anderson` | `False` | Enable Anderson/DIIS mixing. |
| `anderson_depth` | `8` | Number of historical iterates used for extrapolation. Larger values can accelerate convergence but increase per-iteration overhead marginally. |

**Automatic fallback.** If Anderson mixing stalls (repeated extrapolation rejections or failure to converge), the solver automatically falls back to plain fixed-point iteration for the remaining budget of iterations, ensuring robust convergence.

The solver operates in either **lazy** (default) or **eager** mode:

- **`set_lazy()` / `set_eager()`** — in lazy mode, `solve()` must be called explicitly. In eager mode, `solve()` runs automatically after each `add_window` call.

Convergence parameters can be adjusted at any time:

- **`set_tol(tol)`** — convergence tolerance on max |Δf_k| (default: 1e-7).
- **`set_max_iter(max_iter)`** — maximum iterations (default: 100,000).

## Results

**`solve()`** returns a `WhamResult` (also retrievable later via **`result()`**) containing:

| Field | Description |
|---|---|
| `log_prob` | Log unbiased probability (or density if volumes are set). Shape = grid_shape. `NaN` for empty bins. |
| `free_energies` | Per-window dimensionless free energies, shape (K,). |
| `bin_edges` | Per-dimension bin edges (or `None`). |
| `bin_centers` | Per-dimension bin centers (or `None`). |
| `bin_volumes` | Bin volumes if set, else `None`. |
| `converged` | Whether the iteration converged within `max_iter`. |
| `n_iterations` | Number of iterations performed. |
| `convergence_history` | Max $|\Delta f_k|$ per iteration. |
| `overlap_histogram` | (K, K) pairwise histogram overlap fractions. |
| `overlap_matrix` | (K, K) Shirts–Chodera statistical overlap matrix. |
| `n_eff` | Effective sample size per bin. `NaN` for empty bins. |
| `errors` | `WhamErrors` instance (or `None` if `estimate_errors()` has not been called). |

## Error Estimation

After solving, call **`estimate_errors()`** to compute uncertainties on the free energies and per-bin log-probabilities:

```python
# Fast analytical estimate (default)
errors = solver.estimate_errors(method="analytical")

# Robust bootstrap estimate (slower)
errors = solver.estimate_errors(method="bootstrap", n_bootstrap=200, seed=42)
```

The result is cached internally: subsequent calls to `result()` or `plot_marginal()` will automatically include the error estimates. The cache is invalidated when `solve()` is called again or windows are modified.

### Analytical method (Fisher information)

The analytical method computes the **Hessian** of the WHAM negative log-likelihood at the converged solution and inverts it to obtain the covariance matrix of the free-energy parameters $\{g_k\}$. The Hessian is:

$$H_{ij} = \delta_{ij} \sum_l M_l\, W_{il} \;-\; \sum_l M_l\, W_{il}\, W_{jl}$$

where $W_{il} = N_i \exp(g_i - \beta U_{il}) / D_l$ are the WHAM weights and $M_l = \sum_k n_{kl}$ is the total count in bin $l$. Since $g_1 = 0$ is fixed (gauge condition), the inversion operates on the reduced $(K{-}1) \times (K{-}1)$ sub-matrix.

Per-bin variance of $\ln p_l$ has two contributions:

$$\mathrm{Var}(\ln p_l) = \frac{1}{M_l} + \sum_{ij} W_{il}\, W_{jl}\, \mathrm{Cov}(g_i, g_j)$$

The first term is pure counting noise; the second propagates free-energy uncertainty. This approach is exact in the asymptotic (large-sample) regime, where the MLE is approximately Gaussian (Refs. 6, 7).

**Cost:** A single $K \times K$ matrix inversion plus $K \times M$ matrix operations—negligible compared to `solve()`.

### Bootstrap method

The bootstrap method repeatedly resamples each window's histogram via **multinomial resampling** (drawing $N_k$ counts with replacement from the observed bin probabilities), then re-solves WHAM for each replicate (warm-started from the original free energies). Statistics (variance, confidence intervals) are collected over all replicates (Refs. 6, 8).

**Cost:** Approximately `n_bootstrap` × cost of `solve()`. For large grids, this can be significant; 200 replicates is typically sufficient for stable variance estimates.

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `method` | `"analytical"` | `"analytical"` (fast, asymptotic) or `"bootstrap"` (robust, expensive). |
| `n_bootstrap` | `200` | Number of bootstrap replicates (bootstrap only). |
| `ci_level` | `0.95` | Confidence level for bootstrap CI (bootstrap only). |
| `seed` | `None` | Random seed for bootstrap reproducibility. |
| `store_replicates` | `False` | If `True`, store all bootstrap replicate arrays (can be large). |
| `verbose` | `False` | Print progress during bootstrap. |

### WhamErrors fields

| Field | Description |
|---|---|
| `method` | `"analytical"` or `"bootstrap"`. |
| `var_log_prob` | Per-bin variance of $\ln p$, shape = grid_shape. |
| `std_log_prob` | Per-bin standard deviation of $\ln p$, shape = grid_shape. |
| `var_free_energies` | Per-window variance of $g_k$, shape (K,). |
| `std_free_energies` | Per-window std of $g_k$, shape (K,). |
| `cov_free_energies` | Full covariance matrix, shape (K, K). |
| `log_prob_ci_lo/hi` | Bootstrap confidence interval bounds (bootstrap only). |

## Overlap Diagnostics

WHAMpy provides three overlap and quality diagnostics computed automatically:

- **Histogram overlap** — for each pair of windows, the overlap integral of their normalized histograms. Stored as a symmetric (K, K) matrix.
- **Shirts–Chodera statistical overlap matrix** — a (K, K) matrix whose smallest nonzero eigenvalue indicates the bottleneck in information transfer between windows.
- **Effective sample size** — per-bin $N_\text{eff}$ identifying statistically weak regions.

The convenience method **`check_overlap(window_index=-1)`** checks a single window's overlap against all others, returning whether the maximum overlap exceeds the configurable threshold (default: 15%) along with per-pair overlap values. The threshold is adjustable via **`set_overlap_threshold(threshold)`**.

## Serialization

Both `WhamSolver` and `WhamResult` support full save/load round-trips via compressed `.npz` files:

- **`WhamSolver.save(path)` / `WhamSolver.load(path)`** — persist and restore the complete solver state (grid, windows, settings, converged state).
- **`WhamResult.save(path)` / `WhamResult.load(path)`** — persist and restore a result independently of the solver.

## Plotting

The solver provides a built-in visualization method for inspecting marginalized results:

```python
fig, (ax_prob, ax_fe, ax_hist) = solver.plot_marginal(
    dim=0,
    savefn="my_figure",   # saves .pdf, .png, .svg
)
```

**`plot_marginal(dim, ...)`** creates a three-panel figure:

- **Row 1** — marginalised unbiased probability $p(x_d)$ along dimension *dim*.
- **Row 2** — marginalised free energy $-\ln p(x_d)$ in units of kT, with per-window curves aligned to the WHAM combined estimate.
- **Row 3** — raw biased (observed) histogram marginals for each window.

All panels show per-window curves in distinct colours and the combined WHAM estimate as a solid black line. If any window has a constant (or zero) bias, it is automatically identified as the unbiased reference and plotted as a dashed black line. An explicit `unbiased_histogram` argument overrides auto-detection.

If `estimate_errors()` has been called, the probability and free-energy panels include a **95 % confidence interval** (shaded grey region) around the WHAM combined curve, propagated from the cached error estimates.

| Parameter | Default | Description |
|---|---|---|
| `dim` | `0` | Dimension index to marginalize onto. |
| `unbiased_histogram` | `None` | Known unbiased histogram for reference overlay. Auto-detected from windows if not provided. |
| `savefn` | `None` | Base filename; saves `<savefn>.pdf`, `.png`, `.svg`. |
| `figsize` | `(10, 12)` | Figure size in inches. |
| `cmap` | `"tab10"` | Matplotlib colormap for window colours. |
| `count_threshold` | `0.01` | Fraction of per-window max below which bins are masked to suppress noise. |

## Dependencies

- Python ≥ 3.8
- NumPy ≥ 1.20
- SciPy ≥ 1.6
- Matplotlib (optional, for `plot_marginal`)

## References

1. S. Kumar, J. M. Rosenberg, D. Bouzida, R. H. Swendsen, and P. A. Kollman, "The Weighted Histogram Analysis Method for Free-Energy Calculations on Biomolecules. I. The Method," *J. Comput. Chem.*, **13**, 1011–1021 (1992). — Original WHAM formulation.

2. M. R. Shirts and J. D. Chodera, "Statistically optimal analysis of samples from multiple equilibrium states," *J. Chem. Phys.*, **129**, 124105 (2008). — MBAR / statistical overlap matrix.

3. D. G. Anderson, "Iterative Procedures for Nonlinear Integral Equations," *J. ACM*, **12**, 547–560 (1965). — Anderson acceleration (Anderson mixing).

4. P. Pulay, "Convergence Acceleration of Iterative Sequences. The Case of SCF Iteration," *Chem. Phys. Lett.*, **73**, 393–398 (1980). — DIIS (Direct Inversion in the Iterative Subspace), the SCF variant of Anderson mixing.

5. H. F. Walker and P. Ni, "Anderson Acceleration for Fixed-Point Iterations," *SIAM J. Numer. Anal.*, **49**, 1715–1735 (2011). — Convergence analysis of Anderson acceleration.

6. F. Zhu and G. Hummer, "Convergence and error estimation in free energy calculations using the weighted histogram analysis method," *J. Comput. Chem.*, **33**, 453–465 (2012). — WHAM convergence properties and the Fisher-information / Hessian-based analytical error estimate used in `estimate_errors(method="analytical")`.

7. B. Roux, "The calculation of the potential of mean force using computer simulations," *Comput. Phys. Commun.*, **91**, 275–282 (1995). — Free-energy error propagation through WHAM weights; Poisson counting noise + free-energy covariance decomposition.

8. B. Efron and R. J. Tibshirani, *An Introduction to the Bootstrap*, Chapman & Hall/CRC, 1993. — Bootstrap resampling theory underlying `estimate_errors(method="bootstrap")`.

## License

GNU General Public License v2 (GPLv2). See [LICENSE](LICENSE) for details.