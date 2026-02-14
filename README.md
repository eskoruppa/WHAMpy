# WHAMpy

A general-purpose Python implementation of the **Weighted Histogram Analysis Method (WHAM)** for combining umbrella sampling simulations. Supports arbitrary dimensionality, progressive window addition with warm-start convergence, overlap diagnostics, and full state serialization.

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

WHAMpy provides two main objects:

- **`WhamSolver`** — manages simulation windows, runs the WHAM self-consistency iteration, and computes diagnostics.
- **`WhamResult`** — a lightweight, self-contained dataclass holding all outputs of a WHAM computation.

All bias potentials are expected in **dimensionless units** (βU, i.e., units of kT). All internal arithmetic is performed in log-space using `scipy.special.logsumexp` for numerical stability.

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

- **`solve()`** — run the WHAM self-consistency iteration and return a `WhamResult`. Automatically warm-starts from previous free energies when windows are added incrementally, dramatically reducing iteration count.

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
| `convergence_history` | Max |Δf_k| per iteration. |
| `overlap_histogram` | (K, K) pairwise histogram overlap fractions. |
| `overlap_matrix` | (K, K) Shirts–Chodera statistical overlap matrix. |
| `n_eff` | Effective sample size per bin. `NaN` for empty bins. |

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

## Dependencies

- Python ≥ 3.8
- NumPy ≥ 1.20
- SciPy ≥ 1.6

## License

GNU General Public License v2 (GPLv2). See [LICENSE](LICENSE) for details.