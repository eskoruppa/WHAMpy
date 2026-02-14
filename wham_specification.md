# WHAM Solver — Full Specification

## Purpose

Implementation of the Weighted Histogram Analysis Method (WHAM) for combining
umbrella sampling simulations of DNA cyclization. The solver handles 3 simultaneous
bias coordinates (end-to-end distance, bend angle, twist angle) but is designed to
support arbitrary dimensionality. The primary end goal is computing the cyclization
J-factor from the unbiased probability density in the closure region.

## Architecture

### Class: `WhamSolver`

Single class encapsulating all WHAM functionality: data management, solving,
diagnostics, and serialization.

### Dataclass: `WhamResult`

Lightweight, self-contained result container returned by `WhamSolver.result()`.
Independently serializable.

---

## Constructor

```python
WhamSolver(
    bin_edges: Optional[List[np.ndarray]] = None,
    tol: float = 1e-7,
    max_iter: int = 100000,
    lazy: bool = True,
    overlap_threshold: float = 0.15,
)
```

- `bin_edges`: List of 1D arrays defining bin edges for each dimension.
  If provided, bin centers and bin widths are computed automatically.
  Can also be set later via `set_bin_edges()`.
- `tol`: Convergence tolerance on max|Δf_k| between iterations (in kT units).
- `max_iter`: Maximum number of WHAM iterations.
- `lazy`: If True (default), `add_window` does not trigger solving. If False
  (eager mode), `solve()` is called automatically after each `add_window`.
- `overlap_threshold`: Default threshold for histogram overlap diagnostic (0.15 = 15%).

---

## Grid Definition Methods

### `set_bin_edges(bin_edges: List[np.ndarray]) -> None`
Sets bin edges for all dimensions. Computes and stores bin centers and bin widths.
Raises exception if windows have already been added with a different grid shape.

### `set_bin_centers(bin_centers: List[np.ndarray]) -> None`
Sets bin centers directly. Required for `add_window` with `bias_function`.
Bin widths are inferred assuming uniform spacing per dimension.

### `set_bin_volumes(volumes: np.ndarray = None, volume_function: Callable = None) -> None`
Sets bin volumes for Jacobian correction (probability → density conversion).
- `volumes`: Pre-computed array of shape grid_shape.
- `volume_function`: Callable receiving meshgrid arrays `(*coords) -> np.ndarray`.
  Evaluated immediately; only the resulting array is stored.
- If neither is provided, raises exception.
- Default (if never called): uniform volumes (product of bin widths per dimension,
  or 1.0 if only bin centers are known).

---

## Window Management Methods

### `add_window(histogram: np.ndarray, bias_array: np.ndarray = None, bias_function: Callable = None) -> int`
Adds a single umbrella sampling window.

- `histogram`: Integer array of shape grid_shape containing bin counts.
- `bias_array`: Pre-computed βU array of shape grid_shape (bias evaluated at bin centers).
- `bias_function`: Callable `(*meshgrid_coords) -> np.ndarray` returning βU on the grid.
  Requires bin_centers to be set. Evaluated immediately; result stored as array.
- If neither `bias_array` nor `bias_function` is provided: raises ValueError.
- If both are provided: `bias_array` takes precedence (with warning).
- Validates grid shape consistency with existing windows.
- Returns the index of the added window.
- If grid_shape is not yet established (first window), infers it from the histogram shape.
- If eager mode is active, calls `solve()` automatically.
- Marks internal state as dirty (not converged).

### `add_windows(windows: List[Tuple[np.ndarray, np.ndarray]]) -> List[int]`
Adds multiple windows from a list of (histogram, bias_array) tuples.
Simple loop over `add_window`. Returns list of indices.

### `remove_window(index: int) -> None`
Removes window at the given index. Reindexes remaining windows.
Invalidates converged state.

### `remove_last_window() -> None`
Convenience method: removes the most recently added window.
Useful in the progressive workflow when overlap criterion fails.

### `replace_window(index: int, histogram: np.ndarray, bias_array: np.ndarray = None, bias_function: Callable = None) -> None`
Replaces the window at the given index. Equivalent to remove + add but preserves ordering.

### Properties
- `n_windows -> int`: Number of currently stored windows.
- `grid_shape -> tuple`: Shape of the histogram grid.

---

## Solver Control Methods

### `set_eager(eager: bool = True) -> None`
Sets eager mode. `_lazy_solve = not eager`.

### `set_lazy(lazy: bool = True) -> None`
Sets lazy mode. `_lazy_solve = lazy`.

### `set_tol(tol: float) -> None`
Sets convergence tolerance.

### `set_max_iter(max_iter: int) -> None`
Sets maximum iterations.

### `set_overlap_threshold(threshold: float) -> None`
Sets the histogram overlap threshold for diagnostics.

---

## Core Solver

### `solve() -> WhamResult`
Runs the WHAM self-consistency iteration. Returns a `WhamResult`.

**Procedure:**
1. Stack all stored histograms into (K, M_full) matrix (M_full = product of grid_shape).
2. Compute combined mask: bins where sum over all windows > 0.
3. Reduce to active bins: histograms (K, M_active), biases (K, M_active).
4. Warm-start: if converged free_energies exist from a previous solve, use them
   (padding with 0.0 for any new windows).
5. Run WHAM iteration in log-space (see algorithm below).
6. Expand log_prob back to full grid shape (NaN for inactive bins).
7. Apply bin volume correction if volumes are set: log_density = log_prob - log(ΔV).
8. Compute overlap diagnostics.
9. Compute effective sample sizes.
10. Store converged state internally (free_energies, log_prob) for warm-start.
11. Return WhamResult.

**WHAM Algorithm (log-space, vectorized):**

Given:
- `log_n`: (K, M) — log of histogram counts (−∞ for zero entries)
- `beta_U`: (K, M) — bias potentials
- `log_N`: (K,) — log of total counts per window
- `f_k`: (K,) — free energies (initialized to 0 or warm-start values)

Precompute:
- `log_C`: (M,) — `logsumexp(log_n, axis=0)` — total log-counts per bin

Iterate until convergence:
```
log_denom = logsumexp(log_N[:, None] + f_k[:, None] - beta_U, axis=0)  # (M,)
log_p = log_C - log_denom                                                # (M,)
f_k_new = -logsumexp(log_p[None, :] - beta_U, axis=1)                   # (K,)
f_k_new -= f_k_new[0]                                                    # fix reference
delta = max|f_k_new - f_k|
f_k = f_k_new
```

All operations are vectorized numpy/scipy with no Python loops over bins or windows.
`scipy.special.logsumexp` is used throughout for numerical stability.

---

## Result Access

### `result() -> WhamResult`
Returns the result of the most recent solve. Raises exception if not yet solved.

### `WhamResult` dataclass fields:
```python
@dataclass
class WhamResult:
    log_prob: np.ndarray            # Log probability (density if volumes set), grid_shape.
                                    # NaN for bins with no data.
    free_energies: np.ndarray       # (K,) per-window dimensionless free energies.
    bin_edges: List[np.ndarray]     # Per-dimension bin edges (None if not set).
    bin_centers: List[np.ndarray]   # Per-dimension bin centers (None if not set).
    bin_volumes: Optional[np.ndarray]  # Bin volumes if set, else None.
    converged: bool                 # Whether iteration converged within max_iter.
    n_iterations: int               # Number of iterations performed.
    convergence_history: np.ndarray # (n_iterations,) max|Δf| per iteration.
    overlap_histogram: np.ndarray   # (K, K) pairwise histogram overlap fractions.
    overlap_matrix: np.ndarray      # (K, K) Shirts-Chodera statistical overlap matrix.
    n_eff: np.ndarray               # Effective sample size per bin, grid_shape. NaN for empty.
```

---

## Overlap Diagnostics

### Histogram Overlap (default diagnostic)
For each pair of windows (k, l), compute the overlap integral of their normalized histograms:
```
overlap(k, l) = Σ_i min(p_k(i), p_l(i))
```
where p_k(i) = n_{ki} / N_k. Stored as a (K, K) symmetric matrix.

Default threshold: 0.15 (15%). Tunable via `set_overlap_threshold()`.

### Shirts-Chodera Overlap Matrix
Computed after convergence using the WHAM weights:
```
O_{kl} = Σ_i [ N_k exp(f_k - βU_k(i)) · N_l exp(f_l - βU_l(i)) ] / [ Σ_j N_j exp(f_j - βU_j(i)) ]^2
```
The smallest nonzero eigenvalue indicates the bottleneck in information transfer.
Stored as a (K, K) matrix in the result.

### Effective Sample Size
Per-bin effective sample size:
```
N_eff(i) = (Σ_k n_{ki})^2 / Σ_k (n_{ki}^2 / N_k)
```
Useful for identifying statistically weak regions where additional windows are needed.

### `check_overlap(window_index: int = -1) -> dict`
Convenience method for the progressive workflow. Checks the overlap of a specific
window (default: last added) against all other windows. Returns a dict with:
- `sufficient`: bool — whether max overlap with any other window exceeds threshold.
- `max_overlap`: float — maximum pairwise overlap.
- `overlap_with`: list of (index, overlap_value) for all windows.

---

## Serialization

### `WhamSolver.save(path: str) -> None`
Saves the complete solver state to a single `.npz` file using `np.savez_compressed`.

Contents:
- Grid: bin_edges, bin_centers, bin_widths, grid_shape, bin_volumes
- Windows: histograms (K, M_full), beta_biases (K, M_full), n_windows
- Converged state: free_energies, log_prob, converged, n_iterations, convergence_history
- Settings: tol, max_iter, lazy_solve, overlap_threshold

### `WhamSolver.load(path: str) -> WhamSolver` (classmethod)
Reconstructs a full WhamSolver from a saved `.npz` file.

### `WhamResult.save(path: str) -> None`
Saves the result to a single `.npz` file.

### `WhamResult.load(path: str) -> WhamResult` (classmethod)
Loads a WhamResult from a saved `.npz` file.

---

## Internal Storage

All histograms and bias arrays are stored in **fully expanded form** (original grid shape).
Reduction to active (nonzero) bins happens only within `solve()` and is temporary.
After `solve()` completes, results are expanded back to the full grid shape.

On each `solve()` call:
1. A new active-bin mask is computed from the combined histogram.
2. Histograms and biases are flattened and masked to active bins only.
3. WHAM iteration runs on the reduced arrays.
4. Results are expanded back to the full grid.

This approach is simple, avoids complex sparse-array bookkeeping, and the memory cost
is acceptable for typical 3D grids with ~50 windows.

---

## Progressive Workflow Support

The solver is designed for iterative window addition:
1. Start with the unbiased ensemble as the first window (bias_array = zeros).
2. Analyze the resulting distribution.
3. Choose bias parameters for the next window to achieve ~15% overlap.
4. Run the new simulation, add the window.
5. Solve (warm-started from previous free energies).
6. Check overlap with `check_overlap()`.
7. If insufficient overlap: `remove_last_window()`, adjust parameters, re-run simulation.
8. If sufficient: proceed to next window.
9. Repeat until the closure region is adequately sampled.

Warm-starting: on each `solve()`, if free_energies from a previous solve exist,
they are used as initial values (padded with 0.0 for new windows). This dramatically
reduces iteration count for incremental solves.

---

## Dependencies

- numpy
- scipy (scipy.special.logsumexp)
- Standard library: dataclasses, typing, warnings, pathlib

---

## Deferred Features (not in v1)

- Error estimation (block bootstrap, Ferrenberg-Swendsen error propagation)
- Automated window placement suggestions
- Visualization utilities
