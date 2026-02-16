from __future__ import annotations

import warnings
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable, List, Tuple, Union, TYPE_CHECKING
from pathlib import Path

from .wham_errors import WhamErrors
from .wham_errors import analytical_errors, bootstrap_errors

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class WhamResult:
    """Self-contained result of a WHAM computation.

    All array fields with spatial extent have shape equal to the original
    histogram grid shape.  Bins that had no data across any window are
    filled with ``np.nan`` in *log_prob* and *n_eff*.

    Attributes
    ----------
    log_prob : np.ndarray
        Log of the unbiased probability (density if bin volumes were set).
        Shape = grid_shape.  NaN for empty bins.
    free_energies : np.ndarray
        Dimensionless per-window free energies, shape (K,).
    bin_edges : list of np.ndarray or None
        Per-dimension bin edges.
    bin_centers : list of np.ndarray or None
        Per-dimension bin centers.
    bin_volumes : np.ndarray or None
        Bin volumes (Jacobian correction), shape = grid_shape.
    converged : bool
        Whether the iteration converged within *max_iter*.
    n_iterations : int
        Number of iterations performed.
    convergence_history : np.ndarray
        Max |Δf_k| per iteration, shape (n_iterations,).
    overlap_histogram : np.ndarray
        Pairwise histogram overlap fractions, shape (K, K).
    overlap_matrix : np.ndarray
        Shirts–Chodera statistical overlap matrix, shape (K, K).
    n_eff : np.ndarray
        Effective sample size per bin, shape = grid_shape.  NaN for empty.
    """

    log_prob: np.ndarray
    free_energies: np.ndarray
    bin_edges: Optional[List[np.ndarray]]
    bin_centers: Optional[List[np.ndarray]]
    bin_volumes: Optional[np.ndarray]
    converged: bool
    n_iterations: int
    convergence_history: np.ndarray
    overlap_histogram: np.ndarray
    overlap_matrix: np.ndarray
    n_eff: np.ndarray
    errors: Optional["WhamErrors"] = None

    # -- serialization -----------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        """Save the result to a single ``.npz`` file."""
        path = Path(path)
        data = {}
        data["log_prob"] = self.log_prob
        data["free_energies"] = self.free_energies
        data["converged"] = np.array(self.converged)
        data["n_iterations"] = np.array(self.n_iterations)
        data["convergence_history"] = self.convergence_history
        data["overlap_histogram"] = self.overlap_histogram
        data["overlap_matrix"] = self.overlap_matrix
        data["n_eff"] = self.n_eff

        if self.bin_volumes is not None:
            data["bin_volumes"] = self.bin_volumes

        if self.bin_edges is not None:
            data["n_dims_edges"] = np.array(len(self.bin_edges))
            for i, e in enumerate(self.bin_edges):
                data[f"bin_edges_{i}"] = e

        if self.bin_centers is not None:
            data["n_dims_centers"] = np.array(len(self.bin_centers))
            for i, c in enumerate(self.bin_centers):
                data[f"bin_centers_{i}"] = c

        np.savez_compressed(str(path), **data)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "WhamResult":
        """Load a ``WhamResult`` from a ``.npz`` file."""
        path = Path(path)
        if not path.suffix:
            path = path.with_suffix(".npz")
        with np.load(str(path), allow_pickle=False) as f:
            bin_edges = None
            if "n_dims_edges" in f:
                n = int(f["n_dims_edges"])
                bin_edges = [f[f"bin_edges_{i}"] for i in range(n)]
            bin_centers = None
            if "n_dims_centers" in f:
                n = int(f["n_dims_centers"])
                bin_centers = [f[f"bin_centers_{i}"] for i in range(n)]
            bin_volumes = f["bin_volumes"] if "bin_volumes" in f else None
            return cls(
                log_prob=f["log_prob"],
                free_energies=f["free_energies"],
                bin_edges=bin_edges,
                bin_centers=bin_centers,
                bin_volumes=bin_volumes,
                converged=bool(f["converged"]),
                n_iterations=int(f["n_iterations"]),
                convergence_history=f["convergence_history"],
                overlap_histogram=f["overlap_histogram"],
                overlap_matrix=f["overlap_matrix"],
                n_eff=f["n_eff"],
            )


# ---------------------------------------------------------------------------
# Anderson mixing helper (module-level)
# ---------------------------------------------------------------------------

def _anderson_extrapolate(
    F_hist: np.ndarray,
    R_hist: np.ndarray,
    n_stored: int,
    m: int,
    r_current: np.ndarray,
) -> Optional[np.ndarray]:
    """Compute Anderson-accelerated iterate from history ring buffer.

    Uses the last min(n_stored, m) stored (iterate, residual) pairs to
    solve a small least-squares problem that extrapolates toward the
    fixed point.

    Parameters
    ----------
    F_hist : (m, K) ring buffer of iterates
    R_hist : (m, K) ring buffer of residuals
    n_stored : total number of pairs stored so far
    m : ring buffer capacity (Anderson depth)
    r_current : (K,) current residual

    Returns
    -------
    f_extrapolated : (K,) or None if the least-squares solve fails
    """
    n_use = min(n_stored, m)
    if n_use < 2:
        return None

    # Gather history in chronological order (most recent first)
    indices = [(n_stored - 1 - j) % m for j in range(n_use)]
    R_mat = R_hist[indices]  # (n_use, K), most recent first
    F_mat = F_hist[indices]  # (n_use, K)

    # Difference matrices relative to most recent entry
    dR = R_mat[0] - R_mat[1:]  # (n_use-1, K)
    dF = F_mat[0] - F_mat[1:]  # (n_use-1, K)

    # Solve: theta* = argmin ||r_current - dR^T theta||^2
    # Gram matrix is (n_use-1) x (n_use-1) — tiny
    gram = dR @ dR.T
    rhs = dR @ r_current

    try:
        # Tikhonov regularization for stability
        reg = 1e-10 * np.trace(gram) / max(len(gram), 1)
        gram += reg * np.eye(len(gram))
        theta = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        return None

    # Extrapolated iterate:
    # f_new = (f_current + r_current) - (dF + dR)^T @ theta
    f_new = (F_mat[0] + R_mat[0]) - (dF + dR).T @ theta
    f_new -= f_new[0]  # re-anchor

    return f_new


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class WhamSolver:
    """Weighted Histogram Analysis Method solver.

    Parameters
    ----------
    bin_edges : list of np.ndarray, optional
        Bin edges for each dimension.  If provided, bin centers and widths
        are computed automatically.
    tol : float
        Convergence tolerance (max |Δf_k| in kT).
    max_iter : int
        Maximum self-consistency iterations.
    lazy : bool
        If True (default), ``add_window`` does not trigger solving.
    overlap_threshold : float
        Default histogram-overlap threshold for diagnostics (0–1).
    """

    def __init__(
        self,
        bin_edges: Optional[List[np.ndarray]] = None,
        tol: float = 1e-7,
        max_iter: int = 100_000,
        lazy: bool = True,
        overlap_threshold: float = 0.15,
    ):
        # grid
        self._bin_edges: Optional[List[np.ndarray]] = None
        self._bin_centers: Optional[List[np.ndarray]] = None
        self._bin_widths: Optional[List[np.ndarray]] = None
        self._grid_shape: Optional[tuple] = None
        self._bin_volumes: Optional[np.ndarray] = None

        # per-window data (fully expanded)
        self._histograms: List[np.ndarray] = []
        self._beta_biases: List[np.ndarray] = []

        # converged state
        self._free_energies: Optional[np.ndarray] = None
        self._log_prob: Optional[np.ndarray] = None
        self._converged: bool = False
        self._n_iterations: int = 0
        self._convergence_history: Optional[np.ndarray] = None
        self._overlap_histogram: Optional[np.ndarray] = None
        self._overlap_matrix: Optional[np.ndarray] = None
        self._n_eff: Optional[np.ndarray] = None
        self._errors: Optional["WhamErrors"] = None
        self._dirty: bool = True

        # settings
        self._tol: float = tol
        self._max_iter: int = max_iter
        self._lazy_solve: bool = lazy
        self._overlap_threshold: float = overlap_threshold

        if bin_edges is not None:
            self.set_bin_edges(bin_edges)

    # ---- grid definition -------------------------------------------------

    def set_bin_edges(self, bin_edges: List[np.ndarray]) -> None:
        """Set bin edges for all dimensions."""
        bin_edges = [np.asarray(e, dtype=np.float64) for e in bin_edges]
        centers = [(e[:-1] + e[1:]) / 2.0 for e in bin_edges]
        widths = [np.diff(e) for e in bin_edges]
        shape = tuple(len(c) for c in centers)
        if self._grid_shape is not None and shape != self._grid_shape:
            raise ValueError(
                f"Grid shape {shape} from bin_edges is inconsistent with "
                f"existing grid shape {self._grid_shape}."
            )
        self._bin_edges = bin_edges
        self._bin_centers = centers
        self._bin_widths = widths
        self._grid_shape = shape
        meshwidths = np.meshgrid(*widths, indexing="ij")
        self._bin_volumes = np.ones(shape, dtype=np.float64)
        for mw in meshwidths:
            self._bin_volumes *= mw

    def set_bin_centers(self, bin_centers: List[np.ndarray]) -> None:
        """Set bin centers directly (uniform spacing inferred)."""
        bin_centers = [np.asarray(c, dtype=np.float64) for c in bin_centers]
        shape = tuple(len(c) for c in bin_centers)
        if self._grid_shape is not None and shape != self._grid_shape:
            raise ValueError(
                f"Grid shape {shape} from bin_centers is inconsistent with "
                f"existing grid shape {self._grid_shape}."
            )
        self._bin_centers = bin_centers
        self._grid_shape = shape
        widths = []
        for c in bin_centers:
            if len(c) > 1:
                w = np.full_like(c, c[1] - c[0])
            else:
                w = np.ones_like(c)
            widths.append(w)
        self._bin_widths = widths

    def set_bin_volumes(
        self,
        volumes: Optional[np.ndarray] = None,
        volume_function: Optional[Callable] = None,
    ) -> None:
        """Set bin volumes for Jacobian correction."""
        if volumes is not None:
            volumes = np.asarray(volumes, dtype=np.float64)
            if self._grid_shape is not None and volumes.shape != self._grid_shape:
                raise ValueError(
                    f"Volume array shape {volumes.shape} != grid shape {self._grid_shape}."
                )
            self._bin_volumes = volumes
        elif volume_function is not None:
            if self._bin_centers is None:
                raise RuntimeError(
                    "bin_centers must be set before calling set_bin_volumes "
                    "with a volume_function."
                )
            coords = np.meshgrid(*self._bin_centers, indexing="ij")
            self._bin_volumes = np.asarray(
                volume_function(*coords), dtype=np.float64
            )
        else:
            raise ValueError("Provide either 'volumes' or 'volume_function'.")
        self._dirty = True

    # ---- window management -----------------------------------------------

    @property
    def n_windows(self) -> int:
        return len(self._histograms)

    @property
    def grid_shape(self) -> Optional[tuple]:
        return self._grid_shape

    def add_window(
        self,
        histogram: np.ndarray,
        bias_array: Optional[np.ndarray] = None,
        bias_function: Optional[Callable] = None,
    ) -> int:
        """Add a single umbrella-sampling window."""
        histogram = np.asarray(histogram, dtype=np.float64)
        if self._grid_shape is None:
            self._grid_shape = histogram.shape
        elif histogram.shape != self._grid_shape:
            raise ValueError(
                f"Histogram shape {histogram.shape} != grid shape {self._grid_shape}."
            )

        if bias_array is not None:
            bias = np.asarray(bias_array, dtype=np.float64)
            if bias_function is not None:
                warnings.warn(
                    "Both bias_array and bias_function provided; using bias_array.",
                    stacklevel=2,
                )
        elif bias_function is not None:
            if self._bin_centers is None:
                raise RuntimeError("bin_centers must be set before using bias_function.")
            coords = np.meshgrid(*self._bin_centers, indexing="ij")
            bias = np.asarray(bias_function(*coords), dtype=np.float64)
        else:
            raise ValueError("Provide either 'bias_array' or 'bias_function'.")

        if bias.shape != self._grid_shape:
            raise ValueError(f"Bias shape {bias.shape} != grid shape {self._grid_shape}.")

        self._histograms.append(histogram)
        self._beta_biases.append(bias)
        self._dirty = True
        idx = len(self._histograms) - 1
        if not self._lazy_solve and self.n_windows >= 1:
            self.solve()
        return idx

    def add_windows(self, windows: List[Tuple[np.ndarray, np.ndarray]]) -> List[int]:
        """Add multiple windows from (histogram, bias_array) pairs."""
        was_lazy = self._lazy_solve
        self._lazy_solve = True
        indices = []
        for hist, bias in windows:
            indices.append(self.add_window(hist, bias_array=bias))
        self._lazy_solve = was_lazy
        if not self._lazy_solve and self.n_windows >= 1:
            self.solve()
        return indices

    def remove_window(self, index: int) -> None:
        """Remove the window at *index*."""
        if index < 0 or index >= self.n_windows:
            raise IndexError(f"Window index {index} out of range [0, {self.n_windows}).")
        self._histograms.pop(index)
        self._beta_biases.pop(index)
        if self._free_energies is not None and len(self._free_energies) > 0:
            fe = list(self._free_energies)
            if index < len(fe):
                fe.pop(index)
            self._free_energies = np.array(fe) if fe else None
        self._dirty = True

    def remove_last_window(self) -> None:
        if self.n_windows == 0:
            raise RuntimeError("No windows to remove.")
        self.remove_window(self.n_windows - 1)

    def replace_window(
        self, index: int, histogram: np.ndarray,
        bias_array: Optional[np.ndarray] = None,
        bias_function: Optional[Callable] = None,
    ) -> None:
        """Replace the window at *index* (preserves ordering)."""
        if index < 0 or index >= self.n_windows:
            raise IndexError(f"Window index {index} out of range [0, {self.n_windows}).")
        histogram = np.asarray(histogram, dtype=np.float64)
        if histogram.shape != self._grid_shape:
            raise ValueError(f"Histogram shape {histogram.shape} != grid shape {self._grid_shape}.")
        if bias_array is not None:
            bias = np.asarray(bias_array, dtype=np.float64)
        elif bias_function is not None:
            if self._bin_centers is None:
                raise RuntimeError("bin_centers must be set before using bias_function.")
            coords = np.meshgrid(*self._bin_centers, indexing="ij")
            bias = np.asarray(bias_function(*coords), dtype=np.float64)
        else:
            raise ValueError("Provide either 'bias_array' or 'bias_function'.")
        if bias.shape != self._grid_shape:
            raise ValueError(f"Bias shape {bias.shape} != grid shape {self._grid_shape}.")
        self._histograms[index] = histogram
        self._beta_biases[index] = bias
        self._dirty = True

    # ---- solver control --------------------------------------------------

    def set_eager(self, eager: bool = True) -> None:
        self._lazy_solve = not eager

    def set_lazy(self, lazy: bool = True) -> None:
        self._lazy_solve = lazy

    def set_tol(self, tol: float) -> None:
        self._tol = tol

    def set_max_iter(self, max_iter: int) -> None:
        self._max_iter = max_iter

    def set_overlap_threshold(self, threshold: float) -> None:
        self._overlap_threshold = threshold

    # ---- core WHAM -------------------------------------------------------

    def solve(
        self,
        anderson: bool = False,
        anderson_depth: int = 8,
    ) -> "WhamResult":
        """Run the WHAM self-consistency iteration.

        Parameters
        ----------
        anderson : bool
            If True, use Anderson/DIIS mixing to accelerate convergence.
            Particularly effective when many windows have uneven overlap.
        anderson_depth : int
            History depth for Anderson mixing (ignored if anderson=False).

        Returns
        -------
        WhamResult
        """
        K = self.n_windows
        if K == 0:
            raise RuntimeError("No windows have been added.")

        shape = self._grid_shape
        M_full = int(np.prod(shape))

        # 1. Stack histograms and biases into (K, M_full)
        hist_matrix = np.stack(
            [h.ravel() for h in self._histograms], axis=0
        )
        bias_matrix = np.stack(
            [b.ravel() for b in self._beta_biases], axis=0
        )

        # 2. Active-bin mask
        combined = hist_matrix.sum(axis=0)
        active_mask = combined > 0
        M = int(active_mask.sum())

        if M == 0:
            raise RuntimeError("All bins are empty across all windows.")

        # 3. Reduce to active bins
        hist_active = hist_matrix[:, active_mask]
        bias_active = bias_matrix[:, active_mask]

        with np.errstate(divide="ignore"):
            log_n = np.log(hist_active)

        N_k = hist_active.sum(axis=1)
        log_N = np.log(N_k)
        log_C = self._logsumexp_ax0_full(log_n)

        # 4. Warm-start
        f_k = np.zeros(K, dtype=np.float64)
        if self._free_energies is not None:
            n_prev = len(self._free_energies)
            n_copy = min(n_prev, K)
            f_k[:n_copy] = self._free_energies[:n_copy]
            f_k -= f_k[0]

        # 5. WHAM iteration
        if anderson:
            f_k, log_p, converged, n_iter, history = self._iterate_anderson(
                log_N, f_k, bias_active, log_C,
                self._tol, self._max_iter, anderson_depth,
            )
            # Fallback: if Anderson did not converge, finish with plain
            if not converged:
                remaining = self._max_iter - n_iter
                if remaining > 0:
                    warnings.warn(
                        f"Anderson mixing did not converge after {n_iter} "
                        f"iterations (Δf = {history[-1]:.2e}). "
                        f"Falling back to plain iteration for up to "
                        f"{remaining} more iterations.",
                        stacklevel=2,
                    )
                    f_k2, log_p2, conv2, n2, hist2 = self._iterate_plain(
                        log_N, f_k, bias_active, log_C,
                        self._tol, remaining,
                    )
                    f_k, log_p, converged = f_k2, log_p2, conv2
                    n_iter += n2
                    history = np.concatenate([history, hist2])
        else:
            f_k, log_p, converged, n_iter, history = self._iterate_plain(
                log_N, f_k, bias_active, log_C,
                self._tol, self._max_iter,
            )

        if not converged:
            warnings.warn(
                f"WHAM did not converge within {self._max_iter} iterations "
                f"(final Δf = {history[-1]:.2e}, tol = {self._tol:.2e}).",
                stacklevel=2,
            )

        # 6. Expand back to full grid
        log_prob_full = np.full(M_full, np.nan, dtype=np.float64)
        log_prob_full[active_mask] = log_p

        # 7. Bin volume correction
        if self._bin_volumes is not None:
            log_vol = np.log(self._bin_volumes.ravel())
            log_prob_full[active_mask] -= log_vol[active_mask]

        log_prob_full = log_prob_full.reshape(shape)

        # 8. Diagnostics
        overlap_hist = self._compute_histogram_overlap(hist_active, N_k)
        overlap_mat = self._compute_overlap_matrix(
            hist_active, bias_active, N_k, f_k
        )

        # 9. Effective sample size
        n_eff_active = self._compute_n_eff(hist_active, N_k)
        n_eff_full = np.full(M_full, np.nan, dtype=np.float64)
        n_eff_full[active_mask] = n_eff_active
        n_eff_full = n_eff_full.reshape(shape)

        # 10. Store state for warm-start
        self._free_energies = f_k.copy()
        self._log_prob = log_prob_full.copy()
        self._converged = converged
        self._n_iterations = n_iter
        self._convergence_history = history
        self._overlap_histogram = overlap_hist
        self._overlap_matrix = overlap_mat
        self._n_eff = n_eff_full
        self._errors = None          # invalidate cached errors
        self._dirty = False

        # 11. Build result
        return WhamResult(
            log_prob=log_prob_full,
            free_energies=f_k,
            bin_edges=[e.copy() for e in self._bin_edges] if self._bin_edges else None,
            bin_centers=[c.copy() for c in self._bin_centers] if self._bin_centers else None,
            bin_volumes=self._bin_volumes.copy() if self._bin_volumes is not None else None,
            converged=converged,
            n_iterations=n_iter,
            convergence_history=history,
            overlap_histogram=overlap_hist,
            overlap_matrix=overlap_mat,
            n_eff=n_eff_full,
            errors=self._errors,
        )

    def result(self) -> "WhamResult":
        """Return the most recent result."""
        if self._dirty or self._log_prob is None:
            raise RuntimeError("No valid result available.  Call solve() first.")
        return WhamResult(
            log_prob=self._log_prob.copy(),
            free_energies=self._free_energies.copy(),
            bin_edges=[e.copy() for e in self._bin_edges] if self._bin_edges else None,
            bin_centers=[c.copy() for c in self._bin_centers] if self._bin_centers else None,
            bin_volumes=self._bin_volumes.copy() if self._bin_volumes is not None else None,
            converged=self._converged,
            n_iterations=self._n_iterations,
            convergence_history=self._convergence_history.copy(),
            overlap_histogram=self._overlap_histogram.copy(),
            overlap_matrix=self._overlap_matrix.copy(),
            n_eff=self._n_eff.copy(),
            errors=self._errors,
        )

    # ---- error estimation ------------------------------------------------

    def estimate_errors(
        self,
        method: str = "analytical",
        n_bootstrap: int = 200,
        ci_level: float = 0.95,
        seed: Optional[int] = None,
        store_replicates: bool = False,
        verbose: bool = False,
    ) -> "WhamErrors":
        """Estimate uncertainties on the free energies and log-probabilities.

        Two methods are available:

        * ``"analytical"`` (default) — Computes the covariance of the free
          energy parameters from the Hessian (Fisher information matrix)
          of the WHAM negative log-likelihood, then propagates to per-bin
          uncertainties.  **Fast**: a single K×K matrix inversion,
          negligible compared to ``solve()``.  Assumes the asymptotic
          (large-sample) regime.

        * ``"bootstrap"`` — Multinomial resampling of histogram counts
          followed by repeated WHAM solves.  **Robust** and
          assumption-free, but computationally expensive (~
          ``n_bootstrap`` × cost of ``solve()``).

        The result is cached internally and attached to subsequent
        ``WhamResult`` objects (via ``result()`` or the next
        ``solve()``).  The cache is invalidated whenever ``solve()`` is
        called anew or windows are modified.

        Parameters
        ----------
        method : str
            ``"analytical"`` or ``"bootstrap"``.
        n_bootstrap : int
            Number of bootstrap replicates (only for ``"bootstrap"``).
        ci_level : float
            Confidence level for bootstrap confidence intervals
            (default 0.95, i.e. 95 %).
        seed : int, optional
            Random seed for bootstrap reproducibility.
        store_replicates : bool
            If True, keep all bootstrap replicate arrays (large).
        verbose : bool
            Print progress during bootstrap.

        Returns
        -------
        WhamErrors
        """

        if self._dirty or self._log_prob is None:
            print(f'{self._dirty=}, {self._log_prob=}')
            raise RuntimeError(
                "No valid result available.  Call solve() before "
                "estimate_errors()."
            )

        method_lower = method.lower()
        if method_lower == "analytical":
            errs = analytical_errors(self)
        elif method_lower == "bootstrap":
            errs = bootstrap_errors(
                self,
                n_bootstrap=n_bootstrap,
                ci_level=ci_level,
                seed=seed,
                store_replicates=store_replicates,
                verbose=verbose,
            )
        else:
            raise ValueError(
                f"Unknown error method {method!r}. "
                f"Use 'analytical' or 'bootstrap'."
            )

        self._errors = errs
        return errs

    # ---- iteration engines (private) -------------------------------------

    @staticmethod
    def _iterate_plain(
        log_N: np.ndarray,
        f_k: np.ndarray,
        bias_active: np.ndarray,
        log_C: np.ndarray,
        tol: float,
        max_iter: int,
    ) -> Tuple[np.ndarray, np.ndarray, bool, int, np.ndarray]:
        """Plain fixed-point WHAM iteration using BLAS matrix-vector products.

        Works in the linear (exp) domain.  Two BLAS dgemv calls replace
        3K element-wise numpy passes per iteration, giving ~5-10x
        per-iteration speedup for typical K and M.

        Returns (f_k, log_p, converged, n_iter, history).
        """
        K, M = bias_active.shape

        # --- Pre-compute constant quantities (outside iteration) ---
        N_k = np.exp(log_N)                                   # (K,)
        bias_row_min = bias_active.min(axis=1)                 # (K,)
        exp_neg_bias = np.exp(                                 # (K, M)
            -(bias_active - bias_row_min[:, None])
        )                                                      # entries in (0, 1]
        C = np.exp(log_C)                                      # (M,)

        history = np.empty(max_iter, dtype=np.float64)
        p = None

        for it in range(max_iter):
            # w_k = N_k * exp(f_k - bias_row_min)              (K,)
            w_k = N_k * np.exp(f_k - bias_row_min)

            # denom[i] = sum_k w_k[k] * exp_neg_bias[k, i]    BLAS dgemv
            denom = w_k @ exp_neg_bias                         # (M,)

            # p[i] = C[i] / denom[i]
            p = C / denom                                      # (M,)

            # s[k] = sum_i exp_neg_bias[k, i] * p[i]           BLAS dgemv
            s = exp_neg_bias @ p                               # (K,)

            # f_k_new = -log(s) + bias_row_min, anchored at k=0
            f_k_new = -np.log(s) + bias_row_min                # (K,)
            f_k_new -= f_k_new[0]

            delta = np.max(np.abs(f_k_new - f_k))
            history[it] = delta
            f_k = f_k_new

            if delta < tol:
                log_p = np.log(p)
                return f_k, log_p, True, it + 1, history[: it + 1]

        log_p = np.log(p) if p is not None else np.full(M, np.nan)
        return f_k, log_p, False, max_iter, history

    @staticmethod
    def _iterate_anderson(
        log_N: np.ndarray,
        f_k: np.ndarray,
        bias_active: np.ndarray,
        log_C: np.ndarray,
        tol: float,
        max_iter: int,
        depth: int = 8,
    ) -> Tuple[np.ndarray, np.ndarray, bool, int, np.ndarray]:
        """WHAM iteration with Anderson/DIIS acceleration (BLAS-accelerated).

        Uses BLAS dgemv for the heavy per-iteration work and eliminates
        the expensive safeguard evaluation that previously performed a
        *second* full WHAM step each Anderson iteration.  A lightweight
        check (finiteness + overflow guard) replaces it, halving
        per-iteration cost while preserving robustness.

        Returns (f_k, log_p, converged, n_iter, history).
        """
        K, M = bias_active.shape
        m = depth

        # --- Pre-compute constant quantities ---
        N_k = np.exp(log_N)
        bias_row_min = bias_active.min(axis=1)
        exp_neg_bias = np.exp(-(bias_active - bias_row_min[:, None]))
        C = np.exp(log_C)

        history = np.empty(max_iter, dtype=np.float64)

        # Anderson ring buffer
        F_hist = np.zeros((m, K), dtype=np.float64)
        R_hist = np.zeros((m, K), dtype=np.float64)
        n_stored = 0
        p = None
        consecutive_rejects = 0  # track consecutive Anderson rejections
        max_rejects = 3 * m      # give up on Anderson after this many

        for it in range(max_iter):
            # --- One BLAS-accelerated WHAM step ---
            w_k = N_k * np.exp(f_k - bias_row_min)
            denom = w_k @ exp_neg_bias
            p = C / denom
            s = exp_neg_bias @ p
            g_k = -np.log(s) + bias_row_min
            g_k -= g_k[0]

            r_k = g_k - f_k
            delta = np.max(np.abs(r_k))
            history[it] = delta

            if delta < tol:
                log_p = np.log(p)
                return g_k, log_p, True, it + 1, history[: it + 1]

            # Store in ring buffer
            idx = n_stored % m
            F_hist[idx] = f_k
            R_hist[idx] = r_k
            n_stored += 1

            if n_stored < 2:
                f_k = g_k
                continue

            # Anderson extrapolation
            f_candidate = _anderson_extrapolate(
                F_hist, R_hist, n_stored, m, r_k
            )

            if (
                f_candidate is not None
                and np.all(np.isfinite(f_candidate))
                and np.max(f_candidate - bias_row_min) < 700  # overflow guard
            ):
                f_k = f_candidate
                consecutive_rejects = 0
            else:
                f_k = g_k
                consecutive_rejects += 1
                # Reset history on numerical failure
                if f_candidate is not None and not np.all(
                    np.isfinite(f_candidate)
                ):
                    n_stored = 0

            # If Anderson is consistently unhelpful, bail out early so
            # the caller (solve()) can fall back to plain iteration.
            if consecutive_rejects >= max_rejects:
                log_p = np.log(p)
                return f_k, log_p, False, it + 1, history[: it + 1]

        # Final log_p
        log_p = np.log(p) if p is not None else np.full(M, np.nan)
        return f_k, log_p, False, max_iter, history

    # ---- fast logsumexp (for non-iteration use) --------------------------

    @staticmethod
    def _logsumexp_ax0_full(arr: np.ndarray) -> np.ndarray:
        """logsumexp along axis=0: (K, M) -> (M,). Pure numpy."""
        mx = arr.max(axis=0)
        return mx + np.log(np.exp(arr - mx[None, :]).sum(axis=0))

    # ---- overlap diagnostics ---------------------------------------------

    @staticmethod
    def _compute_histogram_overlap(
        hist_active: np.ndarray, N_k: np.ndarray
    ) -> np.ndarray:
        """Pairwise histogram overlap (K, K).

        Vectorised: the inner pair-loop is replaced by a single
        ``np.minimum`` broadcast per row, reducing Python-loop
        iterations from K*(K-1)/2 to K.
        """
        K = hist_active.shape[0]
        with np.errstate(divide="ignore", invalid="ignore"):
            normed = hist_active / N_k[:, None]  # (K, M)
        overlap = np.empty((K, K), dtype=np.float64)
        for k in range(K):
            overlap[k, k] = 1.0
            if k + 1 < K:
                # Broadcast min over all remaining rows at once
                mins = np.minimum(normed[k], normed[k + 1 :])  # (K-k-1, M)
                row_ov = mins.sum(axis=1)                       # (K-k-1,)
                overlap[k, k + 1 :] = row_ov
                overlap[k + 1 :, k] = row_ov
        return overlap

    @staticmethod
    def _compute_overlap_matrix(
        hist_active: np.ndarray,
        bias_active: np.ndarray,
        N_k: np.ndarray,
        f_k: np.ndarray,
    ) -> np.ndarray:
        """Shirts-Chodera statistical overlap matrix (K, K).

        Fully vectorised using the same BLAS approach as the iteration:
        precompute exp(-bias) once, compute the weight matrix w with
        element-wise + one BLAS gemv, then finish with BLAS dgemm.
        """
        K, M = hist_active.shape
        bias_row_min = bias_active.min(axis=1)                     # (K,)
        exp_neg_bias = np.exp(-(bias_active - bias_row_min[:, None]))  # (K, M)
        w_k = N_k * np.exp(f_k - bias_row_min)                    # (K,)
        denom = w_k @ exp_neg_bias                                 # (M,)
        # w[k,i] = w_k[k] * B[k,i] / denom[i]
        w = (w_k[:, None] * exp_neg_bias)                          # (K, M)
        w /= denom[None, :]                                        # in-place
        return w @ w.T                                             # BLAS dgemm

    @staticmethod
    def _compute_n_eff(
        hist_active: np.ndarray, N_k: np.ndarray
    ) -> np.ndarray:
        """Effective sample size per active bin."""
        C = hist_active.sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            denom = (hist_active ** 2 / N_k[:, None]).sum(axis=0)
            n_eff = np.where(denom > 0, C ** 2 / denom, 0.0)
        return n_eff

    def check_overlap(self, window_index: int = -1) -> dict:
        """Check overlap of one window against all others."""
        if window_index < 0:
            window_index = self.n_windows + window_index
        if self.n_windows < 2:
            return {
                "sufficient": False, "max_overlap": 0.0,
                "overlap_with": [], "threshold": self._overlap_threshold,
            }

        K = self.n_windows
        hist_matrix = np.stack([h.ravel() for h in self._histograms], axis=0)
        combined = hist_matrix.sum(axis=0)
        active_mask = combined > 0
        hist_active = hist_matrix[:, active_mask]
        N_k = hist_active.sum(axis=1)

        with np.errstate(divide="ignore", invalid="ignore"):
            normed = hist_active / N_k[:, None]

        target = normed[window_index]
        pairs = []
        for k in range(K):
            if k == window_index:
                continue
            val = float(np.sum(np.minimum(target, normed[k])))
            pairs.append((k, val))
        pairs.sort(key=lambda x: x[1], reverse=True)
        max_ov = pairs[0][1] if pairs else 0.0
        return {
            "sufficient": max_ov >= self._overlap_threshold,
            "max_overlap": max_ov,
            "overlap_with": pairs,
            "threshold": self._overlap_threshold,
        }

    # ---- serialization ---------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        """Save the complete solver state to a single ``.npz`` file."""
        path = Path(path)
        data = {}
        data["tol"] = np.array(self._tol)
        data["max_iter"] = np.array(self._max_iter)
        data["lazy_solve"] = np.array(self._lazy_solve)
        data["overlap_threshold"] = np.array(self._overlap_threshold)

        if self._grid_shape is not None:
            data["grid_shape"] = np.array(self._grid_shape)
        if self._bin_edges is not None:
            data["n_dims_edges"] = np.array(len(self._bin_edges))
            for i, e in enumerate(self._bin_edges):
                data[f"bin_edges_{i}"] = e
        if self._bin_centers is not None:
            data["n_dims_centers"] = np.array(len(self._bin_centers))
            for i, c in enumerate(self._bin_centers):
                data[f"bin_centers_{i}"] = c
        if self._bin_widths is not None:
            data["n_dims_widths"] = np.array(len(self._bin_widths))
            for i, w in enumerate(self._bin_widths):
                data[f"bin_widths_{i}"] = w
        if self._bin_volumes is not None:
            data["bin_volumes"] = self._bin_volumes

        K = self.n_windows
        data["n_windows"] = np.array(K)
        if K > 0:
            data["histograms"] = np.stack([h.ravel() for h in self._histograms], axis=0)
            data["beta_biases"] = np.stack([b.ravel() for b in self._beta_biases], axis=0)

        data["dirty"] = np.array(self._dirty)
        data["converged"] = np.array(self._converged)
        data["n_iterations"] = np.array(self._n_iterations)
        if self._free_energies is not None:
            data["free_energies"] = self._free_energies
        if self._log_prob is not None:
            data["log_prob"] = self._log_prob
        if self._convergence_history is not None:
            data["convergence_history"] = self._convergence_history
        if self._overlap_histogram is not None:
            data["overlap_histogram"] = self._overlap_histogram
        if self._overlap_matrix is not None:
            data["overlap_matrix"] = self._overlap_matrix
        if self._n_eff is not None:
            data["n_eff"] = self._n_eff

        np.savez_compressed(str(path), **data)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "WhamSolver":
        """Reconstruct a ``WhamSolver`` from a saved ``.npz`` file."""
        path = Path(path)
        if not path.suffix:
            path = path.with_suffix(".npz")
        with np.load(str(path), allow_pickle=False) as f:
            solver = cls(
                tol=float(f["tol"]),
                max_iter=int(f["max_iter"]),
                lazy=bool(f["lazy_solve"]),
                overlap_threshold=float(f["overlap_threshold"]),
            )
            if "grid_shape" in f:
                solver._grid_shape = tuple(int(x) for x in f["grid_shape"])
            if "n_dims_edges" in f:
                n = int(f["n_dims_edges"])
                solver._bin_edges = [f[f"bin_edges_{i}"] for i in range(n)]
            if "n_dims_centers" in f:
                n = int(f["n_dims_centers"])
                solver._bin_centers = [f[f"bin_centers_{i}"] for i in range(n)]
            if "n_dims_widths" in f:
                n = int(f["n_dims_widths"])
                solver._bin_widths = [f[f"bin_widths_{i}"] for i in range(n)]
            if "bin_volumes" in f:
                solver._bin_volumes = f["bin_volumes"]

            K = int(f["n_windows"])
            if K > 0:
                shape = solver._grid_shape
                hists = f["histograms"]
                biases = f["beta_biases"]
                for k in range(K):
                    solver._histograms.append(hists[k].reshape(shape))
                    solver._beta_biases.append(biases[k].reshape(shape))

            solver._dirty = bool(f["dirty"])
            solver._converged = bool(f["converged"])
            solver._n_iterations = int(f["n_iterations"])
            if "free_energies" in f:
                solver._free_energies = f["free_energies"]
            if "log_prob" in f:
                solver._log_prob = f["log_prob"]
            if "convergence_history" in f:
                solver._convergence_history = f["convergence_history"]
            if "overlap_histogram" in f:
                solver._overlap_histogram = f["overlap_histogram"]
            if "overlap_matrix" in f:
                solver._overlap_matrix = f["overlap_matrix"]
            if "n_eff" in f:
                solver._n_eff = f["n_eff"]

        return solver

    # ---- plotting ---------------------------------------------------------

    def _detect_unbiased_window(self, atol: float = 1e-10) -> Optional[int]:
        """Return the index of an unbiased window, or *None*.

        A window is considered unbiased if its bias array is constant
        (or zero) across all bins — i.e.
        ``max(bias) - min(bias) < atol``.
        """
        for k, bias in enumerate(self._beta_biases):
            bias_range = bias.max() - bias.min()
            if bias_range < atol:
                return k
        return None

    def plot_marginal(
        self,
        dim: int = 0,
        *,
        unbiased_histogram: Optional[np.ndarray] = None,
        savefn: Optional[Union[str, Path]] = None,
        figsize: Tuple[float, float] = (10, 12),
        cmap: str = "tab10",
        count_threshold: float = 0.01,
    ):
        """Plot marginalized probability, free energy, and biased histograms.

        Creates a three-row figure:

        * **Row 1** — marginalized unbiased probability :math:`p(x_d)`.
        * **Row 2** — marginalized free energy
          :math:`-\\ln p(x_d)` (per-window curves aligned to the
          WHAM combined curve).
        * **Row 3** — raw biased (observed) histogram marginals.

        Per-window curves are obtained by reweighting each window's
        histogram with the converged free energies.  Bins where the
        marginal count falls below *count_threshold* × max(marginal
        count for that window) are masked to suppress noise-amplification
        artefacts at the edges of each window's support.

        If any window has a constant (or zero) bias it is automatically
        identified as the unbiased window and plotted as a dashed black
        reference line (labelled "Unbiased") in all three subplots.
        An explicitly supplied *unbiased_histogram* takes precedence.

        Parameters
        ----------
        dim : int
            Dimension index to keep (all other dimensions are summed
            over).
        unbiased_histogram : np.ndarray, optional
            A histogram (same grid shape as the windows) representing the
            known unbiased distribution.  If provided, it is normalised
            and shown as a dashed black reference.  If *None*, the method
            attempts to auto-detect an unbiased window among the added
            windows.
        savefn : str or Path, optional
            Base filename (without extension).  The figure is saved as
            ``<savefn>.pdf``, ``<savefn>.png``, and ``<savefn>.svg``.
        figsize : tuple of float
            Figure size in inches, ``(width, height)``.
        cmap : str
            Matplotlib colormap name used for window colours.
        count_threshold : float
            Fraction of the per-window maximum marginal count below which
            bins are masked (set to NaN) before plotting.  Suppresses
            spurious spikes from noise amplification at window edges.
            Default 0.01 (1 %).

        Returns
        -------
        fig : matplotlib.figure.Figure
        axes : tuple of matplotlib.axes.Axes
            ``(ax_prob, ax_fe, ax_hist)``
        """
        import matplotlib.pyplot as plt

        if self._log_prob is None or self._dirty:
            raise RuntimeError(
                "No valid result available.  Call solve() first."
            )
        if self._bin_centers is None:
            raise RuntimeError(
                "bin_centers must be set (via set_bin_edges or "
                "set_bin_centers) before plotting."
            )

        ndim = len(self._grid_shape)
        if dim < 0 or dim >= ndim:
            raise ValueError(
                f"dim={dim} out of range for {ndim}-dimensional grid."
            )

        centers = self._bin_centers[dim]
        K = self.n_windows
        sum_axes = tuple(a for a in range(ndim) if a != dim)

        # --- Auto-detect unbiased window ---
        unbiased_idx: Optional[int] = None
        if unbiased_histogram is None:
            unbiased_idx = self._detect_unbiased_window()

        # --- Marginalize full WHAM log_prob ---
        log_prob = self._log_prob.copy()
        prob_full = np.exp(log_prob - np.nanmax(log_prob))
        prob_full = np.where(np.isnan(log_prob), 0.0, prob_full)
        prob_marginal = prob_full.sum(axis=sum_axes)
        prob_marginal /= prob_marginal.sum()

        # --- Marginalised error bands (if errors are cached) ---
        # Var(p_marg) = sum_{other axes} p_i^2 * Var(ln p_i)  (delta method)
        # then  std(F_marg) = std(p_marg) / p_marg   where F = -ln p
        prob_marginal_std = None
        fe_marginal_std = None
        ci_z = 1.96  # 95 % confidence
        if self._errors is not None:
            var_lp = self._errors.var_log_prob.copy()
            var_lp = np.where(np.isnan(var_lp), 0.0, var_lp)
            # Var(p_i) ≈ p_i^2 * Var(ln p_i)   (for the un-normalised p)
            var_p = prob_full ** 2 * var_lp
            var_p_marginal = var_p.sum(axis=sum_axes)
            # Normalise:  prob_marginal was normalised by total = prob_full.sum()
            total = prob_full.sum()
            if total > 0:
                var_p_marginal /= total ** 2
            prob_marginal_std = np.sqrt(np.maximum(var_p_marginal, 0.0))
            # For the free energy: std(-ln p) = std(p) / p
            with np.errstate(divide="ignore", invalid="ignore"):
                fe_marginal_std = np.where(
                    prob_marginal > 0,
                    prob_marginal_std / prob_marginal,
                    np.nan,
                )

        # --- Per-window reweighted marginals ---
        f_k = self._free_energies  # (K,)
        per_window_marginals = []       # normalised, with NaN masking
        per_window_raw_marginals = []   # raw biased histogram marginals
        for k in range(K):
            hist_k = self._histograms[k].astype(np.float64)
            bias_k = self._beta_biases[k]

            # Raw biased histogram marginal (for 3rd plot)
            raw_marg = hist_k.sum(axis=sum_axes)
            per_window_raw_marginals.append(raw_marg)

            # Reweight: p_unbiased_k(i) ∝ n_k(i) * exp(bias_k(i) + f_k[k])
            with np.errstate(divide="ignore"):
                log_w = np.log(np.maximum(hist_k, 0.0)) + bias_k + f_k[k]
            log_w = np.where(hist_k > 0, log_w, np.nan)
            w = np.exp(log_w - np.nanmax(log_w))
            w = np.where(np.isnan(log_w), 0.0, w)
            marg = w.sum(axis=sum_axes)
            total = marg.sum()
            if total > 0:
                marg /= total

            # Mask low-count bins to suppress noise-amplification spikes.
            # Use the RAW histogram marginal as the reliability indicator:
            # bins where the raw count is < count_threshold * max(raw)
            # have too little data for a trustworthy reweighted estimate.
            raw_max = raw_marg.max()
            if raw_max > 0:
                unreliable = raw_marg < count_threshold * raw_max
                marg = np.where(unreliable, np.nan, marg)

            per_window_marginals.append(marg)

        # --- Unbiased reference ---
        unbiased_marginal = None
        if unbiased_histogram is not None:
            ub = np.asarray(unbiased_histogram, dtype=np.float64)
            ub_marg = ub.sum(axis=sum_axes)
            ub_total = ub_marg.sum()
            if ub_total > 0:
                ub_marg /= ub_total
            unbiased_marginal = ub_marg
        elif unbiased_idx is not None:
            # Use the auto-detected unbiased window's raw histogram
            ub_marg = per_window_raw_marginals[unbiased_idx].copy()
            ub_total = ub_marg.sum()
            if ub_total > 0:
                ub_marg /= ub_total
            unbiased_marginal = ub_marg

        # --- Colours (skip unbiased_idx so it gets black instead) ---
        cm = plt.get_cmap(cmap)
        n_coloured = K if unbiased_idx is None else K - 1
        colours = []
        ci = 0
        for k in range(K):
            if k == unbiased_idx:
                colours.append("black")  # placeholder, will use dashed
            else:
                colours.append(cm(ci / max(n_coloured, 1)))
                ci += 1

        # --- Figure (3 rows) ---
        fig, (ax_prob, ax_fe, ax_hist) = plt.subplots(
            3, 1, figsize=figsize, sharex=True,
        )
        fig.subplots_adjust(hspace=0.08)

        # ============================================================
        # Row 1: Probability
        # ============================================================
        for k in range(K):
            if k == unbiased_idx:
                continue  # plotted separately as dashed black
            ax_prob.plot(
                centers, per_window_marginals[k],
                color=colours[k], alpha=0.6, linewidth=0.8,
                label=f"Window {k}",
            )
        ax_prob.plot(
            centers, prob_marginal,
            color="black", linewidth=2.0, label="WHAM combined",
        )
        if prob_marginal_std is not None:
            lo = np.maximum(prob_marginal - ci_z * prob_marginal_std, 0.0)
            hi = prob_marginal + ci_z * prob_marginal_std
            ax_prob.fill_between(
                centers, lo, hi,
                color="black", alpha=0.15, label="95 % CI",
            )
        if unbiased_marginal is not None:
            lbl = (f"Unbiased (win {unbiased_idx})"
                   if unbiased_idx is not None else "Unbiased")
            ax_prob.plot(
                centers, unbiased_marginal,
                color="black", linewidth=1.5, linestyle="--",
                label=lbl,
            )
        ax_prob.set_ylabel(r"$p\,(x_{" + str(dim) + r"})$", fontsize=13)
        ax_prob.legend(
            fontsize=7, ncol=max(1, (K + 2) // 6),
            loc="upper right", framealpha=0.8,
        )
        ax_prob.set_title(
            f"Marginalised probability along dimension {dim}",
            fontsize=14,
        )
        ax_prob.tick_params(labelbottom=False)

        # ============================================================
        # Row 2: Free energy = -ln p, aligned to WHAM combined
        # ============================================================
        with np.errstate(divide="ignore"):
            fe_marginal = -np.log(
                np.where(prob_marginal > 0, prob_marginal, np.nan)
            )
        fe_marginal -= np.nanmin(fe_marginal)

        for k in range(K):
            if k == unbiased_idx:
                continue
            mk = per_window_marginals[k]
            with np.errstate(divide="ignore"):
                fe_k = -np.log(np.where(mk > 0, mk, np.nan))

            # Align to WHAM curve: compute median offset in overlap region
            valid = np.isfinite(fe_k) & np.isfinite(fe_marginal)
            if valid.any():
                shift = np.median(fe_marginal[valid] - fe_k[valid])
                fe_k += shift
            else:
                fe_k -= np.nanmin(fe_k)  # fallback: self-shift

            ax_fe.plot(
                centers, fe_k,
                color=colours[k], alpha=0.6, linewidth=0.8,
                label=f"Window {k}",
            )

        ax_fe.plot(
            centers, fe_marginal,
            color="black", linewidth=2.0, label="WHAM combined",
        )
        if fe_marginal_std is not None:
            fe_lo = fe_marginal - ci_z * fe_marginal_std
            fe_hi = fe_marginal + ci_z * fe_marginal_std
            ax_fe.fill_between(
                centers, fe_lo, fe_hi,
                color="black", alpha=0.15, label="95 % CI",
            )
        if unbiased_marginal is not None:
            with np.errstate(divide="ignore"):
                fe_ub = -np.log(
                    np.where(unbiased_marginal > 0, unbiased_marginal, np.nan)
                )
            # Align unbiased curve to WHAM the same way
            valid_ub = np.isfinite(fe_ub) & np.isfinite(fe_marginal)
            if valid_ub.any():
                shift_ub = np.median(fe_marginal[valid_ub] - fe_ub[valid_ub])
                fe_ub += shift_ub
            else:
                fe_ub -= np.nanmin(fe_ub)
            lbl = (f"Unbiased (win {unbiased_idx})"
                   if unbiased_idx is not None else "Unbiased")
            ax_fe.plot(
                centers, fe_ub,
                color="black", linewidth=1.5, linestyle="--",
                label=lbl,
            )
        ax_fe.set_ylabel(
            r"$-\ln\, p\,(x_{" + str(dim) + r"})$  [kT]", fontsize=13,
        )
        ax_fe.tick_params(labelbottom=False)

        # ============================================================
        # Row 3: Raw biased (observed) histogram marginals
        # ============================================================
        for k in range(K):
            raw = per_window_raw_marginals[k]
            raw_norm = raw / raw.sum() if raw.sum() > 0 else raw
            if k == unbiased_idx:
                continue  # plotted separately
            ax_hist.plot(
                centers, raw_norm,
                color=colours[k], alpha=0.6, linewidth=0.8,
                label=f"Window {k}",
            )
        if unbiased_idx is not None:
            raw_ub = per_window_raw_marginals[unbiased_idx]
            raw_ub_norm = raw_ub / raw_ub.sum() if raw_ub.sum() > 0 else raw_ub
            ax_hist.plot(
                centers, raw_ub_norm,
                color="black", linewidth=1.5, linestyle="--",
                label=f"Unbiased (win {unbiased_idx})",
            )
        elif unbiased_marginal is not None:
            ax_hist.plot(
                centers, unbiased_marginal,
                color="black", linewidth=1.5, linestyle="--",
                label="Unbiased",
            )
        ax_hist.set_ylabel(
            r"Biased $p\,(x_{" + str(dim) + r"})$", fontsize=13,
        )
        ax_hist.set_xlabel(
            r"$x_{" + str(dim) + r"}$", fontsize=13,
        )
        ax_hist.legend(
            fontsize=7, ncol=max(1, (K + 2) // 6),
            loc="upper right", framealpha=0.8,
        )
        ax_hist.set_title(
            f"Biased histogram marginals along dimension {dim}",
            fontsize=14,
        )

        for ax in (ax_prob, ax_fe, ax_hist):
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        fig.tight_layout()

        # --- Save ---
        if savefn is not None:
            savefn = Path(savefn)
            for ext in ("pdf", "png", "svg"):
                fig.savefig(
                    str(savefn.with_suffix(f".{ext}")),
                    transparent=True if ext in ("pdf", "svg") else False,
                    bbox_inches="tight",
                    dpi=300,
                )
            plt.close(fig)
        else:
            plt.show()

        return fig, (ax_prob, ax_fe, ax_hist)

    def __repr__(self) -> str:
        status = "converged" if self._converged and not self._dirty else "dirty"
        return (
            f"WhamSolver(n_windows={self.n_windows}, "
            f"grid_shape={self._grid_shape}, status={status})"
        )
