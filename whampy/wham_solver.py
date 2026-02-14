"""
Weighted Histogram Analysis Method (WHAM) Solver

General-purpose WHAM implementation supporting arbitrary dimensionality,
progressive window addition with warm-start convergence, overlap diagnostics,
and full serialization. Designed for umbrella sampling of DNA cyclization
simulations but applicable to any umbrella sampling problem.

All bias potentials are expected in dimensionless units (βU, i.e., units of kT).
All internal arithmetic is performed in log-space using logsumexp for numerical
stability.
"""

from __future__ import annotations

import warnings
import numpy as np
from scipy.special import logsumexp
from dataclasses import dataclass
from typing import Optional, Callable, List, Tuple, Union
from pathlib import Path


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
        self._dirty: bool = True  # True when state may not match stored windows

        # settings
        self._tol: float = tol
        self._max_iter: int = max_iter
        self._lazy_solve: bool = lazy
        self._overlap_threshold: float = overlap_threshold

        if bin_edges is not None:
            self.set_bin_edges(bin_edges)

    # ---- grid definition -------------------------------------------------

    def set_bin_edges(self, bin_edges: List[np.ndarray]) -> None:
        """Set bin edges for all dimensions.

        Computes bin centers and widths.  Raises if windows already exist
        with a different grid shape.
        """
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
        # compute default uniform volumes
        meshwidths = np.meshgrid(*widths, indexing="ij")
        self._bin_volumes = np.ones(shape, dtype=np.float64)
        for mw in meshwidths:
            self._bin_volumes *= mw

    def set_bin_centers(self, bin_centers: List[np.ndarray]) -> None:
        """Set bin centers directly.

        Bin widths are inferred assuming uniform spacing per dimension.
        """
        bin_centers = [np.asarray(c, dtype=np.float64) for c in bin_centers]
        shape = tuple(len(c) for c in bin_centers)
        if self._grid_shape is not None and shape != self._grid_shape:
            raise ValueError(
                f"Grid shape {shape} from bin_centers is inconsistent with "
                f"existing grid shape {self._grid_shape}."
            )
        self._bin_centers = bin_centers
        self._grid_shape = shape
        # infer widths (uniform spacing)
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
        """Set bin volumes for Jacobian correction.

        Parameters
        ----------
        volumes : np.ndarray, optional
            Pre-computed volume array of shape grid_shape.
        volume_function : callable, optional
            ``volume_function(*meshgrid_coords) -> np.ndarray``.
            Evaluated immediately at bin centers; only the resulting array
            is stored.

        Exactly one of the two must be provided.
        """
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
        """Number of stored windows."""
        return len(self._histograms)

    @property
    def grid_shape(self) -> Optional[tuple]:
        """Shape of the histogram grid, or None if not yet established."""
        return self._grid_shape

    def add_window(
        self,
        histogram: np.ndarray,
        bias_array: Optional[np.ndarray] = None,
        bias_function: Optional[Callable] = None,
    ) -> int:
        """Add a single umbrella-sampling window.

        Parameters
        ----------
        histogram : np.ndarray
            Bin counts, shape = grid_shape.
        bias_array : np.ndarray, optional
            Pre-computed βU at bin centers, shape = grid_shape.
        bias_function : callable, optional
            ``bias_function(*meshgrid_coords) -> np.ndarray`` returning βU.
            Requires bin_centers to be set.

        Returns
        -------
        int
            Index of the newly added window.
        """
        histogram = np.asarray(histogram, dtype=np.float64)

        # establish or check grid shape
        if self._grid_shape is None:
            self._grid_shape = histogram.shape
        elif histogram.shape != self._grid_shape:
            raise ValueError(
                f"Histogram shape {histogram.shape} != grid shape {self._grid_shape}."
            )

        # resolve bias
        if bias_array is not None:
            bias = np.asarray(bias_array, dtype=np.float64)
            if bias_function is not None:
                warnings.warn(
                    "Both bias_array and bias_function provided; "
                    "using bias_array.",
                    stacklevel=2,
                )
        elif bias_function is not None:
            if self._bin_centers is None:
                raise RuntimeError(
                    "bin_centers must be set before using bias_function."
                )
            coords = np.meshgrid(*self._bin_centers, indexing="ij")
            bias = np.asarray(bias_function(*coords), dtype=np.float64)
        else:
            raise ValueError(
                "Provide either 'bias_array' or 'bias_function'."
            )

        if bias.shape != self._grid_shape:
            raise ValueError(
                f"Bias shape {bias.shape} != grid shape {self._grid_shape}."
            )

        self._histograms.append(histogram)
        self._beta_biases.append(bias)
        self._dirty = True

        idx = len(self._histograms) - 1

        if not self._lazy_solve and self.n_windows >= 1:
            self.solve()

        return idx

    def add_windows(
        self, windows: List[Tuple[np.ndarray, np.ndarray]]
    ) -> List[int]:
        """Add multiple windows from (histogram, bias_array) pairs.

        Parameters
        ----------
        windows : list of (histogram, bias_array) tuples

        Returns
        -------
        list of int
            Indices of the added windows.
        """
        was_lazy = self._lazy_solve
        # temporarily force lazy to avoid repeated solves
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
        # invalidate and adjust stored free energies
        if self._free_energies is not None and len(self._free_energies) > 0:
            fe = list(self._free_energies)
            if index < len(fe):
                fe.pop(index)
            self._free_energies = np.array(fe) if fe else None
        self._dirty = True

    def remove_last_window(self) -> None:
        """Remove the most recently added window."""
        if self.n_windows == 0:
            raise RuntimeError("No windows to remove.")
        self.remove_window(self.n_windows - 1)

    def replace_window(
        self,
        index: int,
        histogram: np.ndarray,
        bias_array: Optional[np.ndarray] = None,
        bias_function: Optional[Callable] = None,
    ) -> None:
        """Replace the window at *index* (preserves ordering)."""
        if index < 0 or index >= self.n_windows:
            raise IndexError(f"Window index {index} out of range [0, {self.n_windows}).")
        # validate new data (temporarily add, then swap)
        histogram = np.asarray(histogram, dtype=np.float64)
        if histogram.shape != self._grid_shape:
            raise ValueError(
                f"Histogram shape {histogram.shape} != grid shape {self._grid_shape}."
            )
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
        """Set eager mode (solve on every add_window)."""
        self._lazy_solve = not eager

    def set_lazy(self, lazy: bool = True) -> None:
        """Set lazy mode (manual solve)."""
        self._lazy_solve = lazy

    def set_tol(self, tol: float) -> None:
        self._tol = tol

    def set_max_iter(self, max_iter: int) -> None:
        self._max_iter = max_iter

    def set_overlap_threshold(self, threshold: float) -> None:
        self._overlap_threshold = threshold

    # ---- core WHAM -------------------------------------------------------

    def solve(self) -> "WhamResult":
        """Run the WHAM self-consistency iteration.

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
        )  # (K, M_full)
        bias_matrix = np.stack(
            [b.ravel() for b in self._beta_biases], axis=0
        )  # (K, M_full)

        # 2. Active-bin mask: bins where combined count > 0
        combined = hist_matrix.sum(axis=0)  # (M_full,)
        active_mask = combined > 0  # bool, (M_full,)
        M = int(active_mask.sum())

        if M == 0:
            raise RuntimeError("All bins are empty across all windows.")

        # 3. Reduce to active bins
        hist_active = hist_matrix[:, active_mask]   # (K, M)
        bias_active = bias_matrix[:, active_mask]   # (K, M)

        # Precompute log quantities
        # Protect against log(0): bins with zero counts in individual windows
        with np.errstate(divide="ignore"):
            log_n = np.log(hist_active)  # (K, M), may contain -inf

        N_k = hist_active.sum(axis=1)  # (K,)
        log_N = np.log(N_k)            # (K,)
        log_C = logsumexp(log_n, axis=0)  # (M,) total log-counts per bin

        # 4. Warm-start
        f_k = np.zeros(K, dtype=np.float64)
        if self._free_energies is not None:
            n_prev = len(self._free_energies)
            n_copy = min(n_prev, K)
            f_k[:n_copy] = self._free_energies[:n_copy]
            f_k -= f_k[0]  # re-anchor

        # 5. WHAM iteration
        converged = False
        history = np.empty(self._max_iter, dtype=np.float64)

        for it in range(self._max_iter):
            # log_p(i) = log_C(i) - logsumexp_k[ log_N(k) + f_k(k) - beta_U(k,i) ]
            log_denom = logsumexp(
                log_N[:, None] + f_k[:, None] - bias_active,  # (K, M)
                axis=0,
            )  # (M,)
            log_p = log_C - log_denom  # (M,)

            # f_k(k) = -logsumexp_i[ log_p(i) - beta_U(k,i) ]
            f_k_new = -logsumexp(
                log_p[None, :] - bias_active,  # (K, M)
                axis=1,
            )  # (K,)
            f_k_new -= f_k_new[0]  # fix reference

            delta = np.max(np.abs(f_k_new - f_k))
            history[it] = delta
            f_k = f_k_new

            if delta < self._tol:
                converged = True
                n_iter = it + 1
                break
        else:
            n_iter = self._max_iter
            warnings.warn(
                f"WHAM did not converge within {self._max_iter} iterations "
                f"(final Δf = {delta:.2e}, tol = {self._tol:.2e}).",
                stacklevel=2,
            )

        history = history[:n_iter]

        # Final log_p with converged f_k
        log_denom = logsumexp(
            log_N[:, None] + f_k[:, None] - bias_active,
            axis=0,
        )
        log_p = log_C - log_denom

        # 6. Expand back to full grid
        log_prob_full = np.full(M_full, np.nan, dtype=np.float64)
        log_prob_full[active_mask] = log_p

        # 7. Apply bin volume correction (probability -> density)
        if self._bin_volumes is not None:
            log_vol = np.log(self._bin_volumes.ravel())
            # only adjust active bins
            log_prob_full[active_mask] -= log_vol[active_mask]

        log_prob_full = log_prob_full.reshape(shape)

        # 8. Overlap diagnostics
        overlap_hist = self._compute_histogram_overlap(hist_active, N_k)
        overlap_mat = self._compute_overlap_matrix(
            hist_active, bias_active, N_k, f_k
        )

        # 9. Effective sample size
        n_eff_active = self._compute_n_eff(hist_active, N_k)
        n_eff_full = np.full(M_full, np.nan, dtype=np.float64)
        n_eff_full[active_mask] = n_eff_active
        n_eff_full = n_eff_full.reshape(shape)

        # 10. Store converged state for warm-start
        self._free_energies = f_k.copy()
        self._log_prob = log_prob_full.copy()
        self._converged = converged
        self._n_iterations = n_iter
        self._convergence_history = history
        self._overlap_histogram = overlap_hist
        self._overlap_matrix = overlap_mat
        self._n_eff = n_eff_full
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
        )

    def result(self) -> "WhamResult":
        """Return the most recent result.

        Raises
        ------
        RuntimeError
            If ``solve()`` has not been called or state is dirty.
        """
        if self._dirty or self._log_prob is None:
            raise RuntimeError(
                "No valid result available.  Call solve() first."
            )
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
        )

    # ---- overlap diagnostics ---------------------------------------------

    @staticmethod
    def _compute_histogram_overlap(
        hist_active: np.ndarray, N_k: np.ndarray
    ) -> np.ndarray:
        """Pairwise histogram overlap (K, K).

        overlap(k, l) = sum_i min(n_ki/N_k, n_li/N_l).
        """
        K = hist_active.shape[0]
        # Normalize each window's histogram
        with np.errstate(divide="ignore", invalid="ignore"):
            normed = hist_active / N_k[:, None]  # (K, M)
        overlap = np.empty((K, K), dtype=np.float64)
        for k in range(K):
            overlap[k, k] = 1.0
            for l in range(k + 1, K):
                val = np.sum(np.minimum(normed[k], normed[l]))
                overlap[k, l] = val
                overlap[l, k] = val
        return overlap

    @staticmethod
    def _compute_overlap_matrix(
        hist_active: np.ndarray,
        bias_active: np.ndarray,
        N_k: np.ndarray,
        f_k: np.ndarray,
    ) -> np.ndarray:
        """Shirts-Chodera statistical overlap matrix (K, K).

        O_{kl} = sum_i w_k(i) * w_l(i)
        where w_k(i) = N_k exp(f_k - beta_U_k(i)) / [sum_j N_j exp(f_j - beta_U_j(i))]
        """
        K, M = hist_active.shape

        # log_w_ki = log(N_k) + f_k - beta_U_ki - log_denom_i
        # where log_denom_i = logsumexp_j[ log(N_j) + f_j - beta_U_ji ]
        log_N = np.log(N_k)
        log_numer = log_N[:, None] + f_k[:, None] - bias_active  # (K, M)
        log_denom = logsumexp(log_numer, axis=0)  # (M,)
        log_w = log_numer - log_denom[None, :]  # (K, M)

        # O_{kl} = sum_i exp(log_w_k(i) + log_w_l(i))
        # For efficiency, compute in a vectorized way for all pairs.
        # w has shape (K, M); O = w @ w.T but in log-space we need care.
        # Since w values are probabilities (bounded), we can work in linear
        # space here.
        w = np.exp(log_w)  # (K, M)
        overlap = w @ w.T  # (K, K)
        return overlap

    @staticmethod
    def _compute_n_eff(
        hist_active: np.ndarray, N_k: np.ndarray
    ) -> np.ndarray:
        """Effective sample size per active bin.

        N_eff(i) = (sum_k n_ki)^2 / sum_k (n_ki^2 / N_k)
        """
        C = hist_active.sum(axis=0)  # (M,)
        with np.errstate(divide="ignore", invalid="ignore"):
            denom = (hist_active ** 2 / N_k[:, None]).sum(axis=0)  # (M,)
            n_eff = np.where(denom > 0, C ** 2 / denom, 0.0)
        return n_eff

    def check_overlap(self, window_index: int = -1) -> dict:
        """Check overlap of one window against all others.

        Parameters
        ----------
        window_index : int
            Index of the window to check (default: last).

        Returns
        -------
        dict with keys:
            ``sufficient`` : bool
                Whether the maximum overlap exceeds the threshold.
            ``max_overlap`` : float
                Maximum pairwise histogram overlap with any other window.
            ``overlap_with`` : list of (int, float)
                (window_index, overlap_value) for all other windows.
            ``threshold`` : float
                The current overlap threshold.
        """
        if window_index < 0:
            window_index = self.n_windows + window_index
        if self.n_windows < 2:
            return {
                "sufficient": False,
                "max_overlap": 0.0,
                "overlap_with": [],
                "threshold": self._overlap_threshold,
            }

        # Compute histogram overlap on the fly (doesn't require a solve)
        K = self.n_windows
        M_full = int(np.prod(self._grid_shape))
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

        # settings
        data["tol"] = np.array(self._tol)
        data["max_iter"] = np.array(self._max_iter)
        data["lazy_solve"] = np.array(self._lazy_solve)
        data["overlap_threshold"] = np.array(self._overlap_threshold)

        # grid
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

        # windows
        K = self.n_windows
        data["n_windows"] = np.array(K)
        if K > 0:
            data["histograms"] = np.stack(
                [h.ravel() for h in self._histograms], axis=0
            )
            data["beta_biases"] = np.stack(
                [b.ravel() for b in self._beta_biases], axis=0
            )

        # converged state
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

            # grid
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

            # windows
            K = int(f["n_windows"])
            if K > 0:
                shape = solver._grid_shape
                hists = f["histograms"]   # (K, M_full)
                biases = f["beta_biases"]  # (K, M_full)
                for k in range(K):
                    solver._histograms.append(hists[k].reshape(shape))
                    solver._beta_biases.append(biases[k].reshape(shape))

            # converged state
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

    # ---- representation --------------------------------------------------

    def __repr__(self) -> str:
        status = "converged" if self._converged and not self._dirty else "dirty"
        return (
            f"WhamSolver(n_windows={self.n_windows}, "
            f"grid_shape={self._grid_shape}, status={status})"
        )
