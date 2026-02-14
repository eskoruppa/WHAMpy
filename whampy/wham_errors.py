"""
Error estimation for WHAM free energies and probabilities.

Two methods are provided:

1. **Bootstrap** (``bootstrap_errors``): Multinomial resampling of histogram
   counts followed by repeated WHAM solves. Robust and assumption-free, but
   computationally expensive.

2. **Analytical / Fisher information** (``analytical_errors``): Computes the
   covariance of the free energy parameters from the Hessian of the WHAM
   negative log-likelihood at the converged solution, then propagates to
   per-bin uncertainties.  Fast (single matrix inversion of size K×K), but
   assumes the asymptotic (large-sample) regime.

Both functions accept a *solved* ``WhamSolver`` and return a ``WhamErrors``
dataclass.

References
----------
* Kumar et al., J. Comput. Chem. 13, 1011 (1992) — original WHAM.
* Zhu & Hummer, J. Comput. Chem. 33, 453 (2012) — convergence & error
  estimation.
* Shirts & Chodera, J. Chem. Phys. 129, 124105 (2008) — statistical overlap.
"""

from __future__ import annotations

import warnings
import numpy as np
from scipy.special import logsumexp
from dataclasses import dataclass
from typing import Optional, Union, TYPE_CHECKING
from pathlib import Path

if TYPE_CHECKING:
    from wham_solver import WhamSolver


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class WhamErrors:
    """Container for WHAM error estimates.

    All spatial arrays have shape = grid_shape of the solver.  Bins that had
    no data are filled with ``np.nan``.

    Attributes
    ----------
    method : str
        ``"bootstrap"`` or ``"analytical"``.
    var_log_prob : np.ndarray
        Variance of log p(x) (or log density) per bin.  Shape = grid_shape.
    std_log_prob : np.ndarray
        Standard deviation of log p(x) per bin.  Shape = grid_shape.
    var_free_energies : np.ndarray
        Variance of the per-window free energies, shape (K,).
    std_free_energies : np.ndarray
        Standard deviation of the per-window free energies, shape (K,).
    cov_free_energies : np.ndarray
        Full covariance matrix of the free energies, shape (K, K).

    Bootstrap-only fields (None for analytical):

    log_prob_ci_lo : np.ndarray or None
        Lower bound of the confidence interval on log_prob.
    log_prob_ci_hi : np.ndarray or None
        Upper bound of the confidence interval on log_prob.
    ci_level : float or None
        Confidence level (e.g. 0.95).
    n_bootstrap : int or None
        Number of bootstrap replicates used.
    bootstrap_log_probs : np.ndarray or None
        All bootstrap log_prob replicates, shape (n_bootstrap, *grid_shape).
        Only stored if ``store_replicates=True``.
    bootstrap_free_energies : np.ndarray or None
        All bootstrap free energies, shape (n_bootstrap, K).
        Only stored if ``store_replicates=True``.
    """

    method: str
    var_log_prob: np.ndarray
    std_log_prob: np.ndarray
    var_free_energies: np.ndarray
    std_free_energies: np.ndarray
    cov_free_energies: np.ndarray

    # bootstrap-only
    log_prob_ci_lo: Optional[np.ndarray] = None
    log_prob_ci_hi: Optional[np.ndarray] = None
    ci_level: Optional[float] = None
    n_bootstrap: Optional[int] = None
    bootstrap_log_probs: Optional[np.ndarray] = None
    bootstrap_free_energies: Optional[np.ndarray] = None

    # -- serialization -----------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        """Save to a single ``.npz`` file."""
        path = Path(path)
        data = {
            "method": np.array(self.method, dtype=str),
            "var_log_prob": self.var_log_prob,
            "std_log_prob": self.std_log_prob,
            "var_free_energies": self.var_free_energies,
            "std_free_energies": self.std_free_energies,
            "cov_free_energies": self.cov_free_energies,
        }
        if self.log_prob_ci_lo is not None:
            data["log_prob_ci_lo"] = self.log_prob_ci_lo
            data["log_prob_ci_hi"] = self.log_prob_ci_hi
            data["ci_level"] = np.array(self.ci_level)
        if self.n_bootstrap is not None:
            data["n_bootstrap"] = np.array(self.n_bootstrap)
        if self.bootstrap_log_probs is not None:
            data["bootstrap_log_probs"] = self.bootstrap_log_probs
        if self.bootstrap_free_energies is not None:
            data["bootstrap_free_energies"] = self.bootstrap_free_energies
        np.savez_compressed(str(path), **data)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "WhamErrors":
        """Load from a ``.npz`` file."""
        path = Path(path)
        if not path.suffix:
            path = path.with_suffix(".npz")
        with np.load(str(path), allow_pickle=False) as f:
            return cls(
                method=str(f["method"]),
                var_log_prob=f["var_log_prob"],
                std_log_prob=f["std_log_prob"],
                var_free_energies=f["var_free_energies"],
                std_free_energies=f["std_free_energies"],
                cov_free_energies=f["cov_free_energies"],
                log_prob_ci_lo=f.get("log_prob_ci_lo"),
                log_prob_ci_hi=f.get("log_prob_ci_hi"),
                ci_level=float(f["ci_level"]) if "ci_level" in f else None,
                n_bootstrap=int(f["n_bootstrap"]) if "n_bootstrap" in f else None,
                bootstrap_log_probs=f.get("bootstrap_log_probs"),
                bootstrap_free_energies=f.get("bootstrap_free_energies"),
            )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_solver_data(solver: "WhamSolver"):
    """Extract the data arrays needed for error estimation from a solved solver.

    Returns
    -------
    hist_matrix : (K, M_full) float64
    bias_matrix : (K, M_full) float64
    grid_shape  : tuple
    active_mask : (M_full,) bool
    log_prob    : (M_full,) float64 (NaN for inactive)
    free_energies : (K,) float64
    bin_volumes : (M_full,) float64 or None
    """
    if solver._dirty or solver._log_prob is None:
        raise RuntimeError(
            "Solver has no converged result. Call solve() first."
        )
    K = solver.n_windows
    shape = solver.grid_shape
    M_full = int(np.prod(shape))

    hist_matrix = np.stack([h.ravel() for h in solver._histograms], axis=0)
    bias_matrix = np.stack([b.ravel() for b in solver._beta_biases], axis=0)

    combined = hist_matrix.sum(axis=0)
    active_mask = combined > 0

    log_prob = solver._log_prob.ravel().copy()
    free_energies = solver._free_energies.copy()

    bin_volumes = None
    if solver._bin_volumes is not None:
        bin_volumes = solver._bin_volumes.ravel().copy()

    return (hist_matrix, bias_matrix, shape, active_mask,
            log_prob, free_energies, bin_volumes)


def _compute_wham_weights(
    hist_active: np.ndarray,
    bias_active: np.ndarray,
    f_k: np.ndarray,
) -> np.ndarray:
    """Compute the WHAM weight matrix W_{ki} (K, M).

    W_{ki} = N_k * f_k * c_{ki} / D_i

    where D_i = sum_j N_j * f_j * c_{ji}, f_k = exp(g_k), c_{ki} = exp(-beta_U_{ki}).

    These weights satisfy sum_k W_{ki} = 1 for all active bins i.
    """
    K, M = hist_active.shape
    N_k = hist_active.sum(axis=1)  # (K,)
    log_N = np.log(N_k)

    # log_w_{ki} = log(N_k) + g_k - beta_U_{ki} - log(D_i)
    # where g_k = log(f_k) = f_k (since our f_k is already log(f_k) in the
    # solver convention... actually no: in the solver, f_k is the free energy
    # defined via f_k_new = -logsumexp(...), and exp(-f_k) = sum_i p_i c_{ki}.
    # So g_k = f_k in our convention.
    #
    # D_i = sum_j N_j exp(f_j - beta_U_{ji})

    log_numer = log_N[:, None] + f_k[:, None] - bias_active  # (K, M)
    log_denom = logsumexp(log_numer, axis=0)  # (M,)
    log_w = log_numer - log_denom[None, :]  # (K, M)
    W = np.exp(log_w)  # (K, M)
    return W


def _run_wham_from_arrays(
    hist_active: np.ndarray,
    bias_active: np.ndarray,
    tol: float = 1e-7,
    max_iter: int = 100_000,
    f_k_init: Optional[np.ndarray] = None,
) -> tuple:
    """Run WHAM self-consistency on pre-reduced arrays.

    Parameters
    ----------
    hist_active : (K, M) histogram counts for active bins.
    bias_active : (K, M) beta*U bias values for active bins.
    tol, max_iter : convergence parameters.
    f_k_init : optional warm-start free energies.

    Returns
    -------
    f_k : (K,) converged free energies.
    log_p : (M,) converged log probabilities (unnormalized).
    converged : bool
    """
    K, M = hist_active.shape

    with np.errstate(divide="ignore"):
        log_n = np.log(hist_active)

    N_k = hist_active.sum(axis=1)
    log_N = np.log(N_k)
    log_C = logsumexp(log_n, axis=0)

    f_k = np.zeros(K, dtype=np.float64) if f_k_init is None else f_k_init.copy()
    f_k -= f_k[0]

    converged = False
    for _ in range(max_iter):
        log_denom = logsumexp(
            log_N[:, None] + f_k[:, None] - bias_active, axis=0
        )
        log_p = log_C - log_denom

        f_k_new = -logsumexp(log_p[None, :] - bias_active, axis=1)
        f_k_new -= f_k_new[0]

        delta = np.max(np.abs(f_k_new - f_k))
        f_k = f_k_new

        if delta < tol:
            converged = True
            break

    # recompute log_p with final f_k
    log_denom = logsumexp(
        log_N[:, None] + f_k[:, None] - bias_active, axis=0
    )
    log_p = log_C - log_denom

    return f_k, log_p, converged


# ---------------------------------------------------------------------------
# Bootstrap error estimation
# ---------------------------------------------------------------------------

def bootstrap_errors(
    solver: "WhamSolver",
    n_bootstrap: int = 200,
    ci_level: float = 0.95,
    seed: Optional[int] = None,
    store_replicates: bool = False,
    tol: Optional[float] = None,
    max_iter: Optional[int] = None,
    verbose: bool = False,
) -> WhamErrors:
    """Estimate WHAM errors by multinomial bootstrap resampling.

    For each bootstrap replicate, the histogram of each window is resampled
    by drawing N_k counts from a multinomial distribution with probabilities
    proportional to the original histogram.  WHAM is then re-solved (warm-
    started from the original free energies), and statistics are collected
    over all replicates.

    Parameters
    ----------
    solver : WhamSolver
        A solved solver (``solve()`` must have been called).
    n_bootstrap : int
        Number of bootstrap replicates.
    ci_level : float
        Confidence level for the confidence intervals (default 0.95).
    seed : int, optional
        Random seed for reproducibility.
    store_replicates : bool
        If True, store all replicate log_prob and free_energy arrays in the
        result (can be large).
    tol : float, optional
        Convergence tolerance for bootstrap WHAM solves.
        Defaults to the solver's tolerance.
    max_iter : int, optional
        Max iterations for bootstrap WHAM solves.
        Defaults to the solver's max_iter.
    verbose : bool
        If True, print progress.

    Returns
    -------
    WhamErrors
    """
    (hist_matrix, bias_matrix, shape, active_mask,
     log_prob_full, f_k_orig, bin_volumes) = _extract_solver_data(solver)

    if tol is None:
        tol = solver._tol
    if max_iter is None:
        max_iter = solver._max_iter

    K = hist_matrix.shape[0]
    M_full = hist_matrix.shape[1]
    M_active = int(active_mask.sum())

    hist_active_orig = hist_matrix[:, active_mask]
    bias_active_orig = bias_matrix[:, active_mask]

    # Per-window total counts and multinomial probabilities
    N_k = hist_active_orig.sum(axis=1).astype(int)  # (K,)
    with np.errstate(divide="ignore", invalid="ignore"):
        probs = hist_active_orig / N_k[:, None]  # (K, M_active)

    rng = np.random.default_rng(seed)

    # Storage for replicates
    all_log_p = np.empty((n_bootstrap, M_full), dtype=np.float64)
    all_f_k = np.empty((n_bootstrap, K), dtype=np.float64)
    n_failed = 0

    for b in range(n_bootstrap):
        if verbose and (b + 1) % 50 == 0:
            print(f"  Bootstrap replicate {b+1}/{n_bootstrap}")

        # Resample each window independently
        hist_boot = np.empty_like(hist_active_orig)
        for k in range(K):
            p_k = probs[k]
            # Handle windows that may have bins with zero probability
            p_k = np.maximum(p_k, 0.0)
            p_sum = p_k.sum()
            if p_sum > 0:
                p_k = p_k / p_sum
            else:
                # Window has no counts in active bins (shouldn't happen)
                p_k = np.ones(M_active) / M_active
            hist_boot[k] = rng.multinomial(N_k[k], p_k).astype(np.float64)

        # Recompute active mask for this replicate
        combined_boot = hist_boot.sum(axis=0)
        boot_active = combined_boot > 0

        if boot_active.sum() == 0:
            n_failed += 1
            all_log_p[b] = np.nan
            all_f_k[b] = np.nan
            continue

        hist_b = hist_boot[:, boot_active]
        bias_b = bias_active_orig[:, boot_active]

        # Warm-start from original free energies
        f_k_b, log_p_b, conv = _run_wham_from_arrays(
            hist_b, bias_b,
            tol=tol, max_iter=max_iter,
            f_k_init=f_k_orig,
        )

        if not conv:
            n_failed += 1

        # Apply bin volume correction (same as in solver.solve)
        log_prob_b_full = np.full(M_full, np.nan, dtype=np.float64)
        active_indices = np.where(active_mask)[0][boot_active]
        log_prob_b_full[active_indices] = log_p_b

        if bin_volumes is not None:
            log_vol = np.log(bin_volumes)
            valid = np.isfinite(log_prob_b_full)
            log_prob_b_full[valid] -= log_vol[valid]

        all_log_p[b] = log_prob_b_full
        all_f_k[b] = f_k_b

    if n_failed > 0:
        warnings.warn(
            f"{n_failed}/{n_bootstrap} bootstrap replicates failed to converge.",
            stacklevel=2,
        )

    # Compute statistics over replicates
    # Shift all log_prob to have the same reference (mean over shared active bins)
    # to remove the arbitrary constant before computing variance.
    ref_valid = np.all(np.isfinite(all_log_p), axis=0)  # bins valid in all replicates
    if ref_valid.sum() > 0:
        # Shift each replicate so its mean over shared bins is zero
        means = np.nanmean(all_log_p[:, ref_valid], axis=1, keepdims=True)
        all_log_p[:, ref_valid] -= means
        # Also shift the original
        orig_mean = np.nanmean(log_prob_full[ref_valid])
        # The variance is shift-invariant, so this is just for CI computation

    with np.errstate(all="ignore"):
        var_log_p = np.nanvar(all_log_p, axis=0, ddof=1)
        std_log_p = np.sqrt(var_log_p)

    var_log_p_full = var_log_p.reshape(shape)
    std_log_p_full = std_log_p.reshape(shape)

    # Confidence intervals
    alpha = 1 - ci_level
    ci_lo_full = np.full(M_full, np.nan)
    ci_hi_full = np.full(M_full, np.nan)

    for i in range(M_full):
        vals = all_log_p[:, i]
        finite = np.isfinite(vals)
        if finite.sum() > 2:
            ci_lo_full[i] = np.nanpercentile(vals[finite], 100 * alpha / 2)
            ci_hi_full[i] = np.nanpercentile(vals[finite], 100 * (1 - alpha / 2))

    ci_lo_full = ci_lo_full.reshape(shape)
    ci_hi_full = ci_hi_full.reshape(shape)

    # Free energy statistics
    # Shift free energies so f_k[0] = 0 (already the case by convention)
    var_f = np.var(all_f_k, axis=0, ddof=1)
    std_f = np.sqrt(var_f)
    cov_f = np.cov(all_f_k, rowvar=False, ddof=1)
    if cov_f.ndim == 0:
        cov_f = cov_f.reshape(1, 1)

    # Build result
    result = WhamErrors(
        method="bootstrap",
        var_log_prob=var_log_p_full,
        std_log_prob=std_log_p_full,
        var_free_energies=var_f,
        std_free_energies=std_f,
        cov_free_energies=cov_f,
        log_prob_ci_lo=ci_lo_full,
        log_prob_ci_hi=ci_hi_full,
        ci_level=ci_level,
        n_bootstrap=n_bootstrap,
    )

    if store_replicates:
        result.bootstrap_log_probs = all_log_p.reshape(n_bootstrap, *shape)
        result.bootstrap_free_energies = all_f_k

    return result


# ---------------------------------------------------------------------------
# Analytical (Fisher information) error estimation
# ---------------------------------------------------------------------------

def analytical_errors(solver: "WhamSolver") -> WhamErrors:
    """Estimate WHAM errors from the Fisher information matrix.

    Computes the Hessian of the WHAM negative log-likelihood at the converged
    solution, inverts it to obtain the covariance matrix of the free energy
    parameters, and propagates the uncertainty to per-bin log-probability
    estimates.

    The variance of ``ln p_l`` has two contributions:

    1. **Counting noise**: ``1 / M_l`` where ``M_l`` is the total count in
       bin *l* across all windows (Poisson approximation).
    2. **Free energy uncertainty**: propagated through the WHAM weights from
       the covariance of the free energy parameters.

    The Hessian of the negative log-likelihood with respect to the free
    energy parameters ``g_k = ln f_k`` is:

    .. math::

        H_{ij} = \\delta_{ij} \\sum_l M_l W_{il}
                 - \\sum_l M_l W_{il} W_{jl}

    where ``W_{il} = N_i f_i c_{il} / D_l`` are the WHAM weights, and
    ``D_l = \\sum_k N_k f_k c_{kl}`` is the WHAM denominator.

    Since ``g_1 = 0`` is fixed (gauge condition), the Hessian is reduced to
    the ``(K-1) × (K-1)`` submatrix for ``g_2, ..., g_K``.

    Parameters
    ----------
    solver : WhamSolver
        A solved solver.

    Returns
    -------
    WhamErrors

    Notes
    -----
    This method assumes the large-sample (asymptotic) regime where the
    maximum likelihood estimator is approximately Gaussian.  For windows
    with very few samples or poor overlap, bootstrap is more reliable.

    References
    ----------
    Zhu & Hummer, J. Comput. Chem. 33, 453–465 (2012), Eq. (15) and
    Appendix A.  Kumar et al., J. Comput. Chem. 13, 1011 (1992).
    """
    (hist_matrix, bias_matrix, shape, active_mask,
     log_prob_full, f_k, bin_volumes) = _extract_solver_data(solver)

    K = hist_matrix.shape[0]
    M_full = hist_matrix.shape[1]

    hist_active = hist_matrix[:, active_mask]   # (K, M)
    bias_active = bias_matrix[:, active_mask]   # (K, M)
    M = hist_active.shape[1]

    # Total counts per bin
    M_l = hist_active.sum(axis=0)  # (M,)

    # Compute WHAM weight matrix W_{ki}: shape (K, M)
    W = _compute_wham_weights(hist_active, bias_active, f_k)

    # ---------------------------------------------------------------
    # Hessian of the negative log-likelihood w.r.t. g_k
    #
    # H_{ij} = delta_{ij} * sum_l M_l W_{il} - sum_l M_l W_{il} W_{jl}
    #
    # In matrix form: H = diag(W @ M_l) - (W * sqrt(M_l)) @ (W * sqrt(M_l))^T
    # ---------------------------------------------------------------

    # W_scaled = W * sqrt(M_l)  — shape (K, M)
    sqrt_M = np.sqrt(M_l)
    W_scaled = W * sqrt_M[None, :]  # (K, M)

    # H = diag(row sums of W * M_l) - W_scaled @ W_scaled^T
    WM = (W * M_l[None, :]).sum(axis=1)  # (K,) — diagonal entries
    H_full = np.diag(WM) - W_scaled @ W_scaled.T  # (K, K)

    # Apply gauge condition: g_1 = 0 fixed, so remove row/col 0
    # Work with g_2, ..., g_K
    if K == 1:
        # Single window: no free energy uncertainty, only counting noise
        cov_g_full = np.zeros((1, 1), dtype=np.float64)
    else:
        H_reduced = H_full[1:, 1:]  # (K-1, K-1)

        # Regularization: if H_reduced is singular (e.g. disconnected windows),
        # add a small ridge
        try:
            cov_g_reduced = np.linalg.inv(H_reduced)  # (K-1, K-1)
        except np.linalg.LinAlgError:
            warnings.warn(
                "Hessian is singular; adding Tikhonov regularization. "
                "Error estimates may be unreliable.",
                stacklevel=2,
            )
            ridge = 1e-10 * np.eye(K - 1)
            cov_g_reduced = np.linalg.inv(H_reduced + ridge)

        # Embed back into full (K, K) matrix with zeros for row/col 0
        cov_g_full = np.zeros((K, K), dtype=np.float64)
        cov_g_full[1:, 1:] = cov_g_reduced

    # ---------------------------------------------------------------
    # Propagate to variance of ln p_l
    #
    # ln p_l = ln M_l - ln D_l     (up to volume correction)
    #
    # d(ln p_l)/d(g_k) = -d(ln D_l)/d(g_k) = -W_{kl}
    #
    # Var(ln p_l) = 1/M_l + sum_{jk} W_{jl} W_{kl} Cov(g_j, g_k)
    # ---------------------------------------------------------------

    # Counting noise contribution
    with np.errstate(divide="ignore"):
        var_count = 1.0 / M_l  # (M,)

    # Free energy contribution: for each bin l, compute w_l^T @ Cov @ w_l
    # where w_l = W[:, l] (the column of weights for bin l).
    # Vectorized: var_fe = diag(W^T @ Cov @ W)
    #           = sum over entries of (Cov @ W) * W
    cov_W = cov_g_full @ W  # (K, M)
    var_fe = (cov_W * W).sum(axis=0)  # (M,)

    var_ln_p_active = var_count + var_fe  # (M,)

    # Expand back to full grid
    var_ln_p_full = np.full(M_full, np.nan, dtype=np.float64)
    var_ln_p_full[active_mask] = var_ln_p_active
    var_ln_p_full = var_ln_p_full.reshape(shape)

    std_ln_p_full = np.full(shape, np.nan, dtype=np.float64)
    valid = np.isfinite(var_ln_p_full)
    std_ln_p_full[valid] = np.sqrt(var_ln_p_full[valid])

    # Free energy variances and covariance
    var_f = np.diag(cov_g_full)
    std_f = np.sqrt(np.maximum(var_f, 0.0))

    return WhamErrors(
        method="analytical",
        var_log_prob=var_ln_p_full,
        std_log_prob=std_ln_p_full,
        var_free_energies=var_f,
        std_free_energies=std_f,
        cov_free_energies=cov_g_full,
    )
