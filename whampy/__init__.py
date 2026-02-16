"""
WHAMpy — Weighted Histogram Analysis Method solver
===================================================

A general-purpose WHAM implementation supporting arbitrary dimensionality,
progressive window addition with warm-start convergence, overlap diagnostics,
analytical and bootstrap error estimation, and full serialization.
Designed for umbrella sampling simulations.

All bias potentials are expected in dimensionless units (βU, i.e., units of kT).
"""

__version__ = "0.2.0"
__author__ = "Enrico Skoruppa"

from .wham_solver import WhamSolver, WhamResult
from .wham_errors import bootstrap_errors, analytical_errors, WhamErrors

__all__ = [
    "WhamSolver",
    "WhamResult",
    "WhamErrors",
    "bootstrap_errors",
    "analytical_errors",
]