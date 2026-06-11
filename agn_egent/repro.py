"""Reproducibility: pin BLAS/threadpool threads to 1.

PyQSOFit's MLE fit is numerically sensitive for weak, under-constrained
components. Multithreaded BLAS makes the floating-point reduction order
non-deterministic, which tips degenerate fits into different local minima
across processes (observed: broad Hbeta FWHM swinging 5000-12000 km/s on a
SNR~5 line). Pinning threads to 1 makes every fit byte-reproducible.

IMPORTANT: the thread-count environment variables only take effect if they are
set *before* numpy (hence its bundled BLAS) is first imported. A runtime
``threadpoolctl.threadpool_limits`` is NOT sufficient on macOS/Accelerate once
BLAS is initialized. Therefore ``import agn_egent`` (which calls
:func:`pin_threads`) must happen before ``import numpy`` for hard cross-process
reproducibility. Batch parallelism should be at the object level (one
single-threaded process per spectrum), which is also the most efficient layout.
"""
from __future__ import annotations

import os

_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def pin_threads(n: int = 1) -> bool:
    """Set thread-count env vars to ``n`` (without clobbering user overrides).

    Returns True if applied before numpy was imported (i.e. effective), False
    otherwise (caller may then rely on env vars set in the launcher).
    """
    import sys
    for var in _THREAD_VARS:
        os.environ.setdefault(var, str(n))
    return "numpy" not in sys.modules
