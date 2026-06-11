"""Central configuration for AGN-Egent.

Stdlib-only on purpose: this module is safe to import *before* numpy so the
thread-pinning env vars take effect (PyQSOFit's MLE fit is reproducible only
with single-threaded BLAS; see agn_egent/repro.py).
"""
import os

# -- LLM (ClaudeInspector) -----------------------------------------------------
DEFAULT_MODEL = "claude-opus-4-8"     # vision model for the QC reviewer
DEFAULT_MAX_TOKENS = 2048

# -- pipeline defaults ---------------------------------------------------------
DEFAULT_MAX_ITERATIONS = 4            # agent QC loop cap
DEFAULT_NSAMP = 25                    # Monte-Carlo resamples for error bars

# -- paths ---------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RUNS_DIR = os.path.join(DATA_DIR, "runs")

# -- reproducibility -----------------------------------------------------------
THREAD_VARS = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
)


def pin_threads(n: int = 1) -> None:
    """Pin BLAS/threadpool threads to ``n``. Call before importing numpy."""
    for var in THREAD_VARS:
        os.environ.setdefault(var, str(n))
