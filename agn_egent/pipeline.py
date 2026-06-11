"""Top-level decomposition entry point.

``decompose(spectrum, config)`` is the single surface the agentic QC layer
(Phase 3) drives: it picks a backend, runs the fit, and returns a
:class:`DecompositionResult`. Backends are registered by name so they stay
swappable.
"""
from __future__ import annotations

import os

from .spectrum import Spectrum
from .backends.base import DecompositionResult, FitBackend
from .backends.qsopar_config import QsoparConfig

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_DIR = os.path.join(PROJ_ROOT, "data", "runs")

_BACKENDS: dict[str, type] = {}


def register_backend(name: str, cls: type) -> None:
    _BACKENDS[name] = cls


def get_backend(name: str) -> FitBackend:
    if name not in _BACKENDS:
        raise KeyError(f"unknown backend {name!r}; available: {sorted(_BACKENDS)}")
    return _BACKENDS[name]()


def covered_config(config: QsoparConfig, spectrum: Spectrum,
                   margin_aa: float = 5.0) -> QsoparConfig:
    """Drop line components whose rest wavelength falls outside the spectrum.

    A blunt but effective ingest guard: at a given redshift some target lines
    (e.g. Halpha) may not be covered, and fitting them yields meaningless
    zero-amplitude "measurements". Trim them before fitting.
    """
    rw = spectrum.rest_wave
    lo, hi = rw.min() + margin_aa, rw.max() - margin_aa
    new = config.copy()
    new.lines = [l for l in new.lines if lo <= l.lam <= hi]
    return new


def decompose(spectrum: Spectrum,
              config: QsoparConfig | None = None,
              backend: str = "pyqsofit",
              workdir: str | None = None,
              make_figure: bool = True) -> DecompositionResult:
    """Run a full spectral decomposition.

    Parameters
    ----------
    spectrum : Spectrum         observed-frame spectrum to fit
    config   : QsoparConfig     model spec; defaults to SDSS optical Hb+Ha
    backend  : str              registered backend name (default 'pyqsofit')
    workdir  : str              scratch dir for qsopar.fits / figures
    make_figure : bool          render the diagnostic decomposition plot
    """
    if config is None:
        config = QsoparConfig.sdss_optical_default()
    config = covered_config(config, spectrum)  # ingest guard: only fit covered lines
    if workdir is None:
        workdir = os.path.join(RUNS_DIR, spectrum.name or "unnamed")
    os.makedirs(workdir, exist_ok=True)

    return get_backend(backend).decompose(
        spectrum, config, workdir=workdir, make_figure=make_figure)


# -- register built-in backends ------------------------------------------------
def _register_builtins():
    from .backends.pyqsofit_backend import PyQSOFitBackend
    register_backend("pyqsofit", PyQSOFitBackend)


_register_builtins()
