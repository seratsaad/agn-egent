"""Backend-agnostic result model + the protocol every fitter implements.

The agentic QC layer (Phase 3) consumes *only* these types, never a specific
fitting engine. That keeps PyQSOFit / pyspeckit / future backends swappable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass
class LineMeasurement:
    """Measured properties of one emission-line complex (broad component).

    The ``*_err`` fields are 1-sigma uncertainties from the backend's Monte-Carlo
    resampling; they are NaN for a plain MLE fit (no MC).
    """

    name: str                     # complex name, e.g. "Hb", "Ha"
    fwhm_kms: float = np.nan       # broad-component FWHM [km/s]
    sigma_kms: float = np.nan      # broad-component line dispersion [km/s]
    flux: float = np.nan           # integrated broad flux [1e-17 erg/s/cm^2]
    ew_aa: float = np.nan          # broad rest-frame equivalent width [A]
    peak_wave: float = np.nan      # peak rest wavelength [A]
    snr: float = np.nan            # broad-line S/N
    fwhm_err: float = np.nan       # 1-sigma errors (NaN unless MC was run)
    flux_err: float = np.nan
    ew_err: float = np.nan
    sigma_err: float = np.nan
    extra: dict = field(default_factory=dict)

    @property
    def has_errors(self) -> bool:
        return np.isfinite(self.fwhm_err)


@dataclass
class DecompositionResult:
    """Everything the pipeline produces for one spectrum.

    `components` holds named flux arrays all sampled on `rest_wave`:
        data, model, residual, conti, pl, feii_op, feii_uv, balmer, poly,
        host, qso, line_total, broad, narrow
    Missing components (e.g. host when decompose_host=False) are simply absent.
    """

    name: str
    z: float
    backend: str
    rest_wave: np.ndarray
    components: dict[str, np.ndarray]
    lines: dict[str, LineMeasurement]
    continuum: dict[str, float]            # L5100, PL_norm, PL_slope, FeII flux...
    quality: dict[str, float]              # per-complex reduced chi^2, etc.
    params: dict[str, float]               # raw best-fit parameter dict
    config: dict = field(default_factory=dict)   # serialized model config (provenance)
    figure_path: str | None = None
    log: list = field(default_factory=list)      # agent/decision provenance (Phase 3)
    # per narrow-line-component diagnostics, keyed by linename:
    #   {"OIII5007c": {"peak_wave":.., "amp":.., "snr":.., "compname":..,
    #                  "ngauss":.., "anchor": bool}}
    # lets the agent judge whether a narrow component is a real line or noise.
    narrow_lines: dict = field(default_factory=dict)
    # per *physical* line profile models, summed over that line's Gaussians and
    # sampled on `rest_wave`. Keyed by the physical line rather than the fitting
    # component, so a core+wing pair ("OIII5007c" + "OIII5007w") appears once as
    # "OIII5007" -- the profile you need for a line-shape measurement (W80,
    # asymmetry, double peaks). Broad complexes are keyed "Hb_br" / "Ha_br".
    # Arrays, so this is deliberately not serialized into provenance.
    line_models: dict = field(default_factory=dict)

    # -- convenience accessors -------------------------------------------------
    def component(self, key: str) -> np.ndarray:
        return self.components[key]

    @property
    def broad(self) -> np.ndarray | None:
        return self.components.get("broad")

    @property
    def narrow(self) -> np.ndarray | None:
        return self.components.get("narrow")

    def reduced_chi2(self) -> float:
        """Worst per-complex reduced chi^2 (a quick global quality scalar)."""
        vals = [v for v in self.quality.values() if np.isfinite(v)]
        return max(vals) if vals else np.nan

    def summary(self) -> str:
        lines = [f"{self.name}  z={self.z:.4f}  backend={self.backend}"]
        for k in sorted(("LogL5100", "L5100", "PL_slope")):
            if k in self.continuum:
                lines.append(f"  {k:10s} = {self.continuum[k]:.4g}")
        for nm, m in self.lines.items():
            lines.append(f"  {nm:4s} broad: FWHM={m.fwhm_kms:7.0f} km/s  "
                         f"flux={m.flux:8.1f}  SNR={m.snr:5.1f}")
        for nm, q in self.quality.items():
            lines.append(f"  chi2[{nm}] = {q:.3f}")
        return "\n".join(lines)


@runtime_checkable
class FitBackend(Protocol):
    """A spectral decomposition engine."""

    name: str

    def decompose(self, spectrum, config, workdir: str,
                  make_figure: bool = True) -> DecompositionResult:
        ...
