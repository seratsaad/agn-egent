"""Data-driven decomposition model config.

This is the *edit surface* the agent (Phase 3) manipulates: which line complexes
to fit, how many Gaussians per line (``ngauss`` — the broad-component count),
the tying constraints (vindex/windex/findex), and the continuum options. It
serializes to the ``qsopar.fits`` parameter file PyQSOFit reads.

Keeping this as plain Python data (not a hardcoded FITS file) is what lets the
agent propose a model change as a structured diff rather than rewriting code.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict, replace

import numpy as np
from astropy.io import fits
from astropy.table import Table


@dataclass
class LinePrior:
    """One emission-line Gaussian-set prior (a row of qsopar's line table).

    Field meanings (PyQSOFit schema):
      lam      : vacuum rest wavelength [A]
      compname : complex name (lines sharing it are fit together), e.g. "Hb"
      minwav/maxwav : complex fitting window [A]
      linename : unique component name; suffix "_br"/"_na" marks broad/narrow
      ngauss   : number of Gaussians for this component  <-- key agent knob
      inisca/minsca/maxsca : line amplitude init/bounds
      inisig/minsig/maxsig : line sigma (ln-lambda) init/bounds
      voff     : allowed velocity offset (ln-lambda)
      vindex/windex/findex : nonzero shared index ties velocity/width/flux ratio
      fvalue   : relative flux scale within a findex group
      vary     : 1=free, 0=frozen
    """

    lam: float
    compname: str
    minwav: float
    maxwav: float
    linename: str
    ngauss: int
    inisca: float
    minsca: float
    maxsca: float
    inisig: float
    minsig: float
    maxsig: float
    voff: float
    vindex: int
    windex: int
    findex: int
    fvalue: float
    vary: int = 1

    @property
    def is_broad(self) -> bool:
        return "_br" in self.linename


@dataclass
class ContiPrior:
    parname: str
    initial: float
    vmin: float | None
    vmax: float | None
    vary: int = 1


# Default continuum priors (PL + Fe II UV/opt + Balmer + 2nd-order poly).
_DEFAULT_CONTI = [
    ContiPrior("Fe_uv_norm", 0.0, 0.0, 1e10),
    ContiPrior("Fe_uv_FWHM", 3000, 1200, 18000),
    ContiPrior("Fe_uv_shift", 0.0, -0.01, 0.01),
    ContiPrior("Fe_op_norm", 0.0, 0.0, 1e10),
    ContiPrior("Fe_op_FWHM", 3000, 1200, 18000),
    ContiPrior("Fe_op_shift", 0.0, -0.01, 0.01),
    ContiPrior("PL_norm", 1.0, 0.0, 1e10),
    # Power-law slope constrained to a physical AGN optical range so the
    # continuum cannot rail to an unphysical value or run positive. f_lambda
    # ~ (lambda/3000)^slope; real AGN sit around -1 to -2.5.
    ContiPrior("PL_slope", -1.5, -3.0, 0.0),
    ContiPrior("Blamer_norm", 0.0, 0.0, 1e10),
    ContiPrior("Balmer_Te", 15000, 10000, 50000),
    ContiPrior("Balmer_Tau", 0.5, 0.1, 2.0),
    ContiPrior("conti_a_0", 0.0, None, None),
    ContiPrior("conti_a_1", 0.0, None, None),
    ContiPrior("conti_a_2", 0.0, None, None),
]

_DEFAULT_CONTI_WINDOWS = [
    (3775., 3832.), (4000., 4050.), (4200., 4230.), (4435., 4640.),
    (5100., 5535.), (6005., 6035.), (6110., 6250.), (6800., 7000.),
]


def _hbeta_halpha_lines() -> list[LinePrior]:
    """Standard SDSS optical Hbeta + Halpha complexes (broad + tied narrow)."""
    return [
        # Halpha complex
        LinePrior(6564.61, "Ha", 6400, 6800, "Ha_br",   2, 0, 0, 1e10, 5e-3, 4e-3,  0.05,    0.015, 0, 0, 0, 0.05,  1),
        LinePrior(6564.61, "Ha", 6400, 6800, "Ha_na",   1, 0, 0, 1e10, 1e-3, 5e-4,  0.00169, 0.01,  1, 1, 0, 0.002, 1),
        LinePrior(6549.85, "Ha", 6400, 6800, "NII6549", 1, 0, 0, 1e10, 1e-3, 2.3e-4, 0.00169, 5e-3,  1, 1, 1, 0.001, 1),
        LinePrior(6585.28, "Ha", 6400, 6800, "NII6585", 1, 0, 0, 1e10, 1e-3, 2.3e-4, 0.00169, 5e-3,  1, 1, 1, 0.003, 1),
        LinePrior(6718.29, "Ha", 6400, 6800, "SII6718", 1, 0, 0, 1e10, 1e-3, 2.3e-4, 0.00169, 5e-3,  1, 1, 2, 0.001, 1),
        LinePrior(6732.67, "Ha", 6400, 6800, "SII6732", 1, 0, 0, 1e10, 1e-3, 2.3e-4, 0.00169, 5e-3,  1, 1, 2, 0.001, 1),
        # Hbeta complex
        LinePrior(4862.68, "Hb", 4640, 5100, "Hb_br",     2, 0, 0, 1e10, 5e-3, 4e-3,  0.05,    0.01,  0, 0, 0, 0.01,  1),
        LinePrior(4862.68, "Hb", 4640, 5100, "Hb_na",     1, 0, 0, 1e10, 1e-3, 2.3e-4, 0.00169, 0.01,  1, 1, 0, 0.002, 1),
        LinePrior(4960.30, "Hb", 4640, 5100, "OIII4959c", 1, 0, 0, 1e10, 1e-3, 2.3e-4, 0.00169, 0.01,  1, 1, 0, 0.002, 1),
        LinePrior(5008.24, "Hb", 4640, 5100, "OIII5007c", 1, 0, 0, 1e10, 1e-3, 2.3e-4, 0.00169, 0.01,  1, 1, 0, 0.004, 1),
        LinePrior(4960.30, "Hb", 4640, 5100, "OIII4959w", 1, 0, 0, 1e10, 3e-3, 2.3e-4, 0.004,   0.01,  2, 2, 0, 0.001, 1),
        LinePrior(5008.24, "Hb", 4640, 5100, "OIII5007w", 1, 0, 0, 1e10, 3e-3, 2.3e-4, 0.004,   0.01,  2, 2, 0, 0.002, 1),
    ]


@dataclass
class QsoparConfig:
    """A complete, serializable decomposition model specification."""

    lines: list[LinePrior]
    conti_priors: list[ContiPrior] = field(default_factory=lambda: list(_DEFAULT_CONTI))
    conti_windows: list[tuple] = field(default_factory=lambda: list(_DEFAULT_CONTI_WINDOWS))
    cont_loc: list[float] = field(default_factory=lambda: [1350, 1450, 3000, 4200, 5100])
    fe_flux_range: list[float] = field(default_factory=lambda: [4435, 4685])
    # PyQSOFit.Fit() options (model structure beyond the line list)
    fit_options: dict = field(default_factory=lambda: dict(
        nsmooth=1, deredden=True, reject_badpix=True,
        decompose_host=True, host_prior=False, decomp_na_mask=True,
        npca_gal=5, npca_qso=10, qso_type="CZBIN1",
        # poly=False: the AGN continuum is a pure power law (+ Fe II + host +
        # Balmer). The polynomial term lets the continuum bend away from a power
        # law and rail the slope -- the single biggest source of bad continua.
        Fe_uv_op=True, poly=False, rej_abs_conti=False, rej_abs_line=True,
        MC=False, linefit=True,
    ))

    # -- factory ---------------------------------------------------------------
    @classmethod
    def sdss_optical_default(cls) -> "QsoparConfig":
        """Hbeta+Halpha decomposition with host + Fe II + PL continuum."""
        return cls(lines=_hbeta_halpha_lines())

    # -- agent-facing edits ----------------------------------------------------
    def copy(self) -> "QsoparConfig":
        return QsoparConfig(
            lines=[replace(l) for l in self.lines],
            conti_priors=[replace(c) for c in self.conti_priors],
            conti_windows=list(self.conti_windows),
            cont_loc=list(self.cont_loc),
            fe_flux_range=list(self.fe_flux_range),
            fit_options=dict(self.fit_options),
        )

    def set_ngauss(self, linename: str, ngauss: int) -> "QsoparConfig":
        """Return a copy with the broad-Gaussian count of `linename` changed."""
        new = self.copy()
        for l in new.lines:
            if l.linename == linename:
                l.ngauss = int(ngauss)
        return new

    def remove_line(self, linename: str) -> "QsoparConfig":
        """Return a copy with the named line component removed entirely."""
        new = self.copy()
        new.lines = [l for l in new.lines if l.linename != linename]
        return new

    def drop_broad(self, compname: str) -> "QsoparConfig":
        """Return a copy with the broad component(s) of a complex removed."""
        new = self.copy()
        new.lines = [l for l in new.lines
                     if not (l.compname == compname and l.is_broad)]
        return new

    def set_fit_option(self, key: str, value) -> "QsoparConfig":
        """Return a copy with a single PyQSOFit.Fit() option changed."""
        new = self.copy()
        new.fit_options[key] = value
        return new

    def set_conti_param(self, parname: str, initial: float | None = None,
                        vmin: float | None = None, vmax: float | None = None,
                        vary: int | None = None) -> "QsoparConfig":
        """Return a copy with one continuum prior's init/bounds/vary changed.

        Only the arguments that are not ``None`` are updated; this is how the
        continuum planner pins the power-law slope (initial + narrowed bounds)
        without disturbing the rest of the continuum model.
        """
        new = self.copy()
        for c in new.conti_priors:
            if c.parname == parname:
                if initial is not None:
                    c.initial = float(initial)
                if vmin is not None:
                    c.vmin = float(vmin)
                if vmax is not None:
                    c.vmax = float(vmax)
                if vary is not None:
                    c.vary = int(vary)
        return new

    def set_conti_windows(self, windows: list[tuple]) -> "QsoparConfig":
        """Return a copy whose continuum-fitting windows are replaced.

        These are the (rest-frame) wavelength ranges PyQSOFit uses to anchor the
        continuum; restricting them to line-/Fe II-free regions is the core of
        the planner's job.
        """
        new = self.copy()
        new.conti_windows = [(float(lo), float(hi)) for lo, hi in windows]
        return new

    def to_dict(self) -> dict:
        return {
            "lines": [asdict(l) for l in self.lines],
            "conti_priors": [asdict(c) for c in self.conti_priors],
            "conti_windows": [list(w) for w in self.conti_windows],
            "cont_loc": list(self.cont_loc),
            "fe_flux_range": list(self.fe_flux_range),
            "fit_options": dict(self.fit_options),
        }

    # -- serialization to PyQSOFit's qsopar.fits -------------------------------
    def write_qsopar(self, workdir: str, filename: str = "qsopar.fits") -> str:
        os.makedirs(workdir, exist_ok=True)
        out = os.path.join(workdir, filename)

        line_rows = np.rec.array(
            [(l.lam, l.compname, l.minwav, l.maxwav, l.linename, l.ngauss,
              l.inisca, l.minsca, l.maxsca, l.inisig, l.minsig, l.maxsig,
              l.voff, l.vindex, l.windex, l.findex, l.fvalue, l.vary)
             for l in self.lines],
            formats=("float32, a20, float32, float32, a20, int32, float32, float32, "
                     "float32, float32, float32, float32, float32, int32, int32, "
                     "int32, float32, int32"),
            names=("lambda, compname, minwav, maxwav, linename, ngauss, inisca, "
                   "minsca, maxsca, inisig, minsig, maxsig, voff, vindex, windex, "
                   "findex, fvalue, vary"))
        hdu1 = fits.BinTableHDU(data=line_rows, name="line_priors")

        win_rows = np.rec.array(list(self.conti_windows),
                                formats="float32, float32", names="min, max")
        hdu2 = fits.BinTableHDU(data=win_rows, name="conti_windows")

        conti_rows = np.rec.array(
            [(c.parname, c.initial,
              np.nan if c.vmin is None else c.vmin,
              np.nan if c.vmax is None else c.vmax, c.vary)
             for c in self.conti_priors],
            formats="a20, float32, float32, float32, int32",
            names="parname, initial, min, max, vary")
        hdu3 = fits.BinTableHDU(data=conti_rows, name="conti_priors")

        measure = Table([[self.cont_loc], [[self.fe_flux_range]]],
                        names=("cont_loc", "Fe_flux_range"),
                        dtype=("float32", "float32"))
        hdu4 = fits.BinTableHDU(data=measure, name="measure_info")

        fits.HDUList([fits.PrimaryHDU(), hdu1, hdu2, hdu3, hdu4]).writeto(
            out, overwrite=True)
        return out
