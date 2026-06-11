"""Derived AGN quantities from a decomposition: single-epoch M_BH, L_bol, etc.

These turn the broad-Hbeta FWHM + continuum luminosity into the black-hole mass
and Eddington ratio used by reverberation-calibrated single-epoch estimators
(the quantities arXiv:2601.21974 tracks across epochs).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# Vestergaard & Peterson (2006) Hbeta single-epoch zero-point (FWHM-based;
# matches Shen DR7 logBHHM). A sigma (line-dispersion) based mass is *not*
# computed here: its zero-point needs object-by-object validation against the
# catalog (Shen's FWHM- and sigma-based masses agree to ~0.2 dex, and a naive
# VP06 sigma zero-point on the PyQSOFit second-moment width is mis-calibrated).
# The measured broad-line sigma is still exposed for downstream use. (future work)
VP06_HBETA_ZP = 6.91
# Bolometric correction at 5100A (Richards et al. 2006 / Shen et al. 2011)
BC_5100 = 9.26
# Eddington luminosity per solar mass [erg/s]
L_EDD_PER_MSUN = 1.26e38


@dataclass
class DerivedQuantities:
    fwhm_hbeta_kms: float
    log_L5100: float          # log10(L5100 / erg s^-1)
    log_MBH: float            # log10(M_BH / Msun)
    log_Lbol: float           # log10(L_bol / erg s^-1)
    eddington_ratio: float    # L_bol / L_Edd
    log_MBH_err: float = float("nan")   # 1-sigma (NaN unless MC errors available)
    sigma_hbeta_kms: float = float("nan")   # measured broad-line dispersion

    def _pm(self):
        return f" +/- {self.log_MBH_err:.2f}" if math.isfinite(self.log_MBH_err) else ""

    def __str__(self):
        return (f"FWHM(Hb)={self.fwhm_hbeta_kms:.0f} km/s  "
                f"logL5100={self.log_L5100:.2f}  logM_BH={self.log_MBH:.2f}{self._pm()}  "
                f"logL_bol={self.log_Lbol:.2f}  lambda_Edd={self.eddington_ratio:.3f}")


def black_hole_mass_hbeta(fwhm_kms: float, log_L5100: float) -> float:
    """log10(M_BH/Msun) from broad Hbeta FWHM and log L5100 (VP06, FWHM-based)."""
    return (VP06_HBETA_ZP + 2.0 * math.log10(fwhm_kms / 1000.0)
            + 0.5 * (log_L5100 - 44.0))


def bolometric_luminosity(log_L5100: float) -> float:
    """log10(L_bol/erg s^-1) from log L5100 via the 5100A bolometric correction."""
    return log_L5100 + math.log10(BC_5100)


def eddington_ratio(log_Lbol: float, log_MBH: float) -> float:
    log_L_edd = math.log10(L_EDD_PER_MSUN) + log_MBH
    return 10.0 ** (log_Lbol - log_L_edd)


def derive(result, complex_name: str = "Hb") -> DerivedQuantities | None:
    """Compute M_BH / L_bol / Eddington ratio from a DecompositionResult.

    Returns None if the broad Hbeta FWHM or L5100 is unavailable / non-physical.
    """
    m = result.lines.get(complex_name)
    log_L5100 = result.continuum.get("LogL5100", result.continuum.get("L5100"))
    if m is None or log_L5100 is None:
        return None
    fwhm = m.fwhm_kms
    if not (fwhm and fwhm > 0 and math.isfinite(fwhm) and math.isfinite(log_L5100)):
        return None
    log_mbh = black_hole_mass_hbeta(fwhm, log_L5100)
    log_lbol = bolometric_luminosity(log_L5100)

    # propagate MC uncertainties (FWHM and L5100) into log M_BH, if available
    log_mbh_err = float("nan")
    l5100_err = result.continuum.get("L5100_err")
    if m.has_errors and l5100_err is not None and math.isfinite(l5100_err):
        var = (2.0 / (math.log(10) * fwhm) * m.fwhm_err) ** 2 + (0.5 * l5100_err) ** 2
        log_mbh_err = math.sqrt(var)

    return DerivedQuantities(
        fwhm_hbeta_kms=fwhm, log_L5100=log_L5100, log_MBH=log_mbh,
        log_Lbol=log_lbol, eddington_ratio=eddington_ratio(log_lbol, log_mbh),
        log_MBH_err=log_mbh_err, sigma_hbeta_kms=m.sigma_kms)
