"""Build a synthetic AGN spectrum with known truth, for injection-recovery tests.

A power-law continuum + tied narrow lines ([OIII], [NII], [SII], narrow Balmer)
+ broad Hbeta/Halpha Gaussians, on an SDSS-like log-wavelength grid with
Gaussian noise. Returns a :class:`Spectrum` plus a truth dict so a recovered fit
can be checked against the injected values.

Deliberately omits the Fe II template and host galaxy: the recovery is run with
those switched off, so the test isolates the fitter's *line* accuracy against an
exactly known ground truth rather than against template-matching artifacts.
"""
from __future__ import annotations

import numpy as np

from .spectrum import Spectrum

C_KMS = 299792.458
FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def _gauss(wave, peak, center_obs, fwhm_kms):
    sigma = (fwhm_kms / C_KMS) * center_obs * FWHM_TO_SIGMA
    return peak * np.exp(-0.5 * ((wave - center_obs) / sigma) ** 2)


def _line_flux(peak, center_obs, fwhm_kms):
    """Integrated flux of a Gaussian with the given peak [1e-17 erg/s/cm^2]."""
    sigma = (fwhm_kms / C_KMS) * center_obs * FWHM_TO_SIGMA
    return peak * sigma * np.sqrt(2.0 * np.pi)


def make_synthetic_spectrum(
        z: float = 0.1,
        pl_norm: float = 6.0,           # f at 3000A rest [1e-17 units]
        pl_slope: float = -1.5,         # f_lambda ~ (lambda/3000)^pl_slope
        hbeta_fwhm: float = 4200.0,     # broad Hbeta FWHM [km/s]
        hbeta_peak: float = 9.0,        # broad Hbeta peak [1e-17 units]
        halpha_fwhm: float = 4600.0,    # broad Halpha FWHM [km/s]
        halpha_peak: float = 22.0,
        narrow_fwhm: float = 420.0,     # narrow-line FWHM [km/s]
        snr: float = 30.0,              # continuum S/N near 5100A rest
        seed: int = 0,
        name: str = "synthetic") -> tuple[Spectrum, dict]:
    rng = np.random.default_rng(seed)

    # SDSS-like log-wavelength grid (observed frame)
    loglam = np.arange(np.log10(3700.0), np.log10(9200.0), 1e-4)
    wave = 10 ** loglam
    rest = wave / (1 + z)

    # power-law continuum (f_lambda ~ (rest/3000)^slope), in observed frame flux
    flux = pl_norm * (rest / 3000.0) ** pl_slope

    def add(rest_wl, peak, fwhm):
        nonlocal flux
        flux = flux + _gauss(wave, peak, rest_wl * (1 + z), fwhm)

    # broad Balmer lines (the science target)
    add(4862.68, hbeta_peak, hbeta_fwhm)
    add(6564.61, halpha_peak, halpha_fwhm)
    # narrow Balmer
    add(4862.68, 0.20 * hbeta_peak, narrow_fwhm)
    add(6564.61, 0.30 * halpha_peak, narrow_fwhm)
    # [O III] 4959/5007 (1:3)
    add(5008.24, 6.0, narrow_fwhm)
    add(4960.30, 2.0, narrow_fwhm)
    # [N II] 6549/6585 (1:3) and [S II] 6718/6732
    add(6585.28, 3.0, narrow_fwhm)
    add(6549.85, 1.0, narrow_fwhm)
    add(6718.29, 1.2, narrow_fwhm)
    add(6732.67, 1.0, narrow_fwhm)

    # noise: set sigma so continuum S/N near rest 5100 matches `snr`
    cont_5100 = pl_norm * (5100.0 / 3000.0) ** pl_slope
    err = np.full_like(flux, cont_5100 / snr)
    flux_noisy = flux + rng.normal(0.0, 1.0, size=flux.shape) * err

    truth = {
        "z": z, "pl_norm": pl_norm, "pl_slope": pl_slope,
        "hbeta_fwhm": hbeta_fwhm, "halpha_fwhm": halpha_fwhm,
        "hbeta_flux": _line_flux(hbeta_peak, 4862.68 * (1 + z), hbeta_fwhm),
        "halpha_flux": _line_flux(halpha_peak, 6564.61 * (1 + z), halpha_fwhm),
    }
    spec = Spectrum(wave=wave, flux=flux_noisy, err=err, z=z,
                    name=name, ra=180.0, dec=0.0,
                    meta={"source": "synthetic", "truth": truth})
    return spec, truth
