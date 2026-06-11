"""Generic spectrum loaders for non-SDSS instruments.

Enough to feed heterogeneous data (e.g. HET/LRS2) through the same pipeline,
demonstrating the backend/ingest path is instrument-agnostic.
"""
from __future__ import annotations

import numpy as np
from astropy.io import fits

from .spectrum import Spectrum


def load_row_fits(path: str, z: float, ra: float | None = None,
                  dec: float | None = None, flux_scale: float = 1.0,
                  wave_row: int = 0, flux_row: int = 1, err_row: int = 2,
                  name: str | None = None) -> Spectrum:
    """Load a FITS whose primary HDU is a 2-D array of [wavelength, flux, err, ...].

    `flux_scale` rescales flux/err into the 1e-17 erg/s/cm^2/A convention the
    PyQSOFit backend expects (e.g. pass 1e17 for data stored in cgs).
    """
    with fits.open(path) as hdul:
        data = np.asarray(hdul[0].data, dtype=float)
    wave = data[wave_row]
    flux = data[flux_row] * flux_scale
    err = data[err_row] * flux_scale
    # guard against non-positive / non-finite errors
    bad = ~np.isfinite(err) | (err <= 0)
    if bad.any():
        good = err[~bad]
        err = err.copy()
        err[bad] = np.median(good) if good.size else 1.0
    order = np.argsort(wave)
    return Spectrum(wave=wave[order], flux=flux[order], err=err[order], z=z,
                    name=name or "spectrum", ra=ra, dec=dec,
                    meta={"source": "row_fits", "path": path})
