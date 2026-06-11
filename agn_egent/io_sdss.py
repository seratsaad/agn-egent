"""Loading SDSS spectra into the :class:`Spectrum` model.

Phase 1: local single-fiber SDSS spec FITS files (spec-PLATE-MJD-FIBER.fits).
A `fetch_sdss` by name/coords via astroquery is a Phase 4 addition.
"""
from __future__ import annotations

import os

import numpy as np
from astropy.io import fits

from .spectrum import Spectrum


def spectrum_from_hdulist(hdul, name: str | None = None,
                          path: str | None = None) -> Spectrum:
    """Build a :class:`Spectrum` from an open SDSS spec HDUList.

    HDU1 holds the coadded spectrum (loglam, flux, ivar); HDU2 holds the
    pipeline redshift; HDU0 header holds coordinates / plate-mjd-fiber.
    """
    d1 = hdul[1].data
    lam = 10 ** d1["loglam"]
    flux = d1["flux"].astype(float)
    ivar = d1["ivar"].astype(float)
    with np.errstate(divide="ignore"):
        err = np.where(ivar > 0, 1.0 / np.sqrt(ivar), np.inf)

    z = float(hdul[2].data["z"][0])
    h0 = hdul[0].header
    ra = h0.get("plug_ra")
    dec = h0.get("plug_dec")
    plate = h0.get("plateid")
    mjd = h0.get("mjd")
    fiber = h0.get("fiberid")

    if name is None:
        if plate is not None and mjd is not None and fiber is not None:
            name = f"{int(plate):04d}-{int(mjd)}-{int(fiber):04d}"
        elif path is not None:
            name = os.path.splitext(os.path.basename(path))[0]
        else:
            name = "sdss"

    return Spectrum(
        wave=lam, flux=flux, err=err, z=z, name=name, ra=ra, dec=dec,
        meta={"source": "sdss", "path": path, "plateid": plate,
              "mjd": mjd, "fiberid": fiber},
    )


def load_sdss(path: str, name: str | None = None) -> Spectrum:
    """Read a standard SDSS single-spectrum FITS file from disk."""
    with fits.open(path) as hdul:
        return spectrum_from_hdulist(hdul, name=name, path=path)
