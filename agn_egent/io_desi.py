"""DESI spectra via SPARCL: the second survey the engine can eat.

DESI DR1 is public but its coadd files are organized by healpix, with all
target classes mixed together -- bulk-downloading them to fish out quasars
would mean moving terabytes. NOIRLab's SPARCL service instead serves individual
spectra by id, so a campaign can pull exactly the ~270k z<0.95 QSOs it can fit
and nothing else, and workers can fetch their own spectra just as they do for
SDSS.

Requires ``pip install sparclclient``. Beware: as of writing, sparclclient's
dependency pins can DOWNGRADE numpy below 2.0 -- install it, then check numpy.

DESI wavelength coverage is 3600-9824 A (like SDSS), so with the optical-only
model the usable QSOs are z <~ 0.95 (Hbeta + [O III]); the rest of the 1.6M
QSO catalog needs the future UV (Mg II / C IV) line model. Flux units are
1e-17 erg/s/cm^2/A, same convention as SDSS, so no rescaling.
"""
from __future__ import annotations

import numpy as np

from .spectrum import Spectrum

DESI_DR = "DESI-DR1"
# DESI resolution is ~2000-5500 across the arms; ~100 km/s FWHM is a
# reasonable single number for narrow-line deconvolution (SDSS uses 150).
DESI_INST_FWHM_KMS = 100.0


def _client():
    from sparcl.client import SparclClient
    return SparclClient()


def find_desi_qsos(z_min: float = 0.05, z_max: float = 0.95, n: int = 100,
                   data_release: str = DESI_DR, client=None) -> list:
    """SPARCL records (sparcl_id, targetid, ra, dec, redshift) for DESI QSOs.

    The default redshift window is what the optical Hbeta/[O III] model can
    actually fit. Returns plain dicts usable as campaign targets.
    """
    client = client or _client()
    found = client.find(
        outfields=["sparcl_id", "targetid", "ra", "dec", "redshift"],
        constraints={"data_release": [data_release], "spectype": ["QSO"],
                     "redshift": [z_min, z_max]},
        limit=n)
    out = []
    for r in found.records:
        out.append({"sparcl_id": r["sparcl_id"],
                    "name": f"desi-{r['targetid']}",
                    "ra": r.get("ra"), "dec": r.get("dec"),
                    "z": r.get("redshift")})
    return out


def spectrum_from_sparcl_record(rec, z: float | None = None) -> Spectrum:
    """Build a Spectrum from one retrieved SPARCL record (brz-coadded)."""
    wave = np.asarray(rec.wavelength, dtype=float)
    flux = np.asarray(rec.flux, dtype=float)
    ivar = np.asarray(rec.ivar, dtype=float)
    ok = np.isfinite(wave) & np.isfinite(flux) & (ivar > 0)
    if ok.sum() < 100:
        raise ValueError(f"SPARCL record {getattr(rec, 'targetid', '?')}: "
                         f"only {int(ok.sum())} usable pixels")
    return Spectrum(
        wave=wave[ok], flux=flux[ok], err=1.0 / np.sqrt(ivar[ok]),
        z=float(z if z is not None else rec.redshift),
        name=f"desi-{rec.targetid}",
        ra=float(rec.ra) if rec.ra is not None else None,
        dec=float(rec.dec) if rec.dec is not None else None)


def fetch_desi_spectrum(sparcl_id: str, z: float | None = None,
                        client=None) -> Spectrum:
    """Download one DESI spectrum by its SPARCL id."""
    client = client or _client()
    got = client.retrieve(
        uuid_list=[sparcl_id],
        include=["wavelength", "flux", "ivar", "redshift", "ra", "dec",
                 "targetid"])
    if not got.records:
        raise LookupError(f"no SPARCL record for {sparcl_id}")
    return spectrum_from_sparcl_record(got.records[0], z=z)
