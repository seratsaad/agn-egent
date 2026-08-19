"""Literature cross-match: benchmark our measurements against Shen et al. (2011).

The Shen DR7 quasar property catalog (VizieR J/ApJS/194/45) publishes broad-Hbeta
FWHM, L5100, and single-epoch black-hole masses for ~105k quasars. We can fetch a
quasar's SDSS spectrum, run it through the pipeline, and compare our recovered
FWHM / logL5100 / logM_BH to the catalog — the honest "how accurate are we?" test.

Requires network (astroquery VizieR + SDSS). Functions raise on failure so a
caller can skip gracefully when offline.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .spectrum import Spectrum
from .io_sdss import spectrum_from_hdulist

SHEN_CATALOG = "J/ApJS/194/45"

# Shen column -> our key. logBHHM is the Hbeta FWHM-based VP06 mass (our estimator).
_SHEN_COLS = ["Plate", "MJD", "Fiber", "z", "W(BHb)", "e_W(BHb)",
              "logL5100", "e_logL5100", "logBHHM", "e_logBHHM",
              "logBH", "e_logBH", "SN(Hb)"]


@dataclass
class ShenRecord:
    plate: int
    mjd: int
    fiber: int
    z: float
    fwhm_hbeta: float          # W(BHb) [km/s]
    fwhm_hbeta_err: float
    log_L5100: float
    log_L5100_err: float
    log_MBH_hbeta: float       # logBHHM (VP06 Hbeta)
    log_MBH_hbeta_err: float
    sn_hbeta: float


def query_shen(plate: int, mjd: int, fiber: int) -> ShenRecord:
    """Look up one quasar's published Shen-2011 properties by plate/MJD/fiber."""
    from astroquery.vizier import Vizier
    V = Vizier(columns=_SHEN_COLS,
               column_filters={"Plate": f"={plate}", "Fiber": f"={fiber}"})
    V.ROW_LIMIT = 50
    res = V.get_catalogs(SHEN_CATALOG)
    if not res:
        raise LookupError(f"no Shen catalog rows for plate={plate} fiber={fiber}")
    t = res[0]
    rows = [r for r in t if int(r["MJD"]) == mjd]
    if not rows:
        rows = list(t)
    r = rows[0]
    return ShenRecord(
        plate=int(r["Plate"]), mjd=int(r["MJD"]), fiber=int(r["Fiber"]),
        z=float(r["z"]), fwhm_hbeta=float(r["W(BHb)"]),
        fwhm_hbeta_err=float(r["e_W(BHb)"]),
        log_L5100=float(r["logL5100"]), log_L5100_err=float(r["e_logL5100"]),
        log_MBH_hbeta=float(r["logBHHM"]), log_MBH_hbeta_err=float(r["e_logBHHM"]),
        sn_hbeta=float(r["SN(Hb)"]))


def find_shen_quasars(z_min=0.28, z_max=0.45, sn_min=30.0, fwhm_min=2000.0,
                      n=5, sn_max=None) -> list[tuple[int, int, int]]:
    """Return (plate, mjd, fiber) for Shen quasars (Hbeta well covered).

    ``sn_max`` bounds the broad-Hbeta S/N from above, which is how we build a
    deliberately *low-S/N* (hard) sample: e.g. ``sn_min=3, sn_max=8`` selects
    poor-quality spectra where automated quality control matters most.
    """
    from astroquery.vizier import Vizier
    sn_filter = f">{sn_min}" + (f" && <{sn_max}" if sn_max is not None else "")
    V = Vizier(columns=["Plate", "MJD", "Fiber", "z", "SN(Hb)", "W(BHb)"],
               column_filters={"z": f">{z_min} && <{z_max}",
                               "SN(Hb)": sn_filter, "W(BHb)": f">{fwhm_min}"})
    V.ROW_LIMIT = n
    res = V.get_catalogs(SHEN_CATALOG)
    if not res:
        return []
    return [(int(r["Plate"]), int(r["MJD"]), int(r["Fiber"])) for r in res[0]]


def fetch_sdss_spectrum(plate: int, mjd: int, fiber: int,
                        name: str | None = None) -> Spectrum:
    """Download a quasar's SDSS spectrum and build a :class:`Spectrum`."""
    from astroquery.sdss import SDSS
    sp = SDSS.get_spectra(plate=plate, mjd=mjd, fiberID=fiber)
    if not sp:
        raise LookupError(f"no SDSS spectrum for {plate}-{mjd}-{fiber}")
    nm = name or f"{plate:04d}-{mjd}-{fiber:04d}"
    return spectrum_from_hdulist(sp[0], name=nm)


@dataclass
class Comparison:
    name: str
    shen: ShenRecord
    our_fwhm: float
    our_log_L5100: float
    our_log_MBH: float
    our_log_MBH_err: float = float("nan")
    notes: str = ""

    @property
    def dlog_MBH(self) -> float:
        return self.our_log_MBH - self.shen.log_MBH_hbeta

    @property
    def dlog_L5100(self) -> float:
        return self.our_log_L5100 - self.shen.log_L5100

    @property
    def fwhm_ratio(self) -> float:
        return self.our_fwhm / self.shen.fwhm_hbeta

    def row(self) -> str:
        return (f"{self.name:16s} z={self.shen.z:.3f}  "
                f"FWHM ours/Shen={self.our_fwhm:6.0f}/{self.shen.fwhm_hbeta:6.0f} "
                f"(x{self.fwhm_ratio:.2f})  "
                f"logL5100 d={self.dlog_L5100:+.2f}  "
                f"logM_BH ours/Shen={self.our_log_MBH:.2f}/{self.shen.log_MBH_hbeta:.2f} "
                f"(d={self.dlog_MBH:+.2f})")
