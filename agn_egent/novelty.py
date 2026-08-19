"""Is this candidate already known? Literature context for a shortlist.

A discovery campaign produces a ranked list of unusual objects. Most of them
will already be in the literature -- known double-peaked emitters, known NLS1s,
known changing-look AGN -- and that is useful either way: a known object
validates the selection, an unknown one is a candidate worth following up.

This module attaches that context by cone-searching SIMBAD, which is free and
needs no account. It is deliberately applied only to the shortlist (tens to
hundreds of objects), never to a whole survey, both to be polite to the service
and because the answer only matters once an object is interesting.

Everything degrades gracefully: no network, no SIMBAD entry, or a malformed
response all return a record with ``matched=False`` rather than raising, so a
vetting run never dies halfway through a candidate list.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, asdict

# SIMBAD object types that mean "this is a known active galaxy". A match on one
# of these says the source is a known AGN, NOT that its unusual property is
# known -- almost every SDSS quasar is a known QSO. Novelty is about the
# *property*, so this is context for a human, not an automatic reject.
AGN_TYPES = {"QSO", "AGN", "Sy1", "Sy2", "SyG", "Seyfert", "BLLac", "Blazar",
             "LINER", "QSO_Candidate", "AGN_Candidate"}


@dataclass
class NoveltyRecord:
    """What the literature already says about one candidate position."""
    name: str = ""
    ra: float | None = None
    dec: float | None = None
    matched: bool = False
    main_id: str = ""
    otype: str = ""
    separation_arcsec: float | None = None
    n_simbad_objects: int = 0
    known_agn: bool = False
    error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        if self.error:
            return f"{self.name}: lookup failed ({self.error})"
        if not self.matched:
            return f"{self.name}: no SIMBAD source within the search radius"
        sep = (f"{self.separation_arcsec:.1f}\""
               if self.separation_arcsec is not None else "?")
        return (f"{self.name}: {self.main_id} [{self.otype}] at {sep}"
                + (f", {self.n_simbad_objects} sources in field"
                   if self.n_simbad_objects > 1 else ""))


def simbad_lookup(ra: float, dec: float, name: str = "",
                  radius_arcsec: float = 5.0) -> NoveltyRecord:
    """Cone-search SIMBAD around one position and return the nearest source."""
    rec = NoveltyRecord(name=name, ra=ra, dec=dec)
    try:
        import astropy.units as u
        from astropy.coordinates import SkyCoord
        from astroquery.simbad import Simbad

        sim = Simbad()
        try:
            sim.add_votable_fields("otype")
        except Exception:
            pass          # older/newer astroquery: fall back to default columns
        center = SkyCoord(ra, dec, unit="deg")
        table = sim.query_region(center, radius=radius_arcsec * u.arcsec)
        if table is None or len(table) == 0:
            return rec

        rec.n_simbad_objects = len(table)
        # pick the nearest match rather than the first row SIMBAD happens to return
        cols = {c.lower(): c for c in table.colnames}
        ra_col, dec_col = cols.get("ra"), cols.get("dec")
        best, best_sep = 0, None
        if ra_col and dec_col:
            coords = SkyCoord(table[ra_col], table[dec_col], unit="deg")
            seps = center.separation(coords).arcsec
            best = int(seps.argmin())
            best_sep = float(seps[best])
        row = table[best]
        rec.matched = True
        rec.separation_arcsec = best_sep
        rec.main_id = str(row[cols.get("main_id", "main_id")]).strip()
        if "otype" in cols:
            rec.otype = str(row[cols["otype"]]).strip()
        rec.known_agn = any(t.lower() in rec.otype.lower() for t in AGN_TYPES)
    except Exception as e:
        rec.error = f"{type(e).__name__}: {e}"
    return rec


def annotate(candidates, radius_arcsec: float = 5.0,
             pause_s: float = 0.2, verbose: bool = False) -> list:
    """Look up a list of candidates, one at a time, politely.

    `candidates` is any iterable of objects exposing ``name``, ``ra`` and
    ``dec`` (a mapping with those keys also works). Returns a list of
    :class:`NoveltyRecord` in the same order. A short pause between queries
    keeps a few-hundred-object shortlist from hammering the SIMBAD service.
    """
    out = []
    for c in candidates:
        get = c.get if isinstance(c, dict) else (lambda k, d=None: getattr(c, k, d))
        ra, dec = get("ra"), get("dec")
        name = get("name", "") or ""
        if ra is None or dec is None:
            out.append(NoveltyRecord(name=name, error="no coordinates"))
            continue
        rec = simbad_lookup(float(ra), float(dec), name=name,
                            radius_arcsec=radius_arcsec)
        out.append(rec)
        if verbose:
            print("  " + rec.summary(), flush=True)
        if pause_s:
            time.sleep(pause_s)
    return out
