"""Validate DESI-derived masses against the published SDSS/Shen catalog.

DESI and SDSS observed many of the same quasars with different instruments,
fibers, epochs and reduction pipelines. Running our engine on the DESI spectrum
and comparing to Shen et al. (2011)'s independently-measured SDSS mass is
therefore a genuine cross-instrument check: it folds in everything that differs
between the two surveys, not just our fitting.

What agreement means here, and what it does not: Shen's masses are themselves
single-epoch estimates with ~0.4 dex scatter, so this measures *reproducibility
across instruments*, not accuracy. (The accuracy anchor remains the
reverberation-mapped comparison, N=24, bias +0.02 dex.) A tight, unbiased
distribution says the DESI path is not introducing a systematic; a shifted one
would say it is.

Usage:
    python scripts/validate_desi.py --outdir campaigns/desi_pilot
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
config.pin_threads()

import numpy as np  # noqa: E402

SHEN = "J/ApJS/194/45"
MATCH_RADIUS_ARCSEC = 2.0


def crossmatch_shen(rows, radius_arcsec=MATCH_RADIUS_ARCSEC):
    """Bulk cone-match our rows against Shen DR7; returns name -> shen logMBH."""
    import astropy.units as u
    from astropy.table import Table
    from astroquery.vizier import Vizier

    usable = [r for r in rows
              if r.get("ra") is not None and r.get("dec") is not None
              and (r.get("derived") or {}).get("log_MBH") is not None]
    if not usable:
        return {}, []
    coords = Table({"_RAJ2000": [r["ra"] for r in usable],
                    "_DEJ2000": [r["dec"] for r in usable]})
    V = Vizier(columns=["RAJ2000", "DEJ2000", "logBHHM", "logL5100",
                        "W(BHb)", "z", "_q"])
    V.ROW_LIMIT = -1
    res = V.query_region(coords, radius=radius_arcsec * u.arcsec, catalog=SHEN)
    if not res:
        return {}, usable
    t = res[0]
    out = {}
    for row in t:
        try:
            i = int(row["_q"]) - 1          # VizieR is 1-indexed
            shen_m = float(row["logBHHM"])
        except (KeyError, TypeError, ValueError):
            continue
        if not np.isfinite(shen_m) or not (0 <= i < len(usable)):
            continue
        name = usable[i]["name"]
        out.setdefault(name, shen_m)        # first (nearest) match wins
    return out, usable


def robust_scatter(x):
    """1.4826 * MAD -- the outlier-resistant sigma used throughout this project."""
    x = np.asarray(x, dtype=float)
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if x.size else float("nan")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default="campaigns/desi_pilot")
    ap.add_argument("--trusted-only", action="store_true",
                    help="restrict to clean/reviewed fits (the intended use)")
    args = ap.parse_args(argv)

    with open(os.path.join(args.outdir, "catalog.json")) as fh:
        rows = json.load(fh)
    if args.trusted_only:
        rows = [r for r in rows if r.get("quality_flag") in ("clean", "reviewed")]

    print(f"cross-matching {len(rows)} DESI rows against Shen DR7 "
          f"({MATCH_RADIUS_ARCSEC}\") ...", flush=True)
    shen, usable = crossmatch_shen(rows)
    print(f"  {len(usable)} with a mass and coordinates, {len(shen)} matched")
    if not shen:
        print("no overlap -- nothing to validate (DESI footprint may miss DR7 here)")
        return 0

    d = []
    for r in usable:
        if r["name"] in shen:
            ours = float(r["derived"]["log_MBH"])
            d.append((r["name"], ours, shen[r["name"]], ours - shen[r["name"]]))
    diffs = np.array([x[3] for x in d])
    print()
    print(f"N matched          : {len(diffs)}")
    print(f"median dlog M_BH   : {np.median(diffs):+.3f} dex   (DESI - Shen/SDSS)")
    print(f"mean               : {np.mean(diffs):+.3f} dex")
    print(f"robust scatter     : {robust_scatter(diffs):.3f} dex")
    print(f"|dlog| > 0.5 dex   : {(np.abs(diffs) > 0.5).sum()} "
          f"({100*(np.abs(diffs) > 0.5).mean():.0f}%)")
    print()
    print("Reference points: Shen's own single-epoch masses carry ~0.4 dex")
    print("scatter, and our SDSS clean-sample comparison gave ~0.2 dex. A")
    print("comparable scatter with a near-zero median means the DESI path")
    print("introduces no systematic of its own.")

    worst = sorted(d, key=lambda t: -abs(t[3]))[:5]
    print("\nlargest disagreements (worth an eyeball):")
    for nm, ours, sh, dd in worst:
        print(f"  {nm}: ours {ours:.2f} vs Shen {sh:.2f}  ({dd:+.2f})")

    out = os.path.join(args.outdir, "validation_shen.json")
    with open(out, "w") as fh:
        json.dump({"n": len(diffs), "median_dlog": float(np.median(diffs)),
                   "robust_scatter": robust_scatter(diffs),
                   "objects": [{"name": n, "ours": o, "shen": s, "dlog": x}
                               for n, o, s, x in d]}, fh, indent=2)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
