"""Run a discovery campaign: select SDSS quasars, fit them all, rank the odd ones.

Examples
--------
  # pilot: 200 well-measured low-z quasars, 8 workers
  python scripts/campaign.py --n 200 --workers 8 --outdir campaigns/pilot

  # low-S/N regime, where automated QC matters most
  python scripts/campaign.py --n 500 --sn-min 3 --sn-max 8 --outdir campaigns/lowsn

  # re-rank an existing run without refitting anything (uses the checkpoints)
  python scripts/campaign.py --outdir campaigns/pilot --rank-only

  # add SIMBAD literature context to the shortlist
  python scripts/campaign.py --outdir campaigns/pilot --rank-only --vet

Everything runs on the deterministic rule inspector: no API key, no spend.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402  (pins single-threaded BLAS before numpy loads)
config.pin_threads()


def _load_existing(outdir):
    """Rebuild a BatchReport from a previous run's catalog.json."""
    from agn_egent.batch import BatchReport, BatchRow
    path = os.path.join(outdir, "catalog.json")
    if not os.path.exists(path):
        raise SystemExit(f"no catalog.json in {outdir} -- run a campaign there first")
    with open(path) as fh:
        rows = json.load(fh)
    return BatchReport(rows=[BatchRow(**r) for r in rows])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default="campaigns/pilot")
    ap.add_argument("--n", type=int, default=100, help="number of targets")
    ap.add_argument("--z-min", type=float, default=0.05)
    ap.add_argument("--z-max", type=float, default=0.75,
                    help="upper z: Hbeta+[OIII] must stay in SDSS coverage")
    ap.add_argument("--sn-min", type=float, default=5.0, help="Shen broad-Hb S/N floor")
    ap.add_argument("--sn-max", type=float, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--max-iter", type=int, default=4)
    ap.add_argument("--limit", type=int, default=25,
                    help="max candidates listed per class")
    ap.add_argument("--rank-only", action="store_true",
                    help="re-rank an existing catalog.json, do not fit anything")
    ap.add_argument("--vet", action="store_true",
                    help="cross-match the shortlist against SIMBAD (network)")
    ap.add_argument("--no-resume", action="store_true",
                    help="refit objects even if a checkpoint exists")
    ap.add_argument("--no-gallery", action="store_true",
                    help="skip writing the browsable candidate gallery")
    args = ap.parse_args(argv)

    from agn_egent import campaign as C

    os.makedirs(args.outdir, exist_ok=True)

    if args.rank_only:
        report = _load_existing(args.outdir)
        short = C.shortlist(report, limit=args.limit)
        with open(os.path.join(args.outdir, "shortlist.json"), "w") as fh:
            json.dump({k: [r.name for r in v] for k, v in short.items()}, fh, indent=2)
        print(C.yield_table(report, short))
    else:
        print(f"selecting up to {args.n} targets "
              f"(z {args.z_min}-{args.z_max}, S/N>{args.sn_min}) ...", flush=True)
        targets = C.select_sdss_targets(z_min=args.z_min, z_max=args.z_max,
                                        sn_min=args.sn_min, sn_max=args.sn_max,
                                        n=args.n)
        print(f"  {len(targets)} target(s)", flush=True)
        if not targets:
            raise SystemExit("no targets matched the selection")
        out = C.run_campaign(targets, outdir=args.outdir, max_workers=args.workers,
                             max_iterations=args.max_iter, limit=args.limit,
                             resume=not args.no_resume)
        report, short = out["report"], out["shortlist"]

    recs = None
    if args.vet:
        print()
        recs = C.vet_shortlist(short, outdir=args.outdir)
        unknown = [r for r in recs.values() if not r.matched]
        print(f"\n  {len(recs)} candidate(s) checked, "
              f"{len(unknown)} with no SIMBAD counterpart")

    if not args.no_gallery:
        g = C.write_gallery(short, args.outdir, novelty=recs)
        print(f"\ngallery: {g}")

    print(f"\noutputs in {args.outdir}/  "
          "(catalog.csv, catalog.json, shortlist.json, gallery.html, runs/)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
