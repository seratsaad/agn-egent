"""DESI discovery campaign: fit every optically-reachable DESI DR1 quasar.

Differences from the SDSS campaign runner, both driven by scale (~273k
objects in the z<0.95 window):

* Spectra come from NOIRLab's SPARCL service in **batched retrievals** (a few
  hundred per call), not one HTTP request per object -- polite to the service
  and much faster. Fetching a chunk overlaps with fitting nothing, but chunks
  are small enough (~1 min of fitting each) that the download fraction is tiny.
* Output is **packed**: one sharded row JSON per object, no per-object
  directories, no figures. A quarter-million objects live comfortably inside a
  cluster inode quota; shortlisted candidates get refit with full diagnostics
  afterwards.

Zero spend: deterministic rule inspector, free public services.

Examples
--------
  # local pilot
  python scripts/desi_campaign.py --n 2000 --workers 7 --outdir campaigns/desi_pilot

  # the full optical-window sample (run on the cluster)
  python scripts/desi_campaign.py --n 300000 --workers 40 --outdir campaigns/desi_full
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
config.pin_threads()

RETRIEVE_INCLUDE = ["wavelength", "flux", "ivar", "redshift", "ra", "dec",
                    "targetid"]


def existing_rows(outdir):
    """names already checkpointed (packed layout)."""
    return {os.path.splitext(os.path.basename(p))[0]
            for p in glob.glob(os.path.join(outdir, "runs", "rows", "*", "*.json"))}


def load_rows(outdir):
    rows = []
    for p in glob.glob(os.path.join(outdir, "runs", "rows", "*", "*.json")):
        try:
            with open(p) as fh:
                rows.append(json.load(fh))
        except Exception:
            pass
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default="campaigns/desi_pilot")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--z-min", type=float, default=0.05)
    ap.add_argument("--z-max", type=float, default=0.95,
                    help="0.95 keeps Hbeta+[OIII] inside DESI coverage")
    ap.add_argument("--chunk", type=int, default=400,
                    help="objects per SPARCL retrieval")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--max-iter", type=int, default=4)
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--rank-only", action="store_true",
                    help="assemble + rank existing rows, fit nothing")
    args = ap.parse_args(argv)

    from agn_egent import run_batch
    from agn_egent.batch import BatchReport, BatchRow
    from agn_egent import campaign as C
    from agn_egent.io_desi import (find_desi_qsos, spectrum_from_sparcl_record,
                                   _client)

    os.makedirs(args.outdir, exist_ok=True)
    runs = os.path.join(args.outdir, "runs")

    # -- target list: persisted on first run so resumes see the same sample --
    tpath = os.path.join(args.outdir, "targets.json")
    if os.path.exists(tpath):
        with open(tpath) as fh:
            targets = json.load(fh)
        print(f"resuming with {len(targets)} persisted targets", flush=True)
    else:
        print(f"selecting up to {args.n} DESI QSOs "
              f"(z {args.z_min}-{args.z_max}) via SPARCL ...", flush=True)
        targets = find_desi_qsos(z_min=args.z_min, z_max=args.z_max, n=args.n)
        with open(tpath, "w") as fh:
            json.dump(targets, fh)
        print(f"  {len(targets)} target(s)", flush=True)

    if not args.rank_only:
        client = _client()
        done = existing_rows(args.outdir)
        todo = [t for t in targets if t["name"] not in done]
        print(f"{len(done)} already done, {len(todo)} to fit", flush=True)

        for ci in range(0, len(todo), args.chunk):
            chunk = todo[ci:ci + args.chunk]
            got = client.retrieve(uuid_list=[t["sparcl_id"] for t in chunk],
                                  include=RETRIEVE_INCLUDE)
            zmap = {f"desi-{r.targetid}": next(
                        (t["z"] for t in chunk if t["name"] == f"desi-{r.targetid}"),
                        None) for r in got.records}
            specs = []
            for r in got.records:
                try:
                    specs.append(spectrum_from_sparcl_record(
                        r, z=zmap.get(f"desi-{r.targetid}")))
                except Exception as e:
                    print(f"  skip desi-{getattr(r,'targetid','?')}: {e}",
                          flush=True)
            run_batch(specs, max_workers=args.workers,
                      max_iterations=args.max_iter, workdir=runs,
                      science=True, resume=True, pack=True)
            n_done = len(existing_rows(args.outdir))
            print(f"[chunk {ci//args.chunk + 1}] "
                  f"{n_done}/{len(targets)} done", flush=True)

    # -- assemble the catalog and rank -------------------------------------
    rows = load_rows(args.outdir)
    by_name = {t["name"]: t for t in targets}
    report = BatchReport(rows=[BatchRow(**r) for r in rows])
    for r in report.rows:
        t = by_name.get(r.name)
        if t:
            r.ra, r.dec = t.get("ra"), t.get("dec")
            if r.z is None:
                r.z = t.get("z")
    report.to_csv(os.path.join(args.outdir, "catalog.csv"))
    report.to_json(os.path.join(args.outdir, "catalog.json"))
    short = C.shortlist(report, limit=args.limit)
    with open(os.path.join(args.outdir, "shortlist.json"), "w") as fh:
        json.dump({k: [r.name for r in v] for k, v in short.items()}, fh, indent=2)
    print()
    print(C.yield_table(report, short))
    print(f"\noutputs in {args.outdir}/ (catalog.csv/json, shortlist.json, runs/rows/)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
