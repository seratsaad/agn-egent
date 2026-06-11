#!/usr/bin/env python
"""Compare global vs region-by-region decomposition against Shen DR7.

Three pipelines per object:
  global   -- single global continuum (anchored), all lines fit together
  region   -- Hbeta and Halpha fit in separate windows, each on a local
              anchored continuum (no per-region QC)
  region+qc-- region split with the per-region verify->remedy loop on

Reports median |Δ| vs the published Shen log L5100 / log M_BH(Hbeta).
"""
import argparse
import os
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)
import config  # noqa: E402
config.pin_threads()

import numpy as np  # noqa: E402
from agn_egent import (find_shen_quasars, query_shen, fetch_sdss_spectrum,  # noqa: E402
                       decompose, QsoparConfig, RuleContinuumPlanner,
                       apply_plan_to_config, decompose_by_region, derive)

RUNS = os.path.join(PROJ, "data", "runs", "region_bench")


def global_anchored(spec, wd):
    base = decompose(spec, config=QsoparConfig.sdss_optical_default(),
                     make_figure=False, workdir=os.path.join(wd, "gbase"))
    plan = RuleContinuumPlanner().plan(spec, result=base)
    cfg = apply_plan_to_config(QsoparConfig.sdss_optical_default(), plan)
    return decompose(spec, config=cfg, make_figure=False,
                     workdir=os.path.join(wd, "global"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--z-min", type=float, default=0.1)
    ap.add_argument("--z-max", type=float, default=0.35)   # keep Halpha in range
    ap.add_argument("--sn-min", type=float, default=20.0)
    args = ap.parse_args()
    os.makedirs(RUNS, exist_ok=True)

    targets = find_shen_quasars(z_min=args.z_min, z_max=args.z_max,
                                sn_min=args.sn_min, fwhm_min=2000, n=args.n)
    print(f"fetched {len(targets)} Shen targets")
    rows = []
    for (p, m, f) in targets:
        oid = f"{p}-{m}-{f}"
        wd = os.path.join(RUNS, oid)
        try:
            shen = query_shen(p, m, f)
            spec = fetch_sdss_spectrum(p, m, f, name=oid)
            g = global_anchored(spec, wd)
            r = decompose_by_region(spec, workdir=os.path.join(wd, "region"),
                                    anchor=True, qc=False)
            rq = decompose_by_region(spec, workdir=os.path.join(wd, "region_qc"),
                                     anchor=True, qc=True)
            dg, dr, dq = derive(g, "Hb"), derive(r.merged, "Hb"), derive(rq.merged, "Hb")
            row = {"id": oid, "shen_mbh": shen.log_MBH_hbeta, "shen_l": shen.log_L5100,
                   "g_mbh": dg.log_MBH if dg else np.nan,
                   "r_mbh": dr.log_MBH if dr else np.nan,
                   "q_mbh": dq.log_MBH if dq else np.nan,
                   "g_l": g.continuum.get("LogL5100", np.nan),
                   "r_l": r.merged.continuum.get("LogL5100", np.nan),
                   "q_l": rq.merged.continuum.get("LogL5100", np.nan)}
            rows.append(row)
            print(f"  {oid}: Shen M={shen.log_MBH_hbeta:.2f}  "
                  f"global={row['g_mbh']:.2f} region={row['r_mbh']:.2f} "
                  f"region+qc={row['q_mbh']:.2f}")
        except Exception as e:
            print(f"  [skip] {oid}: {e}")

    def med_abs(key, ref):
        v = [abs(r[key] - r[ref]) for r in rows
             if np.isfinite(r[key]) and np.isfinite(r[ref])]
        return np.median(v) if v else np.nan

    print(f"\nN={len(rows)}  median |Δ| vs Shen:")
    print(f"  logM_BH : global={med_abs('g_mbh','shen_mbh'):.3f}  "
          f"region={med_abs('r_mbh','shen_mbh'):.3f}  "
          f"region+qc={med_abs('q_mbh','shen_mbh'):.3f}")
    print(f"  logL5100: global={med_abs('g_l','shen_l'):.3f}  "
          f"region={med_abs('r_l','shen_l'):.3f}  "
          f"region+qc={med_abs('q_l','shen_l'):.3f}")


if __name__ == "__main__":
    main()
