#!/usr/bin/env python
"""Prototype: does visual-first continuum anchoring beat the global fit?

For each object we run two decompositions and compare them (and, when available,
the Shen DR7 truth):

  baseline  -- the current default config (PyQSOFit global continuum fit)
  anchored  -- two-stage: baseline fit -> plan the continuum on the host-subtracted
               QSO flux (RuleContinuumPlanner lower-envelope window selection) ->
               refit with the power-law slope pinned + windows restricted.

The falsifiable claim: on objects where the baseline power-law slope *rails*
against its physical bound (the continuum is mis-placed), anchoring should move
log L5100 / log M_BH toward the published Shen value.

Usage:
  OMP_NUM_THREADS=1 ... python scripts/anchor_prototype.py            # Shen sample
  OMP_NUM_THREADS=1 ... python scripts/anchor_prototype.py --bundled  # offline
"""
import argparse
import os
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)

import config  # noqa: E402
config.pin_threads()

import numpy as np  # noqa: E402

import agn_egent as ae  # noqa: E402
from agn_egent import (load_sdss, decompose, QsoparConfig, derive,  # noqa: E402
                       RuleContinuumPlanner, apply_plan_to_config,
                       render_continuum_plan, render_diagnostic, verify)

RUNS = os.path.join(PROJ, "data", "runs", "anchor_proto")

# bundled, network-free spectra (z, name)
BUNDLED = [
    ("external/PyQSOFit/example/data/spec-0332-52367-0639.fits", "spec-0332"),
    ("external/PyQSOFit/example/data/spec-0266-51602-0013.fits", "spec-0266-0013"),
    ("external/PyQSOFit/example/data/spec-0266-51602-0107.fits", "spec-0266-0107"),
]

SLOPE_FLOOR, SLOPE_CEIL = -3.0, 0.0


def _railed(slope) -> bool:
    return slope is not None and np.isfinite(slope) and (
        slope <= SLOPE_FLOOR + 0.05 or slope >= SLOPE_CEIL - 0.05)


def run_one(spec, shen=None, planner=None, save_figs=True):
    planner = planner or RuleContinuumPlanner()
    odir = os.path.join(RUNS, spec.name or "obj")
    os.makedirs(odir, exist_ok=True)

    base = decompose(spec, config=QsoparConfig.sdss_optical_default(),
                     make_figure=False, workdir=os.path.join(odir, "base"))
    plan = planner.plan(spec, result=base)
    cfg2 = apply_plan_to_config(QsoparConfig.sdss_optical_default(), plan)
    anc = decompose(spec, config=cfg2, make_figure=False,
                    workdir=os.path.join(odir, "anc"))

    if save_figs:
        render_continuum_plan(spec, plan, os.path.join(odir, "plan.png"), result=base)
        render_diagnostic(base, os.path.join(odir, "base.png"), report=verify(base))
        render_diagnostic(anc, os.path.join(odir, "anc.png"), report=verify(anc))

    db, da = derive(base, "Hb"), derive(anc, "Hb")
    row = {
        "id": spec.name, "z": spec.z, "class": plan.classification,
        "n_win": len(plan.used_windows),
        "base_slope": base.continuum.get("PL_slope", np.nan),
        "anc_slope": anc.continuum.get("PL_slope", np.nan),
        "base_l5100": base.continuum.get("LogL5100", np.nan),
        "anc_l5100": anc.continuum.get("LogL5100", np.nan),
        "base_mbh": db.log_MBH if db else np.nan,
        "anc_mbh": da.log_MBH if da else np.nan,
        "shen_l5100": shen.log_L5100 if shen else np.nan,
        "shen_mbh": shen.log_MBH_hbeta if shen else np.nan,
    }
    row["base_railed"] = _railed(row["base_slope"])
    return row


def _fmt(x, p=2):
    return "   nan" if x is None or not np.isfinite(x) else f"{x:+.{p}f}" if p else f"{x:.0f}"


def print_table(rows):
    hdr = (f"{'object':16s} {'z':>5s} {'class':14s} {'slope_b':>8s} {'slope_a':>8s} "
           f"{'L_b':>6s} {'L_a':>6s} {'L_shen':>7s} {'M_b':>5s} {'M_a':>5s} {'M_shen':>7s} rail")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['id']:16s} {r['z']:5.3f} {r['class']:14s} "
              f"{_fmt(r['base_slope']):>8s} {_fmt(r['anc_slope']):>8s} "
              f"{r['base_l5100']:6.2f} {r['anc_l5100']:6.2f} "
              f"{_fmt(r['shen_l5100'],2):>7s} "
              f"{r['base_mbh']:5.2f} {r['anc_mbh']:5.2f} "
              f"{_fmt(r['shen_mbh'],2):>7s} {'RAIL' if r['base_railed'] else ''}")

    # accuracy summary on objects with Shen truth
    truth = [r for r in rows if np.isfinite(r["shen_mbh"])]
    if truth:
        def dabs(rows_, key, ref):
            v = [abs(r[key] - r[ref]) for r in rows_ if np.isfinite(r[key]) and np.isfinite(r[ref])]
            return np.median(v) if v else np.nan
        print("\nAccuracy vs Shen (median |Δ|):")
        for label, rs in (("all", truth),
                          ("baseline-railed", [r for r in truth if r["base_railed"]])):
            if not rs:
                continue
            print(f"  {label:16s} (N={len(rs):2d})  "
                  f"logM_BH: base={dabs(rs,'base_mbh','shen_mbh'):.3f} "
                  f"anc={dabs(rs,'anc_mbh','shen_mbh'):.3f}   "
                  f"logL5100: base={dabs(rs,'base_l5100','shen_l5100'):.3f} "
                  f"anc={dabs(rs,'anc_l5100','shen_l5100'):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundled", action="store_true",
                    help="offline: run on bundled example spectra (no Shen truth)")
    ap.add_argument("--n", type=int, default=12, help="number of Shen quasars")
    ap.add_argument("--z-min", type=float, default=0.1)
    ap.add_argument("--z-max", type=float, default=0.5)
    ap.add_argument("--sn-min", type=float, default=10.0)
    args = ap.parse_args()

    os.makedirs(RUNS, exist_ok=True)
    rows = []

    if args.bundled:
        for path, name in BUNDLED:
            try:
                spec = load_sdss(os.path.join(PROJ, path), name=name)
                rows.append(run_one(spec))
                print(f"  done {name}")
            except Exception as e:
                print(f"  [skip] {name}: {e}")
    else:
        from agn_egent import find_shen_quasars, query_shen, fetch_sdss_spectrum
        targets = find_shen_quasars(z_min=args.z_min, z_max=args.z_max,
                                    sn_min=args.sn_min, fwhm_min=1500, n=args.n)
        print(f"fetched {len(targets)} Shen targets")
        for (p, m, f) in targets:
            oid = f"{p}-{m}-{f}"
            try:
                shen = query_shen(p, m, f)
                spec = fetch_sdss_spectrum(p, m, f, name=oid)
                rows.append(run_one(spec, shen=shen))
                print(f"  done {oid}  (base railed={rows[-1]['base_railed']})")
            except Exception as e:
                print(f"  [skip] {oid}: {e}")

    print()
    print_table(rows)
    print(f"\nfigures + provenance under {RUNS}")


if __name__ == "__main__":
    main()
