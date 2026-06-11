#!/usr/bin/env python
"""Scaled accuracy benchmark vs. Shen et al. (2011) DR7, with Egent-style metrics.

Runs AGN-Egent on N real SDSS quasars and compares the recovered single-epoch
M_BH / L5100 / FWHM to the published catalog. Reports the *distribution* of
offsets (median, robust scatter, bias) and a per-object regression slope
(cf. Egent's per-spectrum slopes ~0.85-1.19), plus the Egent-style quality-tier
fractions. Saves a scatter plot to docs/assets/benchmark.png and a JSON summary.

Usage:
  OMP_NUM_THREADS=1 ... python scripts/benchmark_shen.py --n 20 [--inspector rule]
"""
import argparse
import json
import os
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)

import config  # noqa: E402
config.pin_threads()

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import agn_egent  # noqa: E402
from agn_egent import (find_shen_quasars, query_shen, fetch_sdss_spectrum,  # noqa: E402
                       run_agent, QsoparConfig, derive, make_inspector)


def _robust_scatter(x):
    x = np.asarray(x)
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def benchmark(n=20, inspector_name="rule", api_key=None, z_min=0.1, z_max=0.7,
              sn_min=20.0):
    insp = make_inspector(inspector_name, api_key=api_key)
    targets = find_shen_quasars(z_min=z_min, z_max=z_max, sn_min=sn_min,
                                fwhm_min=1500, n=n)
    print(f"benchmarking {len(targets)} Shen quasars with the {inspector_name} reviewer...")
    rows = []
    for (p, m, f) in targets:
        oid = f"{p}-{m}-{f}"
        try:
            shen = query_shen(p, m, f)
            spec = fetch_sdss_spectrum(p, m, f, name=oid)
            cfg = QsoparConfig.sdss_optical_default()
            out = run_agent(spec, inspector=insp, config=cfg, finalize_mc=True,
                            nsamp=20, workdir=os.path.join(PROJ, "data", "runs", f"bench_{oid}"))
            dq = derive(out.final_result, "Hb")
            if dq is None:
                print(f"  [skip] {oid}: no broad Hb")
                continue
            rows.append({"id": oid, "z": shen.z, "flag": out.quality_flag,
                         "our_mbh": dq.log_MBH, "shen_mbh": shen.log_MBH_hbeta,
                         "our_l5100": dq.log_L5100, "shen_l5100": shen.log_L5100,
                         "our_fwhm": out.final_result.lines["Hb"].fwhm_kms,
                         "shen_fwhm": shen.fwhm_hbeta})
            print(f"  {oid}: logMBH ours/Shen={dq.log_MBH:.2f}/{shen.log_MBH_hbeta:.2f} "
                  f"[{out.quality_flag}]")
        except Exception as e:
            print(f"  [skip] {oid}: {type(e).__name__}: {str(e)[:70]}")
    return rows


def _metrics(rows):
    dmbh = np.array([r["our_mbh"] - r["shen_mbh"] for r in rows])
    dl = np.array([r["our_l5100"] - r["shen_l5100"] for r in rows])
    our = np.array([r["our_mbh"] for r in rows]); shen = np.array([r["shen_mbh"] for r in rows])
    # robust (Theil-Sen) slope -- not dragged by a single leverage outlier
    if len(rows) > 2:
        from scipy.stats import theilslopes
        slope = float(theilslopes(our, shen)[0])
    else:
        slope = float("nan")
    return {"n": len(rows),
            "median_dlogMBH": round(float(np.median(dmbh)), 3),
            "scatter_dlogMBH": round(float(_robust_scatter(dmbh)), 3),
            "bias_dlogMBH": round(float(np.mean(dmbh)), 3),
            "median_dlogL5100": round(float(np.median(dl)), 3),
            "regression_slope": round(slope, 3)}


def report(rows, inspector_name):
    if not rows:
        print("no objects completed."); return None
    n = len(rows)
    flags = {}
    for r in rows:
        flags[r["flag"]] = flags.get(r["flag"], 0) + 1
    accepted = [r for r in rows if r["flag"] != "flagged"]
    summary = {"inspector": inspector_name, "n_total": n,
               "quality_fractions": {k: round(v / n, 3) for k, v in flags.items()},
               "all": _metrics(rows),
               "accepted": _metrics(accepted) if accepted else None}

    a = summary["accepted"] or summary["all"]
    print("\n=== ACCURACY vs Shen DR7 ===")
    print(f"  N = {n}  (reviewer: {inspector_name})")
    print(f"  quality tiers     = {summary['quality_fractions']}")
    print(f"  -- ACCEPTED (clean+reviewed, N={a['n']}) -- the measurements you keep:")
    print(f"     median ΔlogM_BH = {a['median_dlogMBH']:+.2f} dex   "
          f"robust scatter = {a['scatter_dlogMBH']:.2f} dex   slope = {a['regression_slope']:.2f}")
    print(f"  -- ALL N={n}: median {summary['all']['median_dlogMBH']:+.2f}, "
          f"scatter {summary['all']['scatter_dlogMBH']:.2f}, slope {summary['all']['regression_slope']:.2f}")

    # scatter plot, coloured by quality flag
    fig, ax = plt.subplots(figsize=(5.2, 5))
    our = np.array([r["our_mbh"] for r in rows]); shen = np.array([r["shen_mbh"] for r in rows])
    lim = [min(shen.min(), our.min()) - 0.3, max(shen.max(), our.max()) + 0.3]
    ax.plot(lim, lim, "k--", lw=1, alpha=.6)
    for flag, col, lab in [("clean", "#43c59e", "clean"), ("reviewed", "#5b8cff", "reviewed"),
                           ("flagged", "#d1495b", "flagged (excluded)")]:
        pts = [r for r in rows if r["flag"] == flag]
        if pts:
            ax.scatter([r["shen_mbh"] for r in pts], [r["our_mbh"] for r in pts],
                       c=col, s=42, edgecolor="white", linewidth=.5, zorder=3, label=lab)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Shen DR7  log M$_{BH}$"); ax.set_ylabel("AGN-Egent  log M$_{BH}$")
    ax.set_title(f"accepted N={a['n']}:  scatter={a['scatter_dlogMBH']:.2f} dex  "
                 f"slope={a['regression_slope']:.2f}")
    fig.tight_layout()
    out_png = os.path.join(PROJ, "docs", "assets", "benchmark.png")
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=140)
    print(f"\n  scatter plot -> {out_png}")

    with open(os.path.join(PROJ, "docs", "benchmark.json"), "w") as fh:
        json.dump({"summary": summary, "objects": rows}, fh, indent=1)
    print(f"  summary -> docs/benchmark.json")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--inspector", default="rule", choices=["rule", "claude", "openai"])
    ap.add_argument("--api-key", default=None)
    args = ap.parse_args()
    key = args.api_key or os.environ.get(
        {"claude": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}.get(args.inspector, ""))
    rows = benchmark(n=args.n, inspector_name=args.inspector, api_key=key)
    report(rows, args.inspector)


if __name__ == "__main__":
    main()
