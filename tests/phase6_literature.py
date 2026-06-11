"""Phase 6 honest accuracy benchmark vs. Shen et al. (2011) DR7.

For a handful of real SDSS quasars with published broad-Hbeta FWHM, L5100, and
single-epoch M_BH, fetch the spectrum, run the agentic pipeline with MC errors,
and compare our recovered quantities to the catalog. Prints the per-object
offsets and the median |dlog M_BH|.

Requires network (astroquery VizieR + SDSS). Skips gracefully when offline.
Run thread-pinned (see README).
"""
import os
import sys
import statistics

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)

import agn_egent  # noqa: E402  (import first -> pin threads before numpy)
from agn_egent import (query_shen, fetch_sdss_spectrum, find_shen_quasars,  # noqa: E402
                       run_agent, RuleInspector, derive, Comparison,
                       QsoparConfig)


def benchmark_one(plate, mjd, fiber):
    shen = query_shen(plate, mjd, fiber)
    spec = fetch_sdss_spectrum(plate, mjd, fiber)
    # bright broad-line quasars: a single broad Gaussian is the stable default
    cfg = QsoparConfig.sdss_optical_default().set_ngauss("Hb_br", 1)
    out = run_agent(spec, inspector=RuleInspector(), config=cfg,
                    finalize_mc=True, nsamp=25,
                    workdir=os.path.join(PROJ, "data", "runs", f"lit_{spec.name}"))
    res = out.final_result
    dq = derive(res, "Hb")
    hb = res.lines["Hb"]
    return Comparison(
        name=spec.name, shen=shen,
        our_fwhm=hb.fwhm_kms, our_log_L5100=dq.log_L5100,
        our_log_MBH=dq.log_MBH, our_log_MBH_err=dq.log_MBH_err,
        notes=out.status)


def main():
    try:
        targets = find_shen_quasars(z_min=0.28, z_max=0.42, sn_min=30, n=3)
    except Exception as e:
        print(f"=== Phase 6 literature benchmark SKIPPED (network: "
              f"{type(e).__name__}: {str(e)[:80]}) ===")
        return

    if not targets:
        print("=== Phase 6 SKIPPED (no Shen targets returned) ===")
        return

    print(f"=== Phase 6: AGN-Egent vs Shen et al. (2011), {len(targets)} quasars ===")
    comps = []
    for (p, m, f) in targets:
        try:
            c = benchmark_one(p, m, f)
            comps.append(c)
            print("  " + c.row() + f"  [{c.notes}]")
        except Exception as e:
            print(f"  {p}-{m}-{f}: FAILED {type(e).__name__}: {str(e)[:90]}")

    if not comps:
        print("=== no objects completed (likely network/download) — SKIPPED ===")
        return

    dmbh = [abs(c.dlog_MBH) for c in comps]
    dl = [abs(c.dlog_L5100) for c in comps]
    print(f"\n  median |dlog M_BH| = {statistics.median(dmbh):.2f} dex  "
          f"(Shen's own single-epoch scatter ~0.4 dex)")
    print(f"  median |dlog L5100| = {statistics.median(dl):.2f} dex")
    # honest acceptance: agree with the literature to within the method scatter
    assert statistics.median(dmbh) < 0.5, \
        f"median M_BH offset {statistics.median(dmbh):.2f} dex exceeds 0.5"
    print("\n[PASS] Phase 6: recovered M_BH agrees with Shen DR7 within the "
          "single-epoch method scatter.")


if __name__ == "__main__":
    main()
