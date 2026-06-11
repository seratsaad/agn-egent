#!/usr/bin/env python
"""Precompute the showcase decompositions for the static GitHub Pages site.

Runs the agentic pipeline on a curated set of objects, renders each diagnostic
figure into docs/assets/, and writes docs/data.json (measurements, derived M_BH,
agent decisions, and the Shen DR7 literature comparison where available).

Run thread-pinned:
  OMP_NUM_THREADS=1 ... python scripts/build_site.py
"""
import json
import os
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)

import config  # noqa: E402
config.pin_threads()

import numpy as np  # noqa: E402
import agn_egent  # noqa: E402
from agn_egent import (load_sdss, load_row_fits, fetch_sdss_spectrum,  # noqa: E402
                       make_synthetic_spectrum, QsoparConfig, run_agent,
                       RuleInspector, derive, render_diagnostic, query_shen)

DOCS = os.path.join(PROJ, "docs")
ASSETS = os.path.join(DOCS, "assets")
os.makedirs(ASSETS, exist_ok=True)
EX = os.path.join(PROJ, "external", "PyQSOFit", "example", "data")
J0950 = os.path.join(os.path.dirname(PROJ), "j0950", "HETspec", "PSU22-2-010",
                     "spectrum_20220325_0000008_exp01_orange.fits")


def _agent_log(outcome):
    return [{"iter": s.iteration, "verdict": str(s.verdict.overall),
             "action": s.decision.action,
             "remedy": (s.decision.remedy or {}).get("action"),
             "rationale": s.decision.rationale} for s in outcome.steps]


def _measure(res):
    out = {}
    for comp, m in res.lines.items():
        out[comp] = {"fwhm": _f(m.fwhm_kms), "fwhm_err": _f(m.fwhm_err),
                     "flux": _f(m.flux), "snr": _f(m.snr)}
    dq = derive(res, "Hb")
    cont = {"logL5100": _f(res.continuum.get("LogL5100"))}
    derived = None
    if dq is not None:
        derived = {"logMBH": round(dq.log_MBH, 2),
                   "logMBH_err": _f(dq.log_MBH_err),
                   "logLbol": round(dq.log_Lbol, 2),
                   "lambda_edd": round(dq.eddington_ratio, 3)}
    return out, cont, derived


def _f(x):
    try:
        x = float(x)
        return round(x, 1) if np.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def process(obj_id, spec, label, kind, config_=None, shen=None):
    cfg = config_ or QsoparConfig.sdss_optical_default()
    outcome = run_agent(spec, inspector=RuleInspector(), config=cfg,
                        finalize_mc=True, nsamp=25,
                        workdir=os.path.join(PROJ, "data", "runs", f"site_{obj_id}"))
    res = outcome.final_result
    render_diagnostic(res, os.path.join(ASSETS, f"{obj_id}.png"))
    lines, cont, derived = _measure(res)
    rec = {"id": obj_id, "name": label, "kind": kind, "z": round(spec.z, 4),
           "status": outcome.status, "figure": f"assets/{obj_id}.png",
           "lines": lines, "continuum": cont, "derived": derived,
           "agent": _agent_log(outcome), "shen": shen}
    print(f"  [ok] {obj_id}: {outcome.status}, "
          f"Hb FWHM={lines.get('Hb',{}).get('fwhm')} "
          f"logMBH={derived['logMBH'] if derived else None}")
    return rec


def main():
    records = []

    # 1) the canonical SDSS example -- the agent fixes a degenerate broad Hbeta
    print("SDSS example spec-0332...")
    records.append(process(
        "spec-0332", load_sdss(os.path.join(EX, "spec-0332-52367-0639.fits"),
                               name="spec-0332"),
        "SDSS J1016+0034 (spec-0332-52367-0639)", "SDSS example (low-z AGN)"))

    # 2) real Shen DR7 quasars with literature comparison (the accuracy highlight)
    for (p, m, f) in [(388, 51793, 445), (389, 51795, 409), (2630, 54327, 149)]:
        oid = f"shen-{p}-{m}-{f}"
        print(f"Shen quasar {p}-{m}-{f}...")
        try:
            shen = query_shen(p, m, f)
            spec = fetch_sdss_spectrum(p, m, f, name=oid)
            cfg = QsoparConfig.sdss_optical_default().set_ngauss("Hb_br", 1)
            shen_d = {"logMBH": round(shen.log_MBH_hbeta, 2),
                      "fwhm": round(shen.fwhm_hbeta, 0),
                      "logL5100": round(shen.log_L5100, 2)}
            records.append(process(oid, spec, f"SDSS {p}-{m}-{f}",
                                   "SDSS quasar (Shen DR7)", cfg, shen_d))
        except Exception as e:
            print(f"  [skip] {oid}: {type(e).__name__}: {str(e)[:80]}")

    # 3) J0950 -- heterogeneous instrument (HET/LRS2)
    if os.path.exists(J0950):
        print("J0950 (HET/LRS2)...")
        try:
            spec = load_row_fits(J0950, z=0.2144, ra=147.6531250, dec=51.4772500,
                                 flux_scale=1e17, name="J0950")
            records.append(process("j0950", spec, "J0950+5128 (HET/LRS2)",
                                   "Heterogeneous instrument"))
        except Exception as e:
            print(f"  [skip] j0950: {type(e).__name__}: {str(e)[:80]}")

    # 4) synthetic injection-recovery (known truth)
    print("Synthetic injection...")
    spec, truth = make_synthetic_spectrum(hbeta_fwhm=4200, halpha_fwhm=4600,
                                          snr=30, seed=1, name="synthetic")
    cfg = QsoparConfig.sdss_optical_default()
    for k, v in {"decompose_host": False, "Fe_uv_op": False, "poly": False}.items():
        cfg = cfg.set_fit_option(k, v)
    rec = process("synthetic", spec, "Synthetic spectrum (known truth)",
                  "Injection-recovery", cfg)
    rec["truth"] = {"hbeta_fwhm": round(truth["hbeta_fwhm"]),
                    "halpha_fwhm": round(truth["halpha_fwhm"])}
    records.append(rec)

    with open(os.path.join(DOCS, "data.json"), "w") as fh:
        json.dump(records, fh, indent=1)
    print(f"\nwrote docs/data.json with {len(records)} objects")


if __name__ == "__main__":
    main()
