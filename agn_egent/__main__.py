"""AGN-Egent command-line tool: one spectrum in, a trustworthy mass out.

Examples
--------
  # an SDSS object by plate-mjd-fiber
  python -m agn_egent 650-52143-166

  # a local SDSS-style FITS file
  python -m agn_egent --fits spec.fits --z 0.29

  # use a vision-LLM inspector instead of the deterministic rules (needs an API key)
  python -m agn_egent 650-52143-166 --inspector gemini

Outputs, under --outdir (default ./agn_egent_out/<name>/):
  result.json       derived M_BH / L_bol / Eddington (+ MC errors) and the quality flag
  provenance.json   the full agent trajectory (every config, verdict, decision)
  diagnostic.png    the per-line zoom diagnostic of the accepted fit

By default it runs fully deterministically (no API, no cost) with rule-first gating.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys


def _load_spectrum(args):
    from . import fetch_sdss_spectrum, spectrum_from_hdulist
    if args.fits:
        from astropy.io import fits
        name = args.name or os.path.splitext(os.path.basename(args.fits))[0]
        with fits.open(args.fits) as hdul:
            spec = spectrum_from_hdulist(hdul, name=name)
        if args.z is not None:
            spec.z = args.z
        return spec
    p, m, f = (int(x) for x in args.target.split("-"))
    return fetch_sdss_spectrum(p, m, f, name=args.name or args.target)


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="python -m agn_egent",
        description="Measure a single-epoch AGN black-hole mass from one spectrum.")
    ap.add_argument("target", nargs="?",
                    help="SDSS spectrum id as PLATE-MJD-FIBER (e.g. 650-52143-166)")
    ap.add_argument("--fits", help="local SDSS-style FITS spectrum (instead of an id)")
    ap.add_argument("--z", type=float, help="redshift (required with --fits if absent)")
    ap.add_argument("--name", help="object name for outputs")
    ap.add_argument("--inspector", default="rule",
                    choices=["rule", "claude", "openai", "gemini"],
                    help="QC reviewer (default rule = deterministic, free)")
    ap.add_argument("--model", help="LLM model id (only for an LLM inspector)")
    ap.add_argument("--max-iter", type=int, default=4)
    ap.add_argument("--no-mc", action="store_true", help="skip Monte-Carlo errors")
    ap.add_argument("--outdir", default="agn_egent_out")
    args = ap.parse_args(argv)

    if not args.target and not args.fits:
        ap.error("provide a PLATE-MJD-FIBER target or --fits FILE")

    # importing the package pins single-threaded math for reproducibility
    # (agn_egent/__init__ calls repro.pin_threads before numpy loads). Do not
    # use the root-level config.py here -- it is not installed with the package,
    # so an installed CLI would fail outside a repo checkout.
    from . import run_agent, run_agent_escalate, make_inspector, derive

    try:
        spec = _load_spectrum(args)
    except Exception as e:
        print(f"ERROR loading spectrum: {type(e).__name__}: {e}", file=sys.stderr)
        return 2

    name = spec.name or "object"
    workdir = os.path.join(args.outdir, name)
    os.makedirs(workdir, exist_ok=True)

    finalize_mc = not args.no_mc
    if args.inspector == "rule":
        out = run_agent(spec, inspector=make_inspector("rule"),
                        finalize_mc=finalize_mc, max_iterations=args.max_iter,
                        workdir=os.path.join(workdir, "run"))
        escalated = False
    else:
        llm = make_inspector(args.inspector, model=args.model)
        out, escalated, _ = run_agent_escalate(
            spec, llm_inspector=llm, finalize_mc=finalize_mc,
            max_iterations=args.max_iter, workdir=os.path.join(workdir, "run"))

    d = derive(out.final_result, "Hb")
    prov = os.path.join(workdir, "provenance.json")
    out.save(prov)
    if out.final_result.figure_path and os.path.exists(out.final_result.figure_path):
        shutil.copy(out.final_result.figure_path,
                    os.path.join(workdir, "diagnostic.png"))

    result = {
        "name": name, "z": float(spec.z), "status": out.status,
        "quality_flag": out.quality_flag, "escalated_to_llm": bool(escalated),
        "n_iterations": out.n_iterations,
    }
    if d is not None:
        result.update({
            "log_MBH": round(d.log_MBH, 3),
            "log_MBH_err": (round(d.log_MBH_err, 3)
                            if d.log_MBH_err == d.log_MBH_err else None),
            "fwhm_hbeta_kms": round(d.fwhm_hbeta_kms, 1),
            "log_L5100": round(d.log_L5100, 3),
            "log_Lbol": round(d.log_Lbol, 3),
            "eddington_ratio": round(d.eddington_ratio, 4),
        })
    else:
        result["log_MBH"] = None
        result["note"] = "no trustworthy broad-Hbeta mass (non-detection or failure)"

    sci = out.science
    anom = out.anomaly
    if sci is not None:
        result["science"] = sci.to_dict()
        result["flags"] = sci.active_flags
    if anom is not None:
        result["anomaly"] = anom.to_dict()
    with open(os.path.join(workdir, "result.json"), "w") as fh:
        json.dump(result, fh, indent=2)

    # human-readable summary
    from .trust import trust_statement
    print(f"\n=== AGN-Egent: {name}  (z={spec.z:.4f}) ===")
    print(f"  quality flag : {out.quality_flag}"
          + ("  [LLM-reviewed]" if escalated else ""))
    print(f"  trust        : {trust_statement(out.quality_flag)}")
    if d is not None:
        pm = f" +/- {result['log_MBH_err']}" if result["log_MBH_err"] else ""
        print(f"  log M_BH     : {result['log_MBH']}{pm}  [Msun]")
        print(f"  FWHM(Hbeta)  : {result['fwhm_hbeta_kms']:.0f} km/s")
        print(f"  log L5100    : {result['log_L5100']}   log L_bol: {result['log_Lbol']}")
        print(f"  Eddington    : {result['eddington_ratio']}")
    else:
        print("  log M_BH     : -- (no trustworthy broad-Hbeta mass)")
    if sci is not None:
        print(f"  science      : {sci.summary()}")
    if anom is not None:
        print(f"  {anom.summary()}")
    print(f"  outputs      : {workdir}/  (result.json, provenance.json, diagnostic.png)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
