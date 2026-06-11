#!/usr/bin/env python
"""AGN-Egent command-line interface.

Decompose an AGN spectrum into broad + narrow line components, run the agentic
QC loop, and report the line measurements + single-epoch black-hole mass.

Examples
--------
  # an SDSS FITS file on disk
  python run_agn.py spectra/spec-0332-52367-0639.fits

  # download an SDSS quasar by plate-mjd-fiber and use the LLM reviewer
  python run_agn.py 388-51793-445 --inspector claude

  # a non-SDSS spectrum (wavelength/flux/err rows) needs a redshift
  python run_agn.py j0950_hbeta.fits --z 0.2144 --flux-scale 1e17

  # batch: a text file with one target (path or plate-mjd-fiber) per line
  python run_agn.py --list targets.txt --workers 4
"""
import argparse
import os
import re
import sys

import config
config.pin_threads()                       # MUST precede the numpy import below

import agn_egent  # noqa: E402
from agn_egent import (load_sdss, load_row_fits, fetch_sdss_spectrum,  # noqa: E402
                       QsoparConfig, run_agent, run_batch,
                       make_inspector, TriageInspector,
                       derive, render_diagnostic)

PMF_RE = re.compile(r"^\d{3,5}-\d{4,5}-\d{1,4}$")
_KEY_ENV = {"claude": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}


def build_inspector(args):
    if args.inspector in _KEY_ENV:
        key = args.api_key or os.environ.get(_KEY_ENV[args.inspector])
        if not key:
            sys.exit(f"error: --inspector {args.inspector} needs --api-key or "
                     f"{_KEY_ENV[args.inspector]} in the environment")
        return make_inspector(args.inspector, api_key=key, model=args.model)
    return make_inspector("rule")


def resolve_target(target: str, z=None, flux_scale: float = 1.0):
    """Turn a CLI target (FITS path or PLATE-MJD-FIBER) into a Spectrum."""
    if PMF_RE.match(target):
        p, m, f = (int(x) for x in target.split("-"))
        return fetch_sdss_spectrum(p, m, f)
    if not os.path.exists(target):
        sys.exit(f"error: target not found: {target}")
    try:
        return load_sdss(target)            # standard SDSS spec FITS
    except Exception:
        if z is None:
            sys.exit(f"error: {target} is not an SDSS spec file; pass --z for a "
                     "generic wavelength/flux/err FITS")
        return load_row_fits(target, z=z, flux_scale=flux_scale)


def build_config(args) -> QsoparConfig:
    cfg = QsoparConfig.sdss_optical_default()
    if args.ngauss_hb is not None:
        cfg = cfg.set_ngauss("Hb_br", args.ngauss_hb)
    return cfg


def run_single(target, args):
    spec = resolve_target(target, z=args.z, flux_scale=args.flux_scale)
    out_dir = os.path.join(args.out, spec.name)
    outcome = run_agent(spec, inspector=build_inspector(args),
                        config=build_config(args), max_iterations=args.max_iter,
                        finalize_mc=not args.no_mc, nsamp=args.nsamp,
                        workdir=out_dir, verbose=True)
    res = outcome.final_result

    print("\n" + outcome.summary())
    print("\n" + res.summary())
    dq = derive(res, "Hb")
    if dq is not None:
        print("\nDerived: " + str(dq))

    fig = render_diagnostic(res, os.path.join(out_dir, "diagnostic.png"))
    prov = outcome.save(os.path.join(out_dir, "provenance.json"))
    print(f"\nfigure     -> {fig}")
    print(f"provenance -> {prov}")
    return outcome


def run_list(path, args):
    with open(path) as f:
        targets = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    print(f"resolving {len(targets)} target(s)...")
    specs = [resolve_target(t, z=args.z, flux_scale=args.flux_scale) for t in targets]
    inspector = TriageInspector(build_inspector(args))
    report = run_batch(specs, inspector=inspector, config=build_config(args),
                       max_iterations=args.max_iter, max_workers=args.workers,
                       workdir=os.path.join(args.out, "batch"))
    print("\n" + report.summary())
    csv = report.to_csv(os.path.join(args.out, "batch", "results.csv"))
    report.to_json(os.path.join(args.out, "batch", "results.json"))
    print(f"\nresults table -> {csv}")
    return report


def main():
    ap = argparse.ArgumentParser(description="AGN-Egent: agentic AGN spectral decomposition")
    ap.add_argument("target", nargs="?",
                    help="SDSS spec FITS path, or PLATE-MJD-FIBER to download")
    ap.add_argument("--list", help="text file of targets (one per line) for batch mode")
    ap.add_argument("--inspector", choices=["rule", "claude", "openai"], default="rule",
                    help="QC reviewer: deterministic (rule) or vision LLM (claude / openai)")
    ap.add_argument("--api-key", default=None,
                    help="LLM API key (else read ANTHROPIC_API_KEY / OPENAI_API_KEY)")
    ap.add_argument("--model", default=None,
                    help="LLM model override (e.g. gpt-4o, claude-opus-4-8)")
    ap.add_argument("--no-mc", action="store_true",
                    help="skip Monte-Carlo uncertainties (faster, no error bars)")
    ap.add_argument("--nsamp", type=int, default=config.DEFAULT_NSAMP,
                    help="Monte-Carlo resamples for error bars")
    ap.add_argument("--max-iter", type=int, default=config.DEFAULT_MAX_ITERATIONS,
                    help="agent QC loop iteration cap")
    ap.add_argument("--ngauss-hb", type=int, default=None,
                    help="number of broad Hbeta Gaussians (default from config)")
    ap.add_argument("--z", type=float, default=None,
                    help="redshift, required for non-SDSS (generic) FITS")
    ap.add_argument("--flux-scale", type=float, default=1.0,
                    help="flux rescale for generic FITS (e.g. 1e17 for cgs)")
    ap.add_argument("--workers", type=int, default=None,
                    help="parallel processes for batch mode")
    ap.add_argument("--out", default=config.RUNS_DIR, help="output directory")
    args = ap.parse_args()

    if args.list:
        run_list(args.list, args)
    elif args.target:
        run_single(args.target, args)
    else:
        ap.error("provide a target, or --list FILE for batch mode")


if __name__ == "__main__":
    main()
