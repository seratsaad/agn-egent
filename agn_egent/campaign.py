"""Survey campaigns: run the pipeline over many objects and rank what comes out.

This is the discovery loop. A campaign is:

    select targets -> fit + QC every one -> measure science -> score anomalies
    -> shortlist by class -> (optionally) cross-match the shortlist -> report

Everything runs on the deterministic rule inspector by default, so a campaign
costs CPU time and nothing else. The expensive attention -- a vision LLM, or a
person -- is spent only on the shortlist the ranking produces, which is the
whole point: the engine reduces 10^5 spectra to 10^2 worth looking at.

Candidate classes are defined in :data:`CANDIDATE_CLASSES`. Each is a
(predicate, sort key) pair over :class:`~agn_egent.batch.BatchRow`, so adding a
new search means adding one entry, not a new pipeline.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, asdict

from .batch import run_batch, BatchReport

# Quality tiers a candidate must be in before it is worth a human's time. A
# "flagged" fit means the engine itself does not trust the measurement, so an
# unusual line shape derived from it is far more likely to be a bad fit than a
# discovery. Anomaly candidates are the deliberate exception -- there the point
# *is* that the fit failed.
TRUSTED_TIERS = ("clean", "reviewed")


@dataclass
class Target:
    """One object to observe-and-fit, with enough metadata to cross-match it."""
    name: str
    plate: int | None = None
    mjd: int | None = None
    fiber: int | None = None
    ra: float | None = None
    dec: float | None = None
    z: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _f(x, default=None):
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def select_sdss_targets(z_min: float = 0.05, z_max: float = 0.75,
                        sn_min: float = 5.0, sn_max: float | None = None,
                        fwhm_min: float = 1000.0, n: int = 100) -> list:
    """Pick campaign targets from the Shen (2011) DR7 quasar catalog via VizieR.

    The redshift window defaults to the range where Hbeta and [O III] are inside
    the SDSS optical coverage, which is what the optical engine is validated for.
    Requires network. Returns :class:`Target` records carrying coordinates, so
    the shortlist can be cross-matched later without a second catalog query.

    Note the selection function this imposes: Shen's catalog requires a detected
    broad line, so a campaign built on it cannot discover objects that have no
    broad line at all (true Type 2s, or the "off" state of a changing-look AGN).
    """
    from astroquery.vizier import Vizier
    from .catalog import SHEN_CATALOG

    sn_filter = f">{sn_min}" + (f" && <{sn_max}" if sn_max is not None else "")
    V = Vizier(columns=["SDSS", "RAJ2000", "DEJ2000", "Plate", "MJD", "Fiber",
                        "z", "SN(Hb)", "W(BHb)"],
               column_filters={"z": f">{z_min} && <{z_max}",
                               "SN(Hb)": sn_filter,
                               "W(BHb)": f">{fwhm_min}"})
    V.ROW_LIMIT = n
    res = V.get_catalogs(SHEN_CATALOG)
    if not res:
        return []
    out = []
    for r in res[0]:
        plate, mjd, fiber = int(r["Plate"]), int(r["MJD"]), int(r["Fiber"])
        out.append(Target(name=f"{plate}-{mjd}-{fiber}",
                          plate=plate, mjd=mjd, fiber=fiber,
                          ra=_f(r["RAJ2000"]), dec=_f(r["DEJ2000"]),
                          z=_f(r["z"])))
    return out


def fetch_targets(targets, verbose: bool = False) -> list:
    """Download the SDSS spectra for a target list, skipping ones that fail."""
    from .catalog import fetch_sdss_spectrum
    specs = []
    for t in targets:
        try:
            specs.append(fetch_sdss_spectrum(t.plate, t.mjd, t.fiber, name=t.name))
        except Exception as e:
            if verbose:
                print(f"  skip {t.name}: {type(e).__name__}: {e}", flush=True)
    return specs


# ---------------------------------------------------------------------------
# candidate classes
# ---------------------------------------------------------------------------

def _sci(row, *path, default=None):
    d = row.science or {}
    for p in path:
        if not isinstance(d, dict):
            return default
        d = d.get(p)
    return default if d is None else d


def _anom(row):
    s = (row.anomaly or {}).get("score")
    return s if (s is not None and math.isfinite(s)) else float("-inf")


def _has(flag):
    return lambda r: flag in r.flags


#: name -> (predicate, sort key, one-line description of the search)
CANDIDATE_CLASSES = {
    "double_peaked": (
        _has("double_peaked"),
        lambda r: -max(_sci(r, "hbeta", "data_peak_contrast", default=0) or 0,
                       _sci(r, "halpha", "data_peak_contrast", default=0) or 0),
        "disk emitters: two resolved peaks in a broad line (feeds disk modelling)"),
    "extreme_outflow": (
        _has("extreme_outflow"),
        lambda r: -(_sci(r, "oiii", "w80_kms", default=0) or 0),
        "[O III] W80 > 1000 km/s or centroid blueshift > 300 km/s"),
    "nls1": (
        _has("nls1"),
        lambda r: (_sci(r, "hbeta", "fwhm_kms", default=1e9) or 1e9),
        "narrow-line Seyfert 1: FWHM(Hb) < 2000 km/s, weak [O III]"),
    "strong_feii": (
        _has("strong_feii"),
        lambda r: -(_sci(r, "feii", "r_feii", default=0) or 0),
        "Eigenvector-1 extreme: R_FeII > 1"),
    "type1_9": (
        _has("type1_9"),
        lambda r: -(_sci(r, "halpha", "snr", default=0) or 0),
        "broad Halpha but no broad Hbeta (reddened or in transition)"),
    "offset_broad_line": (
        _has("offset_broad_line"),
        lambda r: -abs(_sci(r, "hbeta", "v50_kms", default=0) or 0),
        "broad-line centroid offset > 1000 km/s (recoil / binary candidates)"),
    # Ranked, not thresholded. Real spectra always leave some coherent residual
    # (imperfect Fe II, host, continuum), so where the population sits depends on
    # the sample; an absolute cut would either pass everything or nothing. Taking
    # the top of the ranking asks the well-posed question -- "which objects in
    # *this* campaign are least well described?" -- and the score is still
    # reported per object for anyone who wants an absolute reading.
    "anomaly": (
        lambda r: math.isfinite(_anom(r)),
        lambda r: -_anom(r),
        "worst coherent residual in the sample -- unclassified oddities"),
}


def shortlist(report: BatchReport, classes=None, limit: int = 25,
              trusted_only: bool = True) -> dict:
    """Rank the campaign's objects into candidate lists, one per class.

    Every class except ``anomaly`` is restricted to fits the engine trusts,
    because an unusual measurement from an untrusted fit is almost always a bad
    fit. The anomaly class keeps everything on purpose: a failed fit is exactly
    its selection criterion.
    """
    classes = classes or list(CANDIDATE_CLASSES)
    out = {}
    for cname in classes:
        pred, key, _desc = CANDIDATE_CLASSES[cname]
        rows = [r for r in report.rows if r.status != "error" and pred(r)]
        if trusted_only and cname != "anomaly":
            rows = [r for r in rows if r.quality_flag in TRUSTED_TIERS]
        rows.sort(key=key)
        out[cname] = rows[:limit]
    return out


def yield_table(report: BatchReport, short: dict) -> str:
    """Human-readable summary of what the campaign found."""
    n = len(report.rows)
    ok = sum(1 for r in report.rows if r.status != "error")
    lines = [f"CAMPAIGN: {n} target(s), {ok} fit, {n - ok} failed",
             report.summary(per_object=False).split("\n")[1]]
    lines.append("")
    lines.append(f"  {'class':<20s} {'n':>5s}   description")
    for cname, rows in short.items():
        desc = CANDIDATE_CLASSES[cname][2]
        lines.append(f"  {cname:<20s} {len(rows):>5d}   {desc}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# running a campaign
# ---------------------------------------------------------------------------

def run_campaign(targets, outdir: str, inspector=None, max_workers: int | None = None,
                 max_iterations: int = 4, limit: int = 25,
                 resume: bool = True, verbose: bool = True) -> dict:
    """Fetch, fit, score and shortlist a list of targets.

    Writes into ``outdir``:

    ``catalog.csv`` / ``catalog.json``  every object with all measured columns
    ``shortlist.json``                  the ranked candidate lists
    ``targets.json``                    the input selection (for reproducibility)
    ``runs/<name>/``                    per-object provenance, diagnostic, checkpoint

    Returns ``{"report":BatchReport, "shortlist":dict, "targets":list}``.
    """
    os.makedirs(outdir, exist_ok=True)
    runs = os.path.join(outdir, "runs")

    with open(os.path.join(outdir, "targets.json"), "w") as fh:
        json.dump([t.to_dict() for t in targets], fh, indent=2)

    if verbose:
        print(f"fetching {len(targets)} spectra ...", flush=True)
    specs = fetch_targets(targets, verbose=verbose)
    if verbose:
        print(f"  got {len(specs)}/{len(targets)}", flush=True)

    report = run_batch(specs, inspector=inspector, max_workers=max_workers,
                       max_iterations=max_iterations, workdir=runs,
                       science=True, resume=resume, progress=verbose)

    # carry the catalog coordinates onto the rows so the shortlist is matchable
    by_name = {t.name: t for t in targets}
    for r in report.rows:
        t = by_name.get(r.name)
        if t is not None:
            r.ra, r.dec = t.ra, t.dec
            if r.z is None:
                r.z = t.z

    report.to_csv(os.path.join(outdir, "catalog.csv"))
    report.to_json(os.path.join(outdir, "catalog.json"))

    short = shortlist(report, limit=limit)
    with open(os.path.join(outdir, "shortlist.json"), "w") as fh:
        json.dump({k: [r.name for r in v] for k, v in short.items()}, fh, indent=2)

    if verbose:
        print()
        print(yield_table(report, short))
    return {"report": report, "shortlist": short, "targets": targets}


def final_diagnostic(outdir: str, name: str) -> str | None:
    """Path to the diagnostic image of the *accepted* fit, relative to `outdir`.

    The loop writes one figure per iteration; the accepted fit is the last one,
    so prefer the figure the provenance records and fall back to the
    highest-numbered iteration directory.
    """
    rundir = os.path.join(outdir, "runs", name)
    prov = os.path.join(rundir, "provenance.json")
    if os.path.exists(prov):
        try:
            with open(prov) as fh:
                steps = json.load(fh).get("steps") or []
            for s in reversed(steps):
                p = s.get("figure_path")
                if p and os.path.exists(p):
                    return os.path.relpath(p, outdir)
        except Exception:
            pass
    if not os.path.isdir(rundir):
        return None
    iters = sorted((d for d in os.listdir(rundir) if d.startswith("iter")),
                   key=lambda d: int(d[4:]) if d[4:].isdigit() else -1)
    for d in reversed(iters):
        p = os.path.join(rundir, d, "diagnostic.png")
        if os.path.exists(p):
            return os.path.relpath(p, outdir)
    return None


def _fmt(v, spec=".2f"):
    if v is None:
        return "&mdash;"
    try:
        if not math.isfinite(float(v)):
            return "&mdash;"
        return format(float(v), spec)
    except (TypeError, ValueError):
        return str(v)


def write_gallery(short: dict, outdir: str, novelty: dict | None = None,
                  filename: str = "gallery.html") -> str:
    """Render the shortlist as a single browsable page of diagnostics.

    Visual vetting is the step that turns a ranked list into real candidates,
    and it needs the fit picture next to the numbers. One self-contained HTML
    file referencing the per-object diagnostics already on disk.
    """
    novelty = novelty or {}
    css = """
    body{font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;
         background:#fbfbfd;color:#1a1a1a}
    header{padding:20px 28px;background:#fff;border-bottom:1px solid #e3e3e8}
    h1{margin:0;font-size:20px} h2{margin:28px 28px 8px;font-size:16px}
    .desc{margin:0 28px 12px;color:#666;font-size:13px}
    .card{display:flex;gap:16px;align-items:flex-start;background:#fff;
          margin:10px 28px;padding:12px;border:1px solid #e3e3e8;border-radius:8px}
    .card img{width:520px;max-width:52vw;border-radius:4px}
    .meta{font-size:13px} .meta b{font-family:ui-monospace,monospace}
    .k{color:#666;display:inline-block;min-width:120px}
    .flag{display:inline-block;background:#eef3ff;color:#24408e;border-radius:10px;
          padding:1px 8px;margin:2px 3px 0 0;font-size:11px}
    .known{color:#8a6d00} .new{color:#0a7a3f;font-weight:600}
    nav a{margin-right:14px}
    """
    parts = [f"<!doctype html><meta charset='utf-8'><title>AGN-Egent candidates</title>"
             f"<style>{css}</style>",
             "<header><h1>AGN-Egent &mdash; candidate gallery</h1><nav>"]
    parts += [f"<a href='#{c}'>{c} ({len(r)})</a>" for c, r in short.items() if r]
    parts.append("</nav></header>")

    for cname, rows in short.items():
        if not rows:
            continue
        desc = CANDIDATE_CLASSES[cname][2]
        parts.append(f"<h2 id='{cname}'>{cname} &mdash; {len(rows)}</h2>"
                     f"<div class='desc'>{desc}</div>")
        for r in rows:
            img = final_diagnostic(outdir, r.name)
            imgtag = (f"<img src='{img}' alt='{r.name}' loading='lazy'>" if img
                      else "<div class='meta'>(no diagnostic image)</div>")
            oiii = (r.science or {}).get("oiii") or {}
            hb = (r.science or {}).get("hbeta") or {}
            fe = (r.science or {}).get("feii") or {}
            nov = novelty.get(r.name)
            if nov is None:
                novhtml = ""
            elif nov.matched:
                novhtml = (f"<div class='known'>SIMBAD: {nov.main_id} "
                           f"[{nov.otype}]</div>")
            else:
                novhtml = "<div class='new'>no SIMBAD counterpart</div>"
            parts.append(
                "<div class='card'>" + imgtag + "<div class='meta'>"
                f"<b>{r.name}</b> &nbsp; z={_fmt(r.z, '.4f')} &nbsp; "
                f"[{r.quality_flag}]<br>"
                f"<span class='k'>log M_BH</span> {_fmt((r.derived or {}).get('log_MBH'))}<br>"
                f"<span class='k'>FWHM(Hb)</span> {_fmt(hb.get('fwhm_kms'), '.0f')} km/s<br>"
                f"<span class='k'>broad asym</span> {_fmt(hb.get('asymmetry'), '+.2f')}<br>"
                f"<span class='k'>peak sep</span> {_fmt(hb.get('peak_separation_kms'), '.0f')} km/s"
                f" (contrast {_fmt(hb.get('peak_contrast'))})<br>"
                f"<span class='k'>[OIII] W80</span> {_fmt(oiii.get('w80_kms'), '.0f')} km/s"
                f" &nbsp; v50 {_fmt(oiii.get('v50_kms'), '+.0f')}<br>"
                f"<span class='k'>R_FeII</span> {_fmt(fe.get('r_feii'))}<br>"
                f"<span class='k'>anomaly</span> {_fmt((r.anomaly or {}).get('score'))}"
                f" ({(r.anomaly or {}).get('worst_window') or '&mdash;'})<br>"
                + "".join(f"<span class='flag'>{f}</span>" for f in r.flags)
                + novhtml + "</div></div>")

    path = os.path.join(outdir, filename)
    with open(path, "w") as fh:
        fh.write("\n".join(parts))
    return path


def vet_shortlist(short: dict, outdir: str | None = None,
                  verbose: bool = True) -> dict:
    """Cross-match every shortlisted candidate against SIMBAD for context.

    Deduplicates first: an object can appear in several candidate classes and
    should only be looked up once. Returns ``{name: NoveltyRecord}``.
    """
    from .novelty import annotate

    seen, uniq = set(), []
    for rows in short.values():
        for r in rows:
            if r.name in seen or r.ra is None or r.dec is None:
                continue
            seen.add(r.name)
            uniq.append({"name": r.name, "ra": r.ra, "dec": r.dec})
    if verbose:
        print(f"cross-matching {len(uniq)} unique candidate(s) against SIMBAD ...",
              flush=True)
    recs = annotate(uniq, verbose=verbose)
    out = {r.name: r for r in recs}
    if outdir:
        with open(os.path.join(outdir, "novelty.json"), "w") as fh:
            json.dump({k: v.to_dict() for k, v in out.items()}, fh, indent=2)
    return out
