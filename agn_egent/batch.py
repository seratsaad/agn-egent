"""Phase 4: batch decomposition with agentic QC only on the hard cases.

Run a list of spectra (Spectrum objects or SDSS FITS paths) through the agent
loop. By default the inspector is wrapped in TriageInspector, so clean fits are
accepted deterministically and the (expensive) reviewer is consulted only for
WARN/FAIL objects -- the throughput win.

Parallelism is at the object level: one worker process per spectrum, each with
single-threaded BLAS (the reproducibility model). Children inherit the parent's
thread-pinning env vars, so launch the batch thread-pinned (see README).
"""
from __future__ import annotations

import os
import csv
import json
import math
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict

from .spectrum import Spectrum
from .io_sdss import load_sdss
from .pipeline import RUNS_DIR
from .backends.qsopar_config import QsoparConfig
from .agent.inspector import TriageInspector
from .agent.loop import run_agent
from .remedies import describe_remedy


@dataclass
class BatchRow:
    name: str
    status: str                       # accepted | rejected | max_iterations | error
    n_iterations: int = 0
    verdict: str = ""                 # final overall verdict
    quality_flag: str = ""            # clean | reviewed | flagged (Egent-style tier)
    reviewed: bool = False            # did it leave triage (a remedy was tried)?
    remedies: list = field(default_factory=list)
    measurements: dict = field(default_factory=dict)   # per-complex FWHM/flux
    continuum: dict = field(default_factory=dict)       # L5100, PL_slope, ...
    derived: dict = field(default_factory=dict)         # M_BH, L_bol, Eddington
    science: dict = field(default_factory=dict)         # outflow / FeII / profile
    anomaly: dict = field(default_factory=dict)         # residual anomaly score
    z: float | None = None
    ra: float | None = None      # carried from the target catalog, for cross-matching
    dec: float | None = None
    error: str = ""

    @property
    def flags(self) -> list:
        """Active science flags (``nls1``, ``double_peaked``, ...)."""
        return sorted(k for k, v in (self.science.get("flags") or {}).items() if v)

    def flat(self) -> dict:
        """A flat dict suitable for a CSV row / value-added catalog."""
        d = {"name": self.name, "ra": self.ra, "dec": self.dec,
             "z": self.z, "status": self.status,
             "quality_flag": self.quality_flag,
             "n_iterations": self.n_iterations, "verdict": self.verdict,
             "reviewed": self.reviewed,
             "remedies": ";".join(describe_remedy(r) for r in self.remedies),
             "error": self.error}
        for comp, m in self.measurements.items():
            d[f"{comp}_broad_fwhm_kms"] = m.get("fwhm_kms")
            d[f"{comp}_broad_flux"] = m.get("flux")
            d[f"{comp}_broad_snr"] = m.get("snr")
        for k in ("LogL5100", "PL_slope", "frac_host_5100"):
            if k in self.continuum:
                d[k] = self.continuum[k]
        for k in ("log_MBH", "log_MBH_err", "log_Lbol", "eddington_ratio"):
            if k in self.derived:
                d[k] = self.derived[k]
        oiii = self.science.get("oiii") or {}
        for k in ("w80_kms", "v50_kms", "asymmetry"):
            if k in oiii:
                d[f"oiii_{k}"] = oiii[k]
        feii = self.science.get("feii") or {}
        if "r_feii" in feii:
            d["r_feii"] = feii["r_feii"]
        for comp in ("hbeta", "halpha"):
            prof = self.science.get(comp) or {}
            for k in ("asymmetry", "v50_kms", "peak_separation_kms", "peak_contrast"):
                if k in prof:
                    d[f"{comp}_{k}"] = prof[k]
        if self.anomaly:
            d["anomaly_score"] = self.anomaly.get("score")
            d["anomaly_window"] = self.anomaly.get("worst_window")
            d["anomaly_continuum_score"] = self.anomaly.get("continuum_score")
        d["flags"] = ";".join(self.flags)
        return d


@dataclass
class BatchReport:
    rows: list[BatchRow] = field(default_factory=list)

    def __len__(self):
        return len(self.rows)

    @property
    def counts(self) -> dict:
        c = {}
        for r in self.rows:
            c[r.status] = c.get(r.status, 0) + 1
        return c

    @property
    def n_reviewed(self) -> int:
        return sum(1 for r in self.rows if r.reviewed)

    @property
    def flag_fractions(self) -> dict:
        """Egent-style confidence-tier fractions: clean / reviewed / flagged."""
        n = len(self.rows) or 1
        c = {"clean": 0, "reviewed": 0, "flagged": 0}
        for r in self.rows:
            c[r.quality_flag] = c.get(r.quality_flag, 0) + 1
        return {k: round(v / n, 3) for k, v in c.items()}

    @property
    def flag_counts(self) -> dict:
        """How many objects carry each science flag -- the campaign yield table."""
        counts: dict[str, int] = {}
        for r in self.rows:
            for f in r.flags:
                counts[f] = counts.get(f, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    def with_flag(self, flag: str) -> list:
        """Rows carrying a given science flag (e.g. ``double_peaked``)."""
        return [r for r in self.rows if flag in r.flags]

    def rank_by_anomaly(self, limit: int | None = None,
                        min_score: float | None = None) -> list:
        """Rows sorted by anomaly score, worst fit first -- the discovery queue.

        Objects with no score (errors, uncovered windows) are excluded rather
        than sorted to the bottom, so a shortlist never silently contains
        unscored junk.
        """
        scored = []
        for r in self.rows:
            s = (r.anomaly or {}).get("score")
            if s is None or not math.isfinite(s):
                continue
            if min_score is not None and s < min_score:
                continue
            scored.append((s, r))
        scored.sort(key=lambda t: -t[0])
        rows = [r for _, r in scored]
        return rows[:limit] if limit else rows

    def to_json(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump(_json_safe([asdict(r) for r in self.rows]), f, indent=2)
        return path

    def to_csv(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        flats = [r.flat() for r in self.rows]
        cols, seen = [], set()
        for fl in flats:
            for k in fl:
                if k not in seen:
                    seen.add(k); cols.append(k)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for fl in flats:
                w.writerow(fl)
        return path

    def summary(self, per_object: bool = True) -> str:
        ff = self.flag_fractions
        lines = [f"BATCH: {len(self)} object(s)  "
                 f"reviewed-by-agent: {self.n_reviewed}  counts: {self.counts}",
                 f"  quality tiers: clean {ff['clean']:.0%} · "
                 f"reviewed {ff['reviewed']:.0%} · flagged {ff['flagged']:.0%}"]
        fc = self.flag_counts
        if fc:
            lines.append("  science flags: "
                         + ", ".join(f"{k} {v}" for k, v in fc.items()))
        if not per_object:
            return "\n".join(lines)
        for r in self.rows:
            tag = "REVIEWED" if r.reviewed else "triage"
            extra = f"  err={r.error}" if r.error else ""
            lines.append(f"  {r.name:20s} {r.status:14s} [{tag}] "
                         f"iters={r.n_iterations} verdict={r.verdict}{extra}")
        return "\n".join(lines)


def _outcome_to_row(outcome, science: bool = True) -> BatchRow:
    res = outcome.final_result
    reviewed = any(s.decision.remedy is not None for s in outcome.steps) or \
        any(s.decision.source not in ("triage", "") for s in outcome.steps)
    remedies = [s.decision.remedy for s in outcome.steps if s.decision.remedy]
    meas, derived_d, sci, anom = {}, {}, {}, {}
    if res is not None:
        for comp, m in res.lines.items():
            meas[comp] = {"fwhm_kms": m.fwhm_kms, "flux": m.flux, "snr": m.snr}
        from .measure import derive
        d = derive(res, "Hb")
        if d is not None:
            derived_d = {"log_MBH": d.log_MBH, "log_MBH_err": d.log_MBH_err,
                         "log_L5100": d.log_L5100, "log_Lbol": d.log_Lbol,
                         "eddington_ratio": d.eddington_ratio}
        if science:
            sci = outcome.science.to_dict() if outcome.science is not None else {}
            anom = outcome.anomaly.to_dict() if outcome.anomaly is not None else {}
    return BatchRow(
        name=outcome.name, status=outcome.status,
        n_iterations=outcome.n_iterations,
        verdict=str(outcome.final_verdict.overall) if outcome.final_verdict else "",
        quality_flag=outcome.quality_flag,
        reviewed=reviewed, remedies=remedies, measurements=meas,
        continuum=dict(res.continuum) if res is not None else {},
        derived=derived_d, science=sci, anomaly=anom,
        z=float(res.z) if res is not None else None)


def _json_safe(obj):
    """Replace NaN/inf with None so checkpoints are valid JSON for any reader."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def _source_name(source) -> str:
    if isinstance(source, dict):
        return source.get("name") or source.get("sdss") or "obj"
    name = getattr(source, "name", None)
    if name:
        return name
    if isinstance(source, str):
        return os.path.splitext(os.path.basename(source))[0]
    return "obj"


# an SDSS spectrum id, "PLATE-MJD-FIBER"
_SDSS_ID = re.compile(r"^\d{3,5}-\d{5}-\d{1,4}$")


def _load_source(source):
    """Turn a batch source into a Spectrum.

    Accepts a Spectrum, a FITS path, an SDSS "PLATE-MJD-FIBER" id, or a dict
    ``{"sdss": id, "z": catalog_z, "name": ...}``. Ids are downloaded here,
    *inside the worker*: a survey campaign then overlaps its downloads with its
    fits instead of fetching thousands of spectra serially up front (and the
    checkpoint test runs before this, so a resumed campaign re-downloads
    nothing). Downloads are retried -- on a 40-worker node an occasional
    dropped connection must not cost a finished object its slot.

    When the dict form supplies a catalog redshift, it overrides the SDSS
    header z. The campaign *selected* on the catalog z, and the two disagree
    exactly for pipeline-z failures (seen: header z=6.1 on a Shen z~0.4
    quasar, which silently pushes every optical line out of coverage).
    """
    if isinstance(source, Spectrum):
        return source
    z_override = None
    if isinstance(source, dict):
        z_override = source.get("z")
        name = source.get("name") or source.get("sdss")
        source = source.get("sdss") or source.get("path")
    else:
        name = source if isinstance(source, str) else None
    if isinstance(source, str) and _SDSS_ID.match(source):
        from .catalog import fetch_sdss_spectrum
        plate, mjd, fiber = (int(x) for x in source.split("-"))
        spec, last = None, None
        for attempt in range(3):
            try:
                spec = fetch_sdss_spectrum(plate, mjd, fiber, name=name or source)
                break
            except Exception as e:      # transient network/service hiccups
                last = e
                time.sleep(2.0 * (attempt + 1))
        if spec is None:
            raise last
    else:
        spec = load_sdss(source)
    if z_override is not None and math.isfinite(z_override) and z_override > -0.01:
        spec.z = float(z_override)
    return spec


def _process_one(source, inspector, config, backend, max_iterations, workdir,
                 science=True, resume=True):
    """Worker: load (if a path), run the agent, return a BatchRow.

    Robust to errors -- one bad object must never kill a campaign -- and
    checkpointed: a finished object writes ``row.json`` into its own directory
    and is skipped on a later run, so an interrupted survey resumes where it
    stopped instead of re-fitting everything.
    """
    objdir = os.path.join(workdir, _source_name(source))
    ckpt = os.path.join(objdir, "row.json")
    if resume and os.path.exists(ckpt):
        try:
            with open(ckpt) as fh:
                row = BatchRow(**json.load(fh))
            # error rows are NOT honored as checkpoints: a transient download
            # failure (or a since-fixed bug) would otherwise be frozen into the
            # catalog forever. Resuming retries exactly the failed objects.
            if row.status != "error":
                return row
        except Exception:
            pass          # unreadable checkpoint: just redo the object
    try:
        spec = _load_source(source)
        objdir = os.path.join(workdir, spec.name or "obj")
        outcome = run_agent(spec, inspector=inspector, config=config,
                            backend=backend, max_iterations=max_iterations,
                            workdir=objdir, verbose=False)
        outcome.save(os.path.join(objdir, "provenance.json"))
        row = _outcome_to_row(outcome, science=science)
    except Exception as e:  # one bad object must not kill the batch
        row = BatchRow(name=_source_name(source), status="error",
                       error=f"{type(e).__name__}: {e}")
    try:
        os.makedirs(objdir, exist_ok=True)
        with open(os.path.join(objdir, "row.json"), "w") as fh:
            json.dump(_json_safe(asdict(row)), fh)
    except OSError:
        pass              # a failed checkpoint write must not fail the object
    return row


def run_batch(sources,
              inspector=None,
              config: QsoparConfig | None = None,
              backend: str = "pyqsofit",
              max_iterations: int = 4,
              max_workers: int | None = None,
              workdir: str | None = None,
              science: bool = True,
              resume: bool = True,
              progress: bool = False) -> BatchReport:
    """Decompose + agentically QC a list of spectra.

    Parameters
    ----------
    sources : list of Spectrum or str   spectra, or SDSS FITS paths
    inspector : Inspector               wrapped in TriageInspector unless it
                                        already is one; defaults to triage+rule
    max_workers : int                   object-level parallelism (1 = sequential)
    science : bool                      also compute the line-shape science and
                                        anomaly score for every object (the
                                        discovery-campaign columns)
    resume : bool                       skip objects that already have a
                                        checkpoint in `workdir` -- lets an
                                        interrupted survey pick up where it left
    progress : bool                     print a running count to stdout
    """
    if inspector is None:
        inspector = TriageInspector()
    elif not isinstance(inspector, TriageInspector):
        inspector = TriageInspector(inspector)
    workdir = workdir or os.path.join(RUNS_DIR, "batch")
    os.makedirs(workdir, exist_ok=True)
    sources = list(sources)
    if max_workers is None:
        max_workers = min(len(sources), os.cpu_count() or 1)

    report = BatchReport()
    total = len(sources)

    def args(s):
        return (s, inspector, config, backend, max_iterations, workdir,
                science, resume)

    def note(row):
        if progress:
            done = len(report.rows)
            print(f"[{done}/{total}] {row.name}: {row.status} "
                  f"({row.quality_flag or '-'})", flush=True)

    if max_workers <= 1:
        for s in sources:
            report.rows.append(_process_one(*args(s)))
            note(report.rows[-1])
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(_process_one, *args(s)): s for s in sources}
            for fut in as_completed(futs):
                report.rows.append(fut.result())
                note(report.rows[-1])

    # stable ordering by input order
    order = {_source_name(s): i for i, s in enumerate(sources)}
    report.rows.sort(key=lambda r: order.get(r.name, 1e9))
    return report
