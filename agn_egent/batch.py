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
    error: str = ""

    def flat(self) -> dict:
        """A flat dict suitable for a CSV row."""
        d = {"name": self.name, "status": self.status,
             "n_iterations": self.n_iterations, "verdict": self.verdict,
             "reviewed": self.reviewed,
             "remedies": ";".join(describe_remedy(r) for r in self.remedies),
             "error": self.error}
        for comp, m in self.measurements.items():
            d[f"{comp}_broad_fwhm_kms"] = m.get("fwhm_kms")
            d[f"{comp}_broad_flux"] = m.get("flux")
            d[f"{comp}_broad_snr"] = m.get("snr")
        for k in ("LogL5100", "PL_slope"):
            if k in self.continuum:
                d[k] = self.continuum[k]
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

    def to_json(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump([asdict(r) for r in self.rows], f, indent=2)
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

    def summary(self) -> str:
        ff = self.flag_fractions
        lines = [f"BATCH: {len(self)} object(s)  "
                 f"reviewed-by-agent: {self.n_reviewed}  counts: {self.counts}",
                 f"  quality tiers: clean {ff['clean']:.0%} · "
                 f"reviewed {ff['reviewed']:.0%} · flagged {ff['flagged']:.0%}"]
        for r in self.rows:
            tag = "REVIEWED" if r.reviewed else "triage"
            extra = f"  err={r.error}" if r.error else ""
            lines.append(f"  {r.name:20s} {r.status:14s} [{tag}] "
                         f"iters={r.n_iterations} verdict={r.verdict}{extra}")
        return "\n".join(lines)


def _outcome_to_row(outcome) -> BatchRow:
    res = outcome.final_result
    reviewed = any(s.decision.remedy is not None for s in outcome.steps) or \
        any(s.decision.source not in ("triage", "") for s in outcome.steps)
    remedies = [s.decision.remedy for s in outcome.steps if s.decision.remedy]
    meas = {}
    if res is not None:
        for comp, m in res.lines.items():
            meas[comp] = {"fwhm_kms": m.fwhm_kms, "flux": m.flux, "snr": m.snr}
    return BatchRow(
        name=outcome.name, status=outcome.status,
        n_iterations=outcome.n_iterations,
        verdict=str(outcome.final_verdict.overall) if outcome.final_verdict else "",
        quality_flag=outcome.quality_flag,
        reviewed=reviewed, remedies=remedies, measurements=meas,
        continuum=dict(res.continuum) if res is not None else {})


def _process_one(source, inspector, config, backend, max_iterations, workdir):
    """Worker: load (if a path), run the agent, return a BatchRow. Robust to errors."""
    try:
        spec = source if isinstance(source, Spectrum) else load_sdss(source)
        outcome = run_agent(spec, inspector=inspector, config=config,
                            backend=backend, max_iterations=max_iterations,
                            workdir=os.path.join(workdir, spec.name or "obj"),
                            verbose=False)
        outcome.save(os.path.join(workdir, spec.name or "obj", "provenance.json"))
        return _outcome_to_row(outcome)
    except Exception as e:  # one bad object must not kill the batch
        name = getattr(source, "name", None) or (
            os.path.basename(source) if isinstance(source, str) else "obj")
        return BatchRow(name=name, status="error", error=f"{type(e).__name__}: {e}")


def run_batch(sources,
              inspector=None,
              config: QsoparConfig | None = None,
              backend: str = "pyqsofit",
              max_iterations: int = 4,
              max_workers: int | None = None,
              workdir: str | None = None) -> BatchReport:
    """Decompose + agentically QC a list of spectra.

    Parameters
    ----------
    sources : list of Spectrum or str   spectra, or SDSS FITS paths
    inspector : Inspector               wrapped in TriageInspector unless it
                                        already is one; defaults to triage+rule
    max_workers : int                   object-level parallelism (1 = sequential)
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
    args = lambda s: (s, inspector, config, backend, max_iterations, workdir)

    if max_workers <= 1:
        for s in sources:
            report.rows.append(_process_one(*args(s)))
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(_process_one, *args(s)): s for s in sources}
            for fut in as_completed(futs):
                report.rows.append(fut.result())

    # stable ordering by input order
    order = {(getattr(s, "name", None) or
              (os.path.splitext(os.path.basename(s))[0] if isinstance(s, str) else "")): i
             for i, s in enumerate(sources)}
    report.rows.sort(key=lambda r: order.get(r.name, 1e9))
    return report
