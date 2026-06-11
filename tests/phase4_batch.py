"""Phase 4 check: batch decomposition with agentic QC only on the hard cases.

Runs the three bundled SDSS spectra through run_batch with object-level
parallelism and a triage+rule inspector. Asserts every object is processed
(no crashes), the known-degenerate object (spec-0332) is reviewed and gets the
set_ngauss remedy, throughput accounting holds (reviewed <= total), and a CSV +
JSON results table are written.

Run thread-pinned (see README) for reproducibility; child workers inherit the
pinning env vars.
"""
import os
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)

import agn_egent  # noqa: E402  (import first -> pin threads before numpy)
from agn_egent import run_batch, RuleInspector, TriageInspector  # noqa: E402

DATA = os.path.join(PROJ, "external", "PyQSOFit", "example", "data")
SPECTRA = [
    os.path.join(DATA, "spec-0266-51602-0013.fits"),
    os.path.join(DATA, "spec-0266-51602-0107.fits"),
    os.path.join(DATA, "spec-0332-52367-0639.fits"),  # known degenerate broad Hb
]
OUT = os.path.join(PROJ, "data", "runs", "batch_phase4")


def main():
    report = run_batch(SPECTRA, inspector=TriageInspector(RuleInspector()),
                       max_iterations=4, max_workers=2, workdir=OUT)
    print(report.summary())

    assert len(report) == 3, "not all objects processed"
    errs = [r for r in report.rows if r.status == "error"]
    assert not errs, f"objects errored: {[(r.name, r.error) for r in errs]}"
    assert all(r.status in ("accepted", "rejected", "max_iterations")
               for r in report.rows)

    # throughput accounting: the reviewer touches at most every object
    assert 0 <= report.n_reviewed <= len(report)
    print(f"\n[check] agent reviewed {report.n_reviewed}/{len(report)} objects "
          f"(the rest were triaged through deterministically)")

    # the known-degenerate object must be reviewed and get the ngauss remedy
    hb = next((r for r in report.rows if r.name == "0332-52367-0639"), None)
    assert hb is not None, "spec-0332 row missing"
    assert hb.reviewed, "the degenerate object should have been reviewed"
    assert any(rem.get("action") == "set_ngauss" for rem in hb.remedies), \
        f"expected a set_ngauss remedy on spec-0332, got {hb.remedies}"
    print(f"[check] spec-0332 reviewed -> {hb.status}, "
          f"remedies={[r for r in hb.remedies]}")

    # measurements populated for every accepted object
    for r in report.rows:
        if r.status == "accepted":
            assert r.measurements, f"{r.name} has no measurements"

    csv_path = report.to_csv(os.path.join(OUT, "results.csv"))
    json_path = report.to_json(os.path.join(OUT, "results.json"))
    assert os.path.getsize(csv_path) > 0 and os.path.getsize(json_path) > 0
    print(f"\n[ok] results table: {csv_path}")
    print(f"[ok] results table: {json_path}")
    print("\n[PASS] Phase 4: batch runs, triages clean fits, reviews hard cases, "
          "and writes an auditable results table.")


if __name__ == "__main__":
    main()
