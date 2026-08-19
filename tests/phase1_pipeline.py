"""Phase 1 check: the deterministic pipeline through the new architecture.

Imports ``agn_egent`` FIRST (before numpy) so thread pinning takes effect and
fits are cross-process reproducible.

Verifies:
  1. Spectrum -> decompose -> DecompositionResult works; host/broad/narrow are
     extracted and (broad+narrow) reproduces PyQSOFit's own line model exactly.
  2. Fits are reproducible (same input -> identical FWHM within a process).
  3. Halpha broad is well-constrained and stable.
  4. The default 2-Gaussian broad Hbeta on this weak (SNR~5) line is degenerate
     -> flagged as low quality. Constraining to ngauss=1 (the agent's future
     move) yields a sane, stable measurement. This motivates Phases 2-3.
"""
import os
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)

import agn_egent  # noqa: E402  (import first -> pins threads before numpy)
import numpy as np  # noqa: E402
from agn_egent import load_sdss, decompose, QsoparConfig  # noqa: E402

EXAMPLE_SPEC = os.path.join(
    PROJ, "external", "PyQSOFit", "example", "data", "spec-0332-52367-0639.fits")


def main():
    spec = load_sdss(EXAMPLE_SPEC, name="phase1")
    print("[ok]", spec)

    config = QsoparConfig.sdss_optical_default()

    # Primary science fit (no figure -> clean global state). Render the figure
    # only at the very end, since a plotting fit pollutes later fits' state.
    result = decompose(spec, config=config, make_figure=False,
                       workdir=os.path.join(PROJ, "data", "runs", "phase1"))
    print("\n" + result.summary())

    c = result.components
    assert {"broad", "narrow", "host"} <= set(c), "missing components"

    # 1) reconstruction is exact vs PyQSOFit's own line model
    recon = c["broad"] + c["narrow"]
    ref = c["line_total"]
    rel = np.sum(np.abs(recon - ref)) / np.sum(np.abs(ref))
    print(f"\n[check] |broad+narrow - line_total|/|line_total| = {rel:.2e}")
    assert rel < 1e-3, f"line reconstruction mismatch: {rel:.3e}"

    # 2) reproducibility: a clean repeat fit must be bit-identical.
    #    (Hard guarantee requires launch-time thread pinning + one fit/process,
    #     the batch layout; cross-process proof in scratch/_probe3.py.)
    rep = decompose(spec, config=config, make_figure=False,
                    workdir=os.path.join(PROJ, "data", "runs", "phase1_rep"))
    #    Bit-identity holds with a BLAS whose threading the env vars actually
    #    pin (OpenBLAS/MKL -- every Linux wheel, so CI and the cluster). The
    #    numpy>=2.0 macOS wheels link Accelerate instead, which honors no
    #    thread pinning, so on macOS the degenerate 2-Gaussian broad Hbeta of
    #    this very object (THE knife-edge case, see check 4) can land in
    #    nearby local minima (~4360-4550 km/s) run to run. Demand bit-identity
    #    where the platform can deliver it, and bounded drift where it cannot
    #    -- a silent tolerance everywhere would gut the guarantee this
    #    project actually relies on for its survey runs.
    blas = np.show_config(mode="dicts").get(
        "Build Dependencies", {}).get("blas", {}).get("name", "")
    accelerate = "accelerate" in str(blas).lower()
    assert result.lines["Ha"].fwhm_kms == rep.lines["Ha"].fwhm_kms, "fit not reproducible"
    if accelerate:
        drift = abs(result.lines["Hb"].fwhm_kms - rep.lines["Hb"].fwhm_kms) \
            / rep.lines["Hb"].fwhm_kms
        assert drift < 0.10, f"Hb drift {drift:.1%} beyond the degenerate-minima band"
        print(f"[WARN] Accelerate BLAS (macOS): thread pinning not honored; "
              f"degenerate Hb repeat drift {drift:.2%} (bounded, flagged by QC). "
              f"Bit-reproducibility requires OpenBLAS/MKL (Linux wheels).")
    else:
        assert result.lines["Hb"].fwhm_kms == rep.lines["Hb"].fwhm_kms, "fit not reproducible"
    print(f"[check] reproducible: repeat fit "
          f"(Hb={rep.lines['Hb'].fwhm_kms:.1f}, Ha={rep.lines['Ha'].fwhm_kms:.1f} km/s)")

    # 3) Halpha broad well-constrained
    ha = result.lines["Ha"]
    print(f"[check] Ha broad FWHM = {ha.fwhm_kms:.0f} km/s (SNR {ha.snr:.1f})")
    assert 3000 < ha.fwhm_kms < 7000 and ha.snr > 10, "Halpha unexpectedly bad"

    # 4) default broad Hbeta is degenerate -> low quality; ngauss=1 fixes it
    hb = result.lines["Hb"]
    degenerate = (hb.snr < 10) or (hb.fwhm_kms > 10000)
    print(f"[check] default Hb (ngauss=2): FWHM={hb.fwhm_kms:.0f} km/s "
          f"SNR={hb.snr:.1f} -> {'FLAGGED (needs review)' if degenerate else 'ok'}")
    assert degenerate, "expected the 2-Gaussian broad Hbeta to be flagged here"

    fixed = decompose(spec, config=config.set_ngauss("Hb_br", 1), make_figure=False,
                      workdir=os.path.join(PROJ, "data", "runs", "phase1_hb1"))
    hb1 = fixed.lines["Hb"]
    print(f"[check] constrained Hb (ngauss=1): FWHM={hb1.fwhm_kms:.0f} km/s "
          f"SNR={hb1.snr:.1f}")
    assert 3000 < hb1.fwhm_kms < 8000, "ngauss=1 Hbeta still unreasonable"

    # Render the diagnostic figure last (a plotting fit pollutes global state).
    figrun = decompose(spec, config=config, make_figure=True,
                       workdir=os.path.join(PROJ, "data", "runs", "phase1"))
    print(f"\n[ok] figure: {figrun.figure_path}")
    print("\n[PASS] Phase 1: architecture works, fits reproducible, degeneracy "
          "detected and fixable via a config edit (preview of the agent).")


if __name__ == "__main__":
    main()
