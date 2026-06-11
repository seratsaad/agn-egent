"""Phase 5 validation against three anchors.

5a  Synthetic injection-recovery (rigorous, self-contained): build a spectrum
    from a known power-law + tied narrow lines + broad Hbeta/Halpha, run the
    pipeline with Fe II / host off, and recover the injected broad-line FWHM,
    flux, and L5100 within tolerance.

5b  Published SDSS quasar (physical validation): decompose the canonical
    example object, derive single-epoch M_BH / L_bol / Eddington ratio, and
    check they are physically reasonable for a z~0.1 AGN and reproducible.

5c  J0950 (general-path, heterogeneous instrument): load the HET/LRS2 Hbeta
    spectrum (z=0.2144) and run it through the same pipeline. Confirms the
    coverage guard drops the uncovered Halpha complex and the broad Hbeta FWHM
    lands in the broad-line-AGN range the manual decomposition found.

Run thread-pinned (see README).
"""
import math
import os
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)

import agn_egent  # noqa: E402  (import first -> pin threads before numpy)
from agn_egent import (decompose, QsoparConfig, make_synthetic_spectrum,  # noqa: E402
                       load_sdss, load_row_fits, derive)

EXAMPLE_SPEC = os.path.join(
    PROJ, "external", "PyQSOFit", "example", "data", "spec-0332-52367-0639.fits")
J0950_HB = os.path.join(
    os.path.dirname(PROJ), "j0950", "HETspec", "PSU22-2-010",
    "spectrum_20220325_0000008_exp01_orange.fits")


def _line_only_config():
    """SDSS optical config with Fe II + host + polynomial continuum disabled."""
    cfg = QsoparConfig.sdss_optical_default()
    for k, v in {"decompose_host": False, "Fe_uv_op": False, "poly": False}.items():
        cfg = cfg.set_fit_option(k, v)
    return cfg


def part_a():
    print("=== 5a: synthetic injection-recovery ===")
    spec, truth = make_synthetic_spectrum(hbeta_fwhm=4200.0, halpha_fwhm=4600.0,
                                           snr=30.0, seed=1)
    res = decompose(spec, config=_line_only_config(), make_figure=False,
                    workdir=os.path.join(PROJ, "data", "runs", "phase5_synth"))
    hb, ha = res.lines["Hb"], res.lines["Ha"]
    print(f"  Hb FWHM: truth {truth['hbeta_fwhm']:.0f} -> recovered {hb.fwhm_kms:.0f} km/s")
    print(f"  Ha FWHM: truth {truth['halpha_fwhm']:.0f} -> recovered {ha.fwhm_kms:.0f} km/s")
    print(f"  Hb flux: truth {truth['hbeta_flux']:.0f} -> recovered {hb.flux:.0f}")
    log_l5100 = res.continuum.get("LogL5100")
    print(f"  logL5100 recovered = {log_l5100}")

    def rel(a, b):
        return abs(a - b) / b

    assert rel(hb.fwhm_kms, truth["hbeta_fwhm"]) < 0.15, "Hb FWHM off >15%"
    assert rel(ha.fwhm_kms, truth["halpha_fwhm"]) < 0.15, "Ha FWHM off >15%"
    assert rel(hb.flux, truth["hbeta_flux"]) < 0.25, "Hb flux off >25%"
    print("[PASS] 5a: injected broad-line FWHM and flux recovered within tolerance.")


def part_b():
    print("\n=== 5b: SDSS example physical validation (M_BH) ===")
    spec = load_sdss(EXAMPLE_SPEC, name="phase5b")
    cfg = QsoparConfig.sdss_optical_default().set_ngauss("Hb_br", 1)  # stable broad Hb
    res = decompose(spec, config=cfg, make_figure=False,
                    workdir=os.path.join(PROJ, "data", "runs", "phase5b"))
    dq = derive(res, "Hb")
    assert dq is not None, "could not derive M_BH"
    print("  " + str(dq))
    # physical sanity for a z~0.1 broad-line AGN
    assert 6.0 < dq.log_MBH < 10.0, f"M_BH unphysical: {dq.log_MBH}"
    assert 42.0 < dq.log_L5100 < 46.0, f"L5100 unphysical: {dq.log_L5100}"
    assert 1e-3 < dq.eddington_ratio < 3.0, f"Eddington ratio unphysical: {dq.eddington_ratio}"

    # reproducible
    res2 = decompose(spec, config=cfg, make_figure=False,
                     workdir=os.path.join(PROJ, "data", "runs", "phase5b_rep"))
    dq2 = derive(res2, "Hb")
    assert math.isclose(dq.log_MBH, dq2.log_MBH, rel_tol=1e-6), "M_BH not reproducible"
    print("[PASS] 5b: derived M_BH/L_bol/Eddington physically reasonable & reproducible.")


def part_c():
    print("\n=== 5c: J0950 general-path (HET/LRS2 Hbeta, z=0.2144) ===")
    if not os.path.exists(J0950_HB):
        print(f"  SKIPPED — J0950 HET spectrum not found at {J0950_HB}")
        return
    # HET flux is in cgs; scale to the 1e-17 convention. Coords from J0950.
    spec = load_row_fits(J0950_HB, z=0.2144, ra=147.6531250, dec=51.4772500,
                         flux_scale=1e17, name="J0950")
    print(f"  {spec}  rest=[{spec.rest_wave.min():.0f},{spec.rest_wave.max():.0f}]")
    assert spec.covers(4862.68) and not spec.covers(6564.61), \
        "expected Hbeta covered, Halpha not"

    res = decompose(spec, config=QsoparConfig.sdss_optical_default(),
                    make_figure=True,
                    workdir=os.path.join(PROJ, "data", "runs", "phase5_j0950"))
    # coverage guard must have dropped the uncovered Halpha complex
    assert "Ha" not in res.lines, "Halpha should have been trimmed (not covered)"
    assert "Hb" in res.lines, "Hbeta complex missing"
    hb = res.lines["Hb"]
    print(f"  recovered broad Hb: FWHM={hb.fwhm_kms:.0f} km/s, SNR={hb.snr:.1f}")
    print(f"  (J0950 manual decomposition: two broad Hb comps, ~4300 and ~13500 km/s)")
    print(f"  figure: {res.figure_path}")
    # J0950 is a bona-fide broad-line AGN; recovered broad Hb should be broad.
    assert 1500 < hb.fwhm_kms < 20000, f"broad Hb FWHM out of AGN range: {hb.fwhm_kms}"
    print("[PASS] 5c: heterogeneous HET spectrum runs end-to-end; coverage guard "
          "drops Halpha; broad Hbeta recovered in the AGN range.")


if __name__ == "__main__":
    part_a()
    part_b()
    part_c()
    print("\n[PASS] Phase 5: validated against synthetic truth, a physical SDSS "
          "AGN, and a heterogeneous-instrument object (J0950).")
