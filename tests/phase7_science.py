"""Phase 7: line-shape science and anomaly scoring, against known truth.

These are unit tests in the strict sense: every profile is constructed
analytically, so the expected answer is known exactly rather than being
whatever the code happened to return when it was written.

Offline, fast, no fitting engine required -- safe for CI. Runs under pytest or
directly (``python tests/phase7_science.py``).
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from agn_egent.backends.base import DecompositionResult, LineMeasurement
from agn_egent import science as S
from agn_egent import anomaly as A

C = S.C_KMS


def _result(line_models=None, err_level=0.1, lines=None, continuum=None,
            narrow_lines=None, residual=None, npix=12000):
    """Build a DecompositionResult from analytic Gaussians.

    `line_models` maps a line key to (rest wavelength, [(amp, v_offset, sigma_kms)]).
    """
    wave = np.linspace(4000.0, 7000.0, npix)
    models = {}
    for key, (lam0, comps) in (line_models or {}).items():
        m = np.zeros_like(wave)
        for amp, v_off, sig_kms in comps:
            lam_c = lam0 * (1.0 + v_off / C)
            sig_aa = lam0 * sig_kms / C
            m += amp * np.exp(-0.5 * ((wave - lam_c) / sig_aa) ** 2)
        models[key] = m
    comps_d = {"err": np.full_like(wave, err_level)}
    if residual is not None:
        comps_d["residual"] = residual(wave)
    return DecompositionResult(
        name="test", z=0.1, backend="analytic", rest_wave=wave,
        components=comps_d, lines=lines or {}, continuum=continuum or {},
        quality={}, params={}, narrow_lines=narrow_lines or {},
        line_models=models)


def _hb_meas(**kw):
    kw.setdefault("fwhm_kms", 4000.0)
    kw.setdefault("sigma_kms", 1700.0)
    kw.setdefault("snr", 20.0)
    return {"Hb": LineMeasurement("Hb", **kw)}


# --- [O III] outflow kinematics ---------------------------------------------

def test_w80_matches_gaussian_analytic():
    """W80 of a Gaussian is 2*1.2816*sigma; recover it to better than 1%."""
    for sigma in (100.0, 250.0, 500.0):
        r = _result({"OIII5007": (S.REST_WAVE["OIII5007"], [(10.0, 0.0, sigma)])})
        o = S.oiii_outflow(r, inst_fwhm_kms=0.0)
        expected = 2 * 1.2816 * sigma
        assert abs(o.w80_kms - expected) / expected < 0.01, \
            f"sigma={sigma}: W80={o.w80_kms} expected {expected}"
        assert abs(o.v50_kms) < 5.0, "a symmetric line must have v50 ~ 0"
        assert abs(o.asymmetry) < 0.01, "a symmetric line must have asymmetry ~ 0"


def test_blue_wing_gives_blueshift_and_negative_asymmetry():
    r = _result({"OIII5007": (S.REST_WAVE["OIII5007"],
                              [(10.0, 0.0, 150.0), (4.0, -600.0, 400.0)])})
    o = S.oiii_outflow(r, inst_fwhm_kms=0.0)
    assert o.v50_kms < 0, "a blue wing must pull the centroid blueward"
    assert o.asymmetry < 0, "a blue wing must give negative asymmetry"
    assert o.outflow, "this profile should be selected as an outflow"


def test_instrumental_width_is_deconvolved():
    """An unresolved line must return W80 ~ 0, not the instrumental width."""
    sigma_inst = S.SDSS_INST_FWHM_KMS / 2.3548
    r = _result({"OIII5007": (S.REST_WAVE["OIII5007"], [(10.0, 0.0, sigma_inst)])})
    o = S.oiii_outflow(r, inst_fwhm_kms=S.SDSS_INST_FWHM_KMS)
    assert o.w80_observed_kms > 100.0
    assert o.w80_kms < 60.0, f"unresolved line should deconvolve to ~0, got {o.w80_kms}"


def test_offline_component_makes_shape_unreliable():
    """A component far off systemic is a degenerate fit, not an outflow.

    This is the guard that keeps a survey outflow list from filling up with bad
    [O III] fits: the numbers are still reported, but the flags stay off.
    """
    lam0 = S.REST_WAVE["OIII5007"]
    bad = lam0 * (1.0 - 2000.0 / C)      # a component 2000 km/s off the line
    r = _result({"OIII5007": (lam0, [(10.0, 0.0, 150.0), (6.0, -2000.0, 300.0)])},
                narrow_lines={"OIII5007c": {"peak_wave": lam0, "snr": 40.0},
                              "OIII5007w": {"peak_wave": bad, "snr": 25.0}})
    o = S.oiii_outflow(r, inst_fwhm_kms=0.0)
    assert o.detected, "the line itself is clearly detected"
    assert not o.reliable, "an off-systemic component must mark the shape unreliable"
    assert not o.outflow and not o.extreme_outflow, \
        "an unreliable shape must not raise outflow flags"


def test_noise_component_makes_shape_unreliable():
    lam0 = S.REST_WAVE["OIII5007"]
    r = _result({"OIII5007": (lam0, [(10.0, 0.0, 150.0), (0.4, -300.0, 500.0)])},
                narrow_lines={"OIII5007c": {"peak_wave": lam0, "snr": 40.0},
                              "OIII5007w": {"peak_wave": lam0, "snr": 1.2}})
    o = S.oiii_outflow(r, inst_fwhm_kms=0.0)
    assert not o.reliable and not o.outflow


# --- broad-line profile shape -----------------------------------------------

def test_double_peak_detected_only_when_resolved():
    """Two blended Gaussians are one peak; well-separated ones are two."""
    def profile(sep):
        r = _result({"Hb_br": (S.REST_WAVE["Hb"],
                               [(5.0, -sep / 2, 1200.0), (5.0, +sep / 2, 1200.0)])},
                    lines=_hb_meas())
        return S.broad_profile(r, "Hb")

    assert not profile(0.0).double_peaked
    assert not profile(800.0).double_peaked, "blended peaks must not be flagged"
    for sep in (3000.0, 6000.0):
        b = profile(sep)
        assert b.double_peaked, f"separation {sep} km/s should be detected"
        assert b.n_peaks == 2
        assert b.peak_contrast > S.DOUBLE_PEAK_MIN_CONTRAST


def test_red_wing_gives_positive_asymmetry_and_offset():
    r = _result({"Hb_br": (S.REST_WAVE["Hb"],
                           [(5.0, 0.0, 1500.0), (2.0, 3000.0, 2500.0)])},
                lines=_hb_meas())
    b = S.broad_profile(r, "Hb")
    assert b.asymmetry > 0, "a red wing must give positive asymmetry"
    assert b.v50_kms > 0


def test_data_double_peak_found_when_model_is_single_peaked():
    """The disk-emitter selector must work on the data, not the fitted model.

    Two Gaussians prefer a core+pedestal solution, so a genuinely double-peaked
    line can be fit by a single-peaked model. Here the data carry two peaks and
    the model does not -- the data test must still fire.
    """
    lam0 = S.REST_WAVE["Hb"]
    wave = np.linspace(4000.0, 7000.0, 12000)

    def gauss(amp, v_off, sig_kms):
        return amp * np.exp(-0.5 * ((wave - lam0 * (1 + v_off / C))
                                    / (lam0 * sig_kms / C)) ** 2)

    data = gauss(5.0, -4000.0, 1500.0) + gauss(5.0, 4000.0, 1500.0)  # disk-like
    model = gauss(9.0, 0.0, 3800.0)                                   # single peak
    r = DecompositionResult(
        name="disk", z=0.1, backend="analytic", rest_wave=wave,
        components={"err": np.full_like(wave, 0.05), "data": data,
                    "conti": np.zeros_like(wave), "narrow": np.zeros_like(wave)},
        lines=_hb_meas(), continuum={}, quality={}, params={},
        line_models={"Hb_br": model})

    b = S.broad_profile(r, "Hb")
    assert b.n_peaks == 1, "the model is single-peaked, as a real fit would be"
    assert not b.double_peaked, "so the model-based test cannot find it"
    assert b.data_double_peaked, "but the data-based test must"
    assert abs(b.data_peak_separation_kms - 8000.0) < 1200.0, \
        f"separation {b.data_peak_separation_kms}"
    assert S.science_report(r).flags["double_peaked"], \
        "the class flag must come from the data test"


def test_oiii_residual_is_not_a_disk_emitter():
    """The dominant false positive: leftover [O III] read as a second horn.

    Narrow-line subtraction never cancels exactly, and [O III] 5007 sits
    +8950 km/s from Hbeta -- inside the real disk-emitter separation range. A
    loose version of this test flagged 61% of a 300-quasar pilot on exactly this
    residual, so it has to stay rejected.
    """
    lam_hb, lam_o3 = S.REST_WAVE["Hb"], S.REST_WAVE["OIII5007"]
    wave = np.linspace(4000.0, 7000.0, 12000)
    broad = 5.0 * np.exp(-0.5 * ((wave - lam_hb) / (lam_hb * 3000.0 / C)) ** 2)
    # a sharp, badly-subtracted [O III] remnant
    resid_o3 = 4.0 * np.exp(-0.5 * ((wave - lam_o3) / (lam_o3 * 300.0 / C)) ** 2)
    r = DecompositionResult(
        name="o3resid", z=0.1, backend="analytic", rest_wave=wave,
        components={"err": np.full_like(wave, 0.05), "data": broad + resid_o3,
                    "conti": np.zeros_like(wave), "narrow": np.zeros_like(wave)},
        lines=_hb_meas(), continuum={}, quality={}, params={},
        line_models={"Hb_br": broad})
    b = S.broad_profile(r, "Hb")
    assert not b.data_double_peaked, \
        f"[O III] residual flagged as a disk emitter (sep={b.data_peak_separation_kms})"


def test_lopsided_wing_bump_is_not_a_disk_emitter():
    """Two peaks of very unequal height are a wing bump, not disk horns."""
    lam0 = S.REST_WAVE["Hb"]
    wave = np.linspace(4000.0, 7000.0, 12000)

    def g(amp, v_off, sig):
        return amp * np.exp(-0.5 * ((wave - lam0 * (1 + v_off / C))
                                    / (lam0 * sig / C)) ** 2)

    data = g(10.0, 0.0, 2500.0) + g(1.2, 5000.0, 900.0)
    r = DecompositionResult(
        name="bump", z=0.1, backend="analytic", rest_wave=wave,
        components={"err": np.full_like(wave, 0.05), "data": data,
                    "conti": np.zeros_like(wave), "narrow": np.zeros_like(wave)},
        lines=_hb_meas(), continuum={}, quality={}, params={},
        line_models={"Hb_br": data})
    assert not S.broad_profile(r, "Hb").data_double_peaked


def test_interpolation_bridge_is_not_a_horn():
    """A peak on masked (interpolated) pixels must not count.

    When strong [O III] dominates a wide stretch of the window, the whole
    stretch is masked and the profile is bridged by interpolation. The bridge
    ends at whatever level the far side sits, and that endpoint looks like a
    horn -- a 5k campaign's top 'disk emitter' (contrast 1.0) was exactly this.
    """
    lam0 = S.REST_WAVE["Hb"]
    wave = np.linspace(4000.0, 7000.0, 12000)
    rng = np.random.default_rng(43)

    def g(amp, v_off, sig):
        return amp * np.exp(-0.5 * ((wave - lam0 * (1 + v_off / C))
                                    / (lam0 * sig / C)) ** 2)

    broad = 6.0 * g(1.0, 0.0, 1800.0)
    # a strong, WIDE narrow complex red of the line, and messy residual flux
    # just beyond it, so the interpolation bridge ends on an elevated shelf
    narrow_model = 40.0 * g(1.0, 7000.0, 900.0)
    shelf = 3.0 * g(1.0, 10500.0, 500.0)
    data = broad + narrow_model * 1.02 + shelf + rng.normal(0, 0.2, wave.size)
    r = DecompositionResult(
        name="bridge", z=0.1, backend="analytic", rest_wave=wave,
        components={"err": np.full_like(wave, 0.2), "data": data,
                    "conti": np.zeros_like(wave), "narrow": narrow_model},
        lines=_hb_meas(), continuum={}, quality={}, params={},
        line_models={"Hb_br": broad})
    b = S.broad_profile(r, "Hb")
    assert not b.data_double_peaked, \
        f"bridge endpoint counted as a horn (sep={b.data_peak_separation_kms})"


def test_oversubtracted_trough_is_not_a_disk_emitter():
    """A negative trough from over-subtraction must not explode the contrast.

    The empirical profile dips below zero wherever continuum/host subtraction
    overshoots; with an unclipped profile, (weaker - trough)/weaker is
    unbounded, and a 5k campaign ranked pure subtraction disasters (contrast
    ~1e8) as its best disk emitters. Clipped, the same object must produce a
    bounded contrast and fail the peak-significance cut.
    """
    lam0 = S.REST_WAVE["Hb"]
    wave = np.linspace(4000.0, 7000.0, 12000)
    rng = np.random.default_rng(31)

    def g(amp, v_off, sig):
        return amp * np.exp(-0.5 * ((wave - lam0 * (1 + v_off / C))
                                    / (lam0 * sig / C)) ** 2)

    # two barely-positive noise bumps around a deeply negative subtraction hole
    data = (0.15 * g(1.0, -4000.0, 900.0) + 0.15 * g(1.0, 4000.0, 900.0)
            - 3.0 * g(1.0, 0.0, 1500.0) + rng.normal(0, 0.05, wave.size))
    r = DecompositionResult(
        name="oversub", z=0.1, backend="analytic", rest_wave=wave,
        components={"err": np.full_like(wave, 0.05), "data": data,
                    "conti": np.zeros_like(wave), "narrow": np.zeros_like(wave)},
        lines=_hb_meas(), continuum={}, quality={}, params={},
        line_models={"Hb_br": np.clip(data, 0, None)})
    b = S.broad_profile(r, "Hb")
    if b.data_n_peaks >= 2:
        assert b.data_peak_contrast <= 1.0 + 1e-9, \
            f"contrast must be bounded, got {b.data_peak_contrast}"
    assert not b.data_double_peaked, "a subtraction hole is not a disk profile"


def test_data_double_peak_not_fired_by_noise():
    """A noisy single-peaked line must not be called a disk emitter."""
    lam0 = S.REST_WAVE["Hb"]
    wave = np.linspace(4000.0, 7000.0, 12000)
    rng = np.random.default_rng(19)
    single = 5.0 * np.exp(-0.5 * ((wave - lam0) / (lam0 * 3000.0 / C)) ** 2)
    r = DecompositionResult(
        name="noisy", z=0.1, backend="analytic", rest_wave=wave,
        components={"err": np.full_like(wave, 0.3),
                    "data": single + rng.normal(0, 0.3, wave.size),
                    "conti": np.zeros_like(wave), "narrow": np.zeros_like(wave)},
        lines=_hb_meas(), continuum={}, quality={}, params={},
        line_models={"Hb_br": single})
    assert not S.broad_profile(r, "Hb").data_double_peaked


def test_undetected_broad_line_is_not_double_peaked():
    """Shape flags must require a detection, or noise becomes a discovery."""
    r = _result({"Hb_br": (S.REST_WAVE["Hb"],
                           [(5.0, -2000.0, 1200.0), (5.0, 2000.0, 1200.0)])},
                lines=_hb_meas(snr=1.0))
    b = S.broad_profile(r, "Hb")
    assert b.n_peaks == 2, "the peaks are there ..."
    assert not b.double_peaked, "... but an undetected line must not be selected"


# --- Fe II -------------------------------------------------------------------

def test_r_feii_ratio():
    r = _result(continuum={"Fe_flux_4435_4685": 50.0},
                lines=_hb_meas(flux=100.0))
    f = S.feii_strength(r)
    assert abs(f.r_feii - 0.5) < 1e-9
    assert not f.strong_feii
    r2 = _result(continuum={"Fe_flux_4435_4685": 150.0}, lines=_hb_meas(flux=100.0))
    assert S.feii_strength(r2).strong_feii


# --- classification ----------------------------------------------------------

def test_nls1_and_type19_flags():
    narrow = _result({"Hb_br": (S.REST_WAVE["Hb"], [(5.0, 0.0, 600.0)])},
                     lines=_hb_meas(fwhm_kms=1500.0, sigma_kms=640.0, flux=100.0))
    rep = S.science_report(narrow)
    assert rep.flags["nls1"], "FWHM 1500 km/s with no [O III] should be an NLS1"

    ha_only = _result({"Ha_br": (S.REST_WAVE["Ha"], [(5.0, 0.0, 2000.0)])},
                      lines={"Ha": LineMeasurement("Ha", fwhm_kms=5000.0,
                                                   sigma_kms=2100.0, snr=15.0),
                             "Hb": LineMeasurement("Hb", fwhm_kms=float("nan"),
                                                   sigma_kms=float("nan"), snr=1.0)})
    rep2 = S.science_report(ha_only)
    assert rep2.flags["type1_9"], "broad Ha without broad Hb is a Type 1.9"


def test_empty_result_does_not_raise():
    rep = S.science_report(_result())
    assert rep.active_flags == []
    assert "flags:" in rep.summary()


# --- anomaly scoring ---------------------------------------------------------

def test_anomaly_null_is_near_one():
    """White-noise residuals must score ~1 and never approach the notable cut.

    Checked at two samplings, because the statistic must not depend on the
    pixel scale: normalizing by the total residual scatter instead of its white
    component would cap the score at sqrt(kernel pixels) and make the null drift
    with resolution.
    """
    rng = np.random.default_rng(7)
    for npix in (2650, 6000):          # SDSS-like, and twice as fine
        scores = np.array([A.anomaly_score(
            _result(residual=lambda w: rng.normal(0, 1, w.size),
                    err_level=1.0, npix=npix)).score for _ in range(8)])
        assert 0.8 < scores.mean() < 1.4, \
            f"npix={npix}: null mean drifted to {scores.mean():.3f}"
        assert scores.max() < A.NOTABLE_SCORE, \
            f"npix={npix}: pure noise hit the notable threshold ({scores.max():.3f})"


def test_anomaly_score_is_unbounded():
    """Score must keep climbing with structure, not saturate at a ceiling.

    The earlier version divided by the total residual scatter, which capped it
    at sqrt(kernel) ~ 3 on SDSS sampling; a 300-object pilot then had a median
    of 2.81 and a maximum of exactly 3.00, so the ranking was meaningless.
    """
    rng = np.random.default_rng(23)
    sig_aa = S.REST_WAVE["Hb"] * 3000.0 / C

    def score_for(amp):
        def resid(w):
            return (rng.normal(0, 1, w.size)
                    + amp * np.exp(-0.5 * ((w - 4900.0) / sig_aa) ** 2))
        return A.anomaly_score(_result(residual=resid, err_level=1.0,
                                       npix=2650)).score

    s2, s4 = score_for(2.0), score_for(4.0)
    assert s4 > s2 > A.NOTABLE_SCORE, f"scores did not climb: {s2:.2f}, {s4:.2f}"
    assert s4 > 4.0, f"score looks capped at {s4:.2f}"


def test_anomaly_ignores_amplitude_only_residual():
    """A large but incoherent residual is chi^2's problem, not an anomaly."""
    rng = np.random.default_rng(11)
    r = _result(residual=lambda w: rng.normal(0, 3, w.size), err_level=1.0, npix=6000)
    rep = A.anomaly_score(r)
    assert rep.windows["Hb"].pull_rms > 2.5, "the residual really is large"
    assert rep.score < A.NOTABLE_SCORE, "but it is not structured, so must not flag"


def test_anomaly_detects_broad_structure():
    rng = np.random.default_rng(3)
    sig_aa = S.REST_WAVE["Hb"] * 3000.0 / C

    def resid(w):
        return rng.normal(0, 1, w.size) + 2.0 * np.exp(-0.5 * ((w - 4900.0) / sig_aa) ** 2)

    rep = A.anomaly_score(_result(residual=resid, err_level=1.0, npix=6000))
    assert rep.score > A.STRONG_SCORE, f"missed a 2-sigma broad bump: {rep.score:.2f}"
    assert rep.worst_window == "Hb"


def test_anomaly_suppresses_unresolved_spike():
    """A cosmic ray must not outrank real broad structure in a survey ranking."""
    rng = np.random.default_rng(5)
    sig_aa = S.REST_WAVE["Hb"] * 150.0 / C

    def resid(w):
        return rng.normal(0, 1, w.size) + 3.0 * np.exp(-0.5 * ((w - 4900.0) / sig_aa) ** 2)

    rep = A.anomaly_score(_result(residual=resid, err_level=1.0, npix=6000))
    assert rep.score < A.NOTABLE_SCORE, f"spike scored {rep.score:.2f}"


def test_anomaly_missing_inputs():
    assert not np.isfinite(A.anomaly_score(_result()).score)


def main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"[PASS] {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"[FAIL] {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
