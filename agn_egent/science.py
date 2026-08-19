"""AGN science quantities beyond the black-hole mass.

`measure.derive` turns a decomposition into M_BH / L_bol / Eddington ratio. This
module reads the *shapes* of the fitted lines instead, and produces the
quantities a survey-scale search actually selects on:

* :func:`oiii_outflow`   -- [O III] 5007 non-parametric width/velocity (W80, v50,
  asymmetry): the standard ionized-outflow diagnostic.
* :func:`feii_strength`  -- R_FeII and the Eigenvector-1 position.
* :func:`broad_profile`  -- broad-line shape: asymmetry, velocity offset and a
  double-peak detector (disk emitters).
* :func:`classify`       -- boolean population flags built from the above.
* :func:`science_report` -- runs all of them and returns one serializable record.

Everything here is deterministic and reads only a
:class:`~agn_egent.backends.base.DecompositionResult`, so it is free to run on
every object of a survey campaign and is backend-independent.

Line shapes are measured on the *fitted model profile*, not the data. That is
deliberate: the model is already deblended (narrow removed from broad, Fe II and
host removed from both) and noise-free, which is what makes a non-parametric
width meaningful at survey S/N. The cost is that a shape can only be as good as
the fit -- so every shape quantity here should be read together with the fit's
quality flag, and the residual-based :mod:`agn_egent.anomaly` score is the
complementary check that catches profiles the model *failed* to capture.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict

import numpy as np

C_KMS = 299792.458

# Vacuum rest wavelengths [A], matching the fitting config (qsopar_config).
REST_WAVE = {
    "Ha": 6564.61,
    "Hb": 4862.68,
    "OIII5007": 5008.24,
    "OIII4959": 4960.30,
    "NII6585": 6585.28,
    "SII6718": 6718.29,
    "SII6732": 6732.67,
    "HeII4687": 4687.02,
}

# Fe II integration window (Boroson & Green 1992 definition of R_FeII).
FEII_WINDOW = (4434.0, 4684.0)

# SDSS spectral resolution R ~ 2000 => a velocity FWHM of ~150 km/s. Narrow-line
# widths are corrected for this in quadrature; override for other instruments.
SDSS_INST_FWHM_KMS = 150.0

# W80 / FWHM for a Gaussian: W80 = 2 * 1.2816 sigma, FWHM = 2.3548 sigma.
W80_PER_FWHM = 1.088

# --- selection thresholds (documented so a campaign can justify its cuts) -----
OUTFLOW_W80_KMS = 600.0        # broader than a typical bulge-driven narrow line
EXTREME_OUTFLOW_W80_KMS = 1000.0
OUTFLOW_V50_KMS = -100.0       # net blueshift of the [O III] centroid
EXTREME_OUTFLOW_V50_KMS = -300.0
NLS1_FWHM_KMS = 2000.0         # classic Osterbrock & Pogge (1985) definition
NLS1_OIII_HB_MAX = 3.0
STRONG_FEII = 1.0              # R_FeII > 1 is the Eigenvector-1 extreme
BROAD_SNR_DETECT = 3.0         # below this a broad line is not detected
# A narrow-line fitting component further than this from systemic is a symptom
# of the degenerate [O III] core/wing pair, not a real outflow component.
MAX_COMPONENT_OFFSET_KMS = 900.0
WEAK_BROAD_EW_AA = 15.0
HOST_DOMINATED_FRAC = 0.5
DOUBLE_PEAK_MIN_SEP_KMS = 1000.0
DOUBLE_PEAK_MIN_CONTRAST = 0.05   # trough must sit 5% below the weaker peak

# --- data-based double-peak (disk emitter) selection -------------------------
# Tuned against a 300-quasar pilot, where a loose version flagged 61% of objects.
# Nearly all of those were narrow-line subtraction residuals read as horns:
# [O III] 5007 lies +8950 km/s from Hbeta and 4959 lies +6030, right inside the
# separation range real disk emitters occupy, so they cannot be excluded by
# separation alone. Hence the mask below plus genuinely strict peak criteria.
DATA_SMOOTH_KMS = 800.0
DATA_DOUBLE_PEAK_MIN_CONTRAST = 0.15
DATA_PEAK_MIN_PROMINENCE = 0.15    # fraction of the profile maximum
DATA_PEAK_MIN_HEIGHT_RATIO = 0.40  # horns are comparable; a wing bump is not
DATA_DOUBLE_PEAK_MIN_SEP_KMS = 2000.0
DATA_DOUBLE_PEAK_MAX_SEP_KMS = 15000.0
DATA_TROUGH_MAX_OFFSET_KMS = 3500.0   # the dip straddles systemic in a disk
DATA_DOUBLE_PEAK_MIN_SNR = 5.0
DATA_SEARCH_WINDOW_KMS = 12000.0
# The dip between the horns must be deep compared with the *noise*, not just
# with the peak height. Without this, a noisy line (Halpha at z~0.36 lands at
# 8900 A, among the worst SDSS sky residuals) produces dips of large relative
# contrast out of pure noise.
DATA_DIP_MIN_SIGNIFICANCE = 4.0
# Narrow lines whose imperfectly-subtracted residuals sit inside the broad-line
# search window; interpolated over before peaks are counted.
NARROW_MASK_LINES = ("OIII5007", "OIII4959", "HeII4687",
                     "NII6585", "NII6549", "SII6718", "SII6732")
NARROW_MASK_HALF_KMS = 1200.0
# The Balmer narrow core sits at the broad line's own centre, so an imperfectly
# subtracted one carves a notch exactly where a disk profile's dip belongs --
# and the "trough straddles systemic" criterion would then *select* for it.
# Interpolating over a width matched to the narrow line removes such a notch
# while leaving a genuine, much wider central depression intact.
NARROW_CORE_MASK_HALF_KMS = 800.0
BROAD_OFFSET_KMS = 1000.0      # |v50(broad)| above this is a notable offset


def _finite(x) -> bool:
    """True for a real, finite number (accepts numpy scalars, rejects None/NaN)."""
    if x is None or isinstance(x, bool):
        return False
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _nan_to_none(x):
    """JSON-safe: NaN/inf are not valid JSON, so serialize them as null."""
    if x is None:
        return None
    if isinstance(x, (np.floating, np.integer)):
        x = float(x)
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def velocity_grid(wave: np.ndarray, lambda0: float) -> np.ndarray:
    """Doppler velocity [km/s] of each wavelength relative to `lambda0`."""
    return C_KMS * (np.asarray(wave, dtype=float) - lambda0) / lambda0


def percentile_velocities(v: np.ndarray, flux: np.ndarray,
                          percentiles=(5, 10, 50, 90, 95)) -> dict:
    """Velocities enclosing given percentiles of the line flux.

    Non-parametric: no assumption that the profile is Gaussian, which is the
    whole point for outflow wings and double-peaked profiles. Negative model
    excursions are clipped so the cumulative distribution is monotonic.
    """
    v = np.asarray(v, dtype=float)
    f = np.clip(np.asarray(flux, dtype=float), 0.0, None)
    f[~np.isfinite(f)] = 0.0
    if v.size < 3 or not np.any(f > 0):
        return {p: float("nan") for p in percentiles}
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(v))])
    total = cum[-1]
    if not (total > 0):
        return {p: float("nan") for p in percentiles}
    frac = cum / total
    return {p: float(np.interp(p / 100.0, frac, v)) for p in percentiles}


def _deconvolve(width_kms: float, inst_fwhm_kms: float, factor: float) -> float:
    """Remove the instrumental width from a measured width in quadrature."""
    if not _finite(width_kms):
        return float("nan")
    inst = factor * inst_fwhm_kms
    return math.sqrt(max(width_kms ** 2 - inst ** 2, 0.0))


def _local_noise(result, lambda0: float, half_window_aa: float = 25.0) -> float:
    err = result.components.get("err")
    if err is None:
        return float("nan")
    w = np.asarray(result.rest_wave, dtype=float)
    sel = np.abs(w - lambda0) <= half_window_aa
    e = np.asarray(err, dtype=float)[sel]
    e = e[np.isfinite(e) & (e > 0)]
    return float(np.median(e)) if e.size else float("nan")


# ---------------------------------------------------------------------------
# [O III] outflows
# ---------------------------------------------------------------------------

@dataclass
class OIIIOutflow:
    """Non-parametric [O III] 5007 kinematics (the ionized-outflow diagnostic)."""
    detected: bool = False
    reliable: bool = False              # shape is trustworthy (see `oiii_outflow`)
    n_components: int = 0
    min_component_snr: float = float("nan")
    snr: float = float("nan")
    flux: float = float("nan")          # integrated model flux [1e-17 cgs]
    w80_kms: float = float("nan")       # instrument-corrected 80% width
    w80_observed_kms: float = float("nan")
    v50_kms: float = float("nan")       # median velocity vs systemic (- = blueshift)
    v05_kms: float = float("nan")
    v10_kms: float = float("nan")
    v90_kms: float = float("nan")
    v95_kms: float = float("nan")
    asymmetry: float = float("nan")     # >0 red wing, <0 blue wing
    peak_v_kms: float = float("nan")
    outflow: bool = False
    extreme_outflow: bool = False

    def to_dict(self) -> dict:
        return {k: _nan_to_none(v) for k, v in asdict(self).items()}


def _line_components(result, physical_line: str) -> list:
    """Diagnostics for each fitting component that makes up one physical line."""
    from .backends.pyqsofit_backend import physical_line_name
    return [d for name, d in (result.narrow_lines or {}).items()
            if physical_line_name(name) == physical_line]


def oiii_outflow(result, inst_fwhm_kms: float = SDSS_INST_FWHM_KMS,
                 window_kms: float = 3000.0) -> OIIIOutflow:
    """Measure [O III] 5007 W80 / v50 / asymmetry from the fitted profile.

    The profile is the sum of *all* [O III] 5007 fitting components (core plus
    any outflow wing), so the measurement does not depend on how the fitter
    happened to split the line between Gaussians.

    W80 is corrected for the instrumental resolution in quadrature; a spectrally
    unresolved line therefore returns W80 ~ 0 rather than ~160 km/s.

    ``reliable`` is the guard that matters at survey scale. The default model
    gives [O III] a core plus a wing Gaussian that are not flux-tied to each
    other or to [O III] 4959, so the pair is degenerate: one component can drift
    several hundred km/s off the line, or latch onto noise. Either is harmless
    for the line flux but inflates W80 and drags v50 -- which would fill an
    outflow candidate list with bad fits instead of outflows. A shape is
    therefore called reliable only when the line is detected, every contributing
    component is itself above the detection threshold, and no component sits
    further than ``MAX_COMPONENT_OFFSET_KMS`` from the systemic wavelength.
    Unreliable shapes are still reported; they are just not allowed to raise the
    outflow flags.
    """
    out = OIIIOutflow()
    model = result.line_models.get("OIII5007")
    if model is None:
        return out
    lam0_line = REST_WAVE["OIII5007"]
    comps = _line_components(result, "OIII5007")
    snrs = [c.get("snr") for c in comps if _finite(c.get("snr"))]
    out.n_components = len(comps)
    out.min_component_snr = float(min(snrs)) if snrs else float("nan")
    offsets = [abs(C_KMS * (float(c["peak_wave"]) - lam0_line) / lam0_line)
               for c in comps if _finite(c.get("peak_wave"))]
    max_offset = max(offsets) if offsets else 0.0
    lam0 = REST_WAVE["OIII5007"]
    v = velocity_grid(result.rest_wave, lam0)
    sel = np.abs(v) <= window_kms
    if sel.sum() < 5:
        return out
    vv, ff = v[sel], np.asarray(model, dtype=float)[sel]

    peak = float(np.nanmax(ff)) if ff.size else float("nan")
    noise = _local_noise(result, lam0)
    out.snr = peak / noise if (_finite(peak) and _finite(noise) and noise > 0) else float("nan")
    out.detected = bool(_finite(out.snr) and out.snr >= BROAD_SNR_DETECT)
    # integrate in wavelength so the flux keeps its usual 1e-17 erg/s/cm^2 units
    lam = np.asarray(result.rest_wave, dtype=float)[sel]
    out.flux = float(np.trapezoid(np.clip(ff, 0.0, None), lam))
    if _finite(peak) and peak > 0:
        out.peak_v_kms = float(vv[int(np.nanargmax(ff))])

    p = percentile_velocities(vv, ff)
    out.v05_kms, out.v10_kms = p[5], p[10]
    out.v50_kms, out.v90_kms, out.v95_kms = p[50], p[90], p[95]
    if _finite(p[90]) and _finite(p[10]):
        out.w80_observed_kms = p[90] - p[10]
        out.w80_kms = _deconvolve(out.w80_observed_kms, inst_fwhm_kms, W80_PER_FWHM)
        if out.w80_observed_kms > 0 and _finite(p[50]):
            out.asymmetry = ((p[90] - p[50]) - (p[50] - p[10])) / out.w80_observed_kms

    out.reliable = bool(
        out.detected
        and (not snrs or out.min_component_snr >= BROAD_SNR_DETECT)
        and max_offset <= MAX_COMPONENT_OFFSET_KMS)

    if out.reliable:
        out.outflow = bool((_finite(out.w80_kms) and out.w80_kms > OUTFLOW_W80_KMS)
                           or (_finite(out.v50_kms) and out.v50_kms < OUTFLOW_V50_KMS))
        out.extreme_outflow = bool(
            (_finite(out.w80_kms) and out.w80_kms > EXTREME_OUTFLOW_W80_KMS)
            or (_finite(out.v50_kms) and out.v50_kms < EXTREME_OUTFLOW_V50_KMS))
    return out


# ---------------------------------------------------------------------------
# Fe II strength / Eigenvector 1
# ---------------------------------------------------------------------------

@dataclass
class FeIIStrength:
    """Optical Fe II strength -- the primary Eigenvector-1 coordinate."""
    feii_flux: float = float("nan")     # integrated 4434-4684 A [1e-17 cgs]
    hbeta_broad_flux: float = float("nan")
    r_feii: float = float("nan")        # F(FeII) / F(broad Hbeta)
    fwhm_hbeta_kms: float = float("nan")
    strong_feii: bool = False

    def to_dict(self) -> dict:
        return {k: _nan_to_none(v) for k, v in asdict(self).items()}


def feii_strength(result) -> FeIIStrength:
    """R_FeII = F(Fe II 4434-4684) / F(broad Hbeta), plus the EV1 coordinates.

    Prefers PyQSOFit's own Fe II window flux when present and falls back to
    integrating the fitted Fe II template over the same window.
    """
    out = FeIIStrength()
    fe = None
    for key, val in result.continuum.items():
        if key.startswith("Fe_flux_44") and _finite(val):
            fe = float(val)
            break
    if fe is None:
        arr = result.components.get("feii_op")
        if arr is not None:
            w = np.asarray(result.rest_wave, dtype=float)
            sel = (w >= FEII_WINDOW[0]) & (w <= FEII_WINDOW[1])
            if sel.sum() > 2:
                fe = float(np.trapezoid(np.asarray(arr, dtype=float)[sel], w[sel]))
    if fe is not None:
        out.feii_flux = fe

    hb = result.lines.get("Hb")
    if hb is not None:
        out.hbeta_broad_flux = float(hb.flux)
        out.fwhm_hbeta_kms = float(hb.fwhm_kms)
        if _finite(fe) and _finite(hb.flux) and hb.flux > 0:
            out.r_feii = fe / float(hb.flux)
    out.strong_feii = bool(_finite(out.r_feii) and out.r_feii > STRONG_FEII)
    return out


# ---------------------------------------------------------------------------
# Broad-line profile shape (asymmetry, offset, double peaks)
# ---------------------------------------------------------------------------

@dataclass
class BroadProfile:
    """Shape of one broad-line complex, measured on the fitted profile."""
    complex_name: str = ""
    detected: bool = False
    snr: float = float("nan")
    fwhm_kms: float = float("nan")
    sigma_kms: float = float("nan")
    shape_fwhm_over_sigma: float = float("nan")   # Gaussian = 2.355
    v50_kms: float = float("nan")                 # centroid offset vs systemic
    peak_v_kms: float = float("nan")
    w80_kms: float = float("nan")
    asymmetry: float = float("nan")               # >0 red wing, <0 blue wing
    n_peaks: int = 0                              # peaks in the *fitted model*
    peak_separation_kms: float = float("nan")
    peak_contrast: float = float("nan")           # trough depth below weaker peak
    double_peaked: bool = False
    offset: bool = False                          # |v50| > BROAD_OFFSET_KMS
    # the same test run on the continuum/narrow-subtracted *data*, which is what
    # actually finds disk emitters (see `broad_profile`)
    data_n_peaks: int = 0
    data_peak_separation_kms: float = float("nan")
    data_peak_contrast: float = float("nan")
    data_dip_significance: float = float("nan")   # dip depth / smoothed noise
    data_double_peaked: bool = False

    def to_dict(self) -> dict:
        return {k: _nan_to_none(v) for k, v in asdict(self).items()}


def _find_peaks(v: np.ndarray, f: np.ndarray, min_prominence_frac: float = 0.05):
    """Local maxima of a profile, as (index, height), strongest first.

    A plain neighbour comparison plus a prominence cut against the deepest
    adjacent trough. Written out rather than pulled from scipy.signal so the
    prominence definition is explicit and the module stays dependency-light.
    """
    if f.size < 5:
        return []
    peak_max = float(np.nanmax(f))
    if not (peak_max > 0):
        return []
    idx = [i for i in range(1, f.size - 1) if f[i] > f[i - 1] and f[i] >= f[i + 1]]
    kept = []
    for i in idx:
        # deepest trough between this peak and any higher peak on either side
        left = f[:i]
        right = f[i + 1:]
        higher_left = np.where(left > f[i])[0]
        higher_right = np.where(right > f[i])[0]
        lo = float(np.min(left[higher_left[-1]:])) if higher_left.size else float(np.min(left)) if left.size else 0.0
        hi = float(np.min(right[:higher_right[0]])) if higher_right.size else float(np.min(right)) if right.size else 0.0
        prominence = f[i] - max(lo, hi)
        if prominence >= min_prominence_frac * peak_max:
            kept.append((i, float(f[i])))
    kept.sort(key=lambda t: -t[1])
    return kept


def _peak_stats(v: np.ndarray, f: np.ndarray, min_prominence_frac: float = 0.05):
    """(n_peaks, separation, contrast) for the two strongest peaks of a profile."""
    peaks = _find_peaks(v, f, min_prominence_frac=min_prominence_frac)
    if len(peaks) < 2:
        return len(peaks), float("nan"), float("nan")
    (i1, h1), (i2, h2) = peaks[0], peaks[1]
    lo, hi = sorted((i1, i2))
    trough = float(np.min(f[lo:hi + 1]))
    weaker = min(h1, h2)
    contrast = (weaker - trough) / weaker if weaker > 0 else float("nan")
    return len(peaks), float(abs(v[i1] - v[i2])), contrast


def _empirical_broad_profile(result, complex_name: str, sel: np.ndarray):
    """The observed broad line: data minus continuum, Fe II, host, narrow lines
    and every *other* broad component.

    This is the profile a person looks at when they call something
    double-peaked, and it does not depend on how the Gaussians were arranged.

    Subtracting the other broad lines matters more than it looks: broad He II
    4687 sits 10800 km/s blueward of Hbeta and would otherwise read as a second
    peak, manufacturing disk-emitter candidates out of ordinary quasars.
    Returns None if the components needed to isolate the line are missing.
    """
    comps = result.components
    data = comps.get("data")
    conti = comps.get("conti")
    if data is None or conti is None:
        return None
    prof = np.asarray(data, dtype=float) - np.asarray(conti, dtype=float)
    narrow = comps.get("narrow")
    if narrow is not None:
        prof = prof - np.asarray(narrow, dtype=float)
    keep = f"{complex_name}_br"
    for key, model in (result.line_models or {}).items():
        if "_br" in key and key != keep:
            prof = prof - np.asarray(model, dtype=float)
    return prof[sel]


def _mask_narrow_residuals(wave: np.ndarray, prof: np.ndarray,
                           complex_name: str | None = None) -> np.ndarray:
    """Interpolate the profile across the narrow-line positions.

    Subtracting the fitted narrow lines never cancels them exactly, and what is
    left is a sharp spike sitting at a fixed velocity from the broad line --
    +8950 km/s for [O III] 5007 relative to Hbeta. That is squarely inside the
    range of real disk-emitter horn separations, so it cannot be rejected after
    the fact; it has to be removed before peaks are counted. Interpolating
    (rather than cutting) keeps the array contiguous so the gap edges do not
    themselves look like peaks.
    """
    bad = np.zeros(wave.shape, dtype=bool)
    for line in NARROW_MASK_LINES:
        lam0 = REST_WAVE.get(line)
        if lam0 is None:
            continue
        bad |= np.abs(velocity_grid(wave, lam0)) <= NARROW_MASK_HALF_KMS
    own = REST_WAVE.get(complex_name) if complex_name else None
    if own is not None:
        bad |= np.abs(velocity_grid(wave, own)) <= NARROW_CORE_MASK_HALF_KMS
    if not bad.any() or bad.all():
        return prof
    out = prof.copy()
    out[bad] = np.interp(wave[bad], wave[~bad], prof[~bad])
    return out


def _smooth_profile(f: np.ndarray, v: np.ndarray, width_kms: float) -> np.ndarray:
    """Boxcar-smooth a profile over a velocity width (noise must not make peaks)."""
    if f.size < 5:
        return f
    dv = float(np.median(np.abs(np.diff(v))))
    if not (dv > 0):
        return f
    k = int(round(width_kms / dv))
    if k < 3:
        return f
    k = k if k % 2 else k + 1
    pad = k // 2
    return np.convolve(np.pad(f, pad, mode="reflect"), np.ones(k) / k, mode="valid")


def broad_profile(result, complex_name: str = "Hb",
                  window_kms: float = 20000.0) -> BroadProfile:
    """Measure the shape of a broad-line complex, including double peaks.

    The double-peak test is the disk-emitter selector: two maxima separated by
    more than ``DOUBLE_PEAK_MIN_SEP_KMS`` with a real trough between them.

    It is run twice, and the *data* version is the one that matters. Measured on
    the fitted model it almost never fires: given two Gaussians, the optimizer
    reliably prefers a narrow-core-plus-broad-pedestal solution over two offset
    components, so the model profile comes out single-peaked even when the data
    are not. On a 300-quasar pilot the model-based test found two peaks in zero
    objects, which says more about the parameterization than about the sky. The
    data version subtracts the continuum, Fe II, host and narrow lines and looks
    at what is left, so it sees the profile whatever the Gaussians did with it.

    ``double_peaked`` (model) is kept because a model that *does* split into two
    peaks is a strong candidate, but ``data_double_peaked`` is the selector.
    """
    out = BroadProfile(complex_name=complex_name)
    lam0 = REST_WAVE.get(complex_name)
    model = result.line_models.get(f"{complex_name}_br")
    if lam0 is None or model is None:
        return out

    meas = result.lines.get(complex_name)
    if meas is not None:
        out.fwhm_kms = float(meas.fwhm_kms)
        out.sigma_kms = float(meas.sigma_kms)
        out.snr = float(meas.snr)
        if _finite(meas.fwhm_kms) and _finite(meas.sigma_kms) and meas.sigma_kms > 0:
            out.shape_fwhm_over_sigma = float(meas.fwhm_kms) / float(meas.sigma_kms)
    out.detected = bool(_finite(out.snr) and out.snr >= BROAD_SNR_DETECT)

    v = velocity_grid(result.rest_wave, lam0)
    sel = np.abs(v) <= window_kms
    if sel.sum() < 5:
        return out
    vv = v[sel]
    ff = np.clip(np.asarray(model, dtype=float)[sel], 0.0, None)
    if not np.any(ff > 0):
        return out

    p = percentile_velocities(vv, ff)
    out.v50_kms = p[50]
    if _finite(p[90]) and _finite(p[10]):
        out.w80_kms = p[90] - p[10]
        if out.w80_kms > 0 and _finite(p[50]):
            out.asymmetry = ((p[90] - p[50]) - (p[50] - p[10])) / out.w80_kms
    out.peak_v_kms = float(vv[int(np.nanargmax(ff))])
    out.offset = bool(_finite(out.v50_kms) and abs(out.v50_kms) > BROAD_OFFSET_KMS)

    n, sep, contrast = _peak_stats(vv, ff)
    out.n_peaks, out.peak_separation_kms, out.peak_contrast = n, sep, contrast
    out.double_peaked = bool(
        out.detected and n >= 2 and sep >= DOUBLE_PEAK_MIN_SEP_KMS
        and _finite(contrast) and contrast >= DOUBLE_PEAK_MIN_CONTRAST)

    _measure_data_peaks(result, complex_name, out, v)
    return out


def _measure_data_peaks(result, complex_name: str, out: BroadProfile,
                        v: np.ndarray) -> None:
    """Disk-emitter test on the observed profile (see `broad_profile`)."""
    dsel = np.abs(v) <= DATA_SEARCH_WINDOW_KMS
    if dsel.sum() < 20:
        return
    dprof = _empirical_broad_profile(result, complex_name, dsel)
    if dprof is None:
        return
    wave = np.asarray(result.rest_wave, dtype=float)[dsel]
    vv = v[dsel]
    prof = _smooth_profile(_mask_narrow_residuals(wave, dprof, complex_name),
                           vv, DATA_SMOOTH_KMS)
    peaks = _find_peaks(vv, prof, min_prominence_frac=DATA_PEAK_MIN_PROMINENCE)
    out.data_n_peaks = len(peaks)
    if len(peaks) < 2:
        return

    (i1, h1), (i2, h2) = peaks[0], peaks[1]
    lo, hi = sorted((i1, i2))
    trough = float(np.min(prof[lo:hi + 1]))
    itrough = lo + int(np.argmin(prof[lo:hi + 1]))
    weaker = min(h1, h2)
    sep = float(abs(vv[i1] - vv[i2]))
    contrast = (weaker - trough) / weaker if weaker > 0 else float("nan")
    out.data_peak_separation_kms, out.data_peak_contrast = sep, contrast

    # significance of the dip against the smoothed noise
    dip_sig = float("nan")
    err = result.components.get("err")
    if err is not None:
        e = np.asarray(err, dtype=float)[dsel]
        e = e[np.isfinite(e) & (e > 0)]
        if e.size:
            k = max(int(round(DATA_SMOOTH_KMS / max(
                float(np.median(np.abs(np.diff(vv)))), 1e-6))), 1)
            sigma_smoothed = float(np.median(e)) / math.sqrt(k)
            if sigma_smoothed > 0:
                dip_sig = (weaker - trough) / sigma_smoothed
    out.data_dip_significance = dip_sig

    out.data_double_peaked = bool(
        _finite(out.snr) and out.snr >= DATA_DOUBLE_PEAK_MIN_SNR
        and DATA_DOUBLE_PEAK_MIN_SEP_KMS <= sep <= DATA_DOUBLE_PEAK_MAX_SEP_KMS
        and _finite(contrast) and contrast >= DATA_DOUBLE_PEAK_MIN_CONTRAST
        # comparable horns: a lopsided pair is a wing bump, not a disk profile
        and weaker >= DATA_PEAK_MIN_HEIGHT_RATIO * max(h1, h2)
        # the dip straddles systemic, as a rotating-disk profile must
        and abs(float(vv[itrough])) <= DATA_TROUGH_MAX_OFFSET_KMS
        # and it must be deeper than the noise can fake
        and _finite(dip_sig) and dip_sig >= DATA_DIP_MIN_SIGNIFICANCE)


# ---------------------------------------------------------------------------
# Population classification
# ---------------------------------------------------------------------------

def _narrow_flux(result, line: str, window_kms: float = 3000.0) -> float:
    model = result.line_models.get(line)
    lam0 = REST_WAVE.get(line)
    if model is None or lam0 is None:
        return float("nan")
    w = np.asarray(result.rest_wave, dtype=float)
    v = velocity_grid(w, lam0)
    sel = np.abs(v) <= window_kms
    if sel.sum() < 3:
        return float("nan")
    return float(np.trapezoid(np.clip(np.asarray(model, dtype=float)[sel], 0.0, None),
                              w[sel]))


def classify(result, hb: BroadProfile, ha: BroadProfile,
             fe: FeIIStrength, oiii: OIIIOutflow, derived=None) -> dict:
    """Boolean population flags. Each is a *candidate* label, not a certainty.

    The flags are deliberately computed from quantities already validated by the
    engine (FWHM, fluxes, host fraction) so that a campaign can select on them
    without a second modelling step.
    """
    flags: dict[str, bool] = {}

    hb_det, ha_det = hb.detected, ha.detected
    flags["broad_hbeta_detected"] = hb_det
    flags["broad_halpha_detected"] = ha_det

    # Type 1.9: broad Halpha but no broad Hbeta (reddened or intrinsically weak)
    flags["type1_9"] = bool(ha_det and not hb_det)

    # NLS1: narrow broad-Hbeta and weak [O III] relative to Hbeta
    oiii_hb = float("nan")
    hb_narrow = _narrow_flux(result, "Hb")
    if _finite(oiii.flux) and _finite(hb_narrow) and hb_narrow > 0:
        oiii_hb = oiii.flux / hb_narrow
    flags["nls1"] = bool(hb_det and _finite(hb.fwhm_kms)
                         and hb.fwhm_kms < NLS1_FWHM_KMS
                         and (not _finite(oiii_hb) or oiii_hb < NLS1_OIII_HB_MAX))

    flags["strong_feii"] = fe.strong_feii
    flags["outflow_candidate"] = oiii.outflow
    flags["extreme_outflow"] = oiii.extreme_outflow
    # the data-based test is the real selector; the model-based one almost never
    # fires (see `broad_profile`), so it only ever adds candidates here
    flags["double_peaked"] = bool(hb.data_double_peaked or ha.data_double_peaked
                                  or hb.double_peaked or ha.double_peaked)
    flags["offset_broad_line"] = bool(hb.offset or ha.offset)

    meas = result.lines.get("Hb")
    flags["weak_broad_hbeta"] = bool(hb_det and meas is not None
                                     and _finite(meas.ew_aa)
                                     and meas.ew_aa < WEAK_BROAD_EW_AA)

    host = result.continuum.get("frac_host_5100", result.continuum.get("frac_host"))
    flags["host_dominated"] = bool(_finite(host) and host > HOST_DOMINATED_FRAC)

    if derived is not None and _finite(getattr(derived, "eddington_ratio", None)):
        flags["high_eddington"] = bool(derived.eddington_ratio > 0.3)
        flags["low_eddington"] = bool(derived.eddington_ratio < 0.01)

    return {"flags": flags, "oiii_hbeta_ratio": _nan_to_none(oiii_hb)}


# ---------------------------------------------------------------------------
# One-call report
# ---------------------------------------------------------------------------

@dataclass
class ScienceReport:
    oiii: OIIIOutflow = field(default_factory=OIIIOutflow)
    feii: FeIIStrength = field(default_factory=FeIIStrength)
    hbeta: BroadProfile = field(default_factory=BroadProfile)
    halpha: BroadProfile = field(default_factory=BroadProfile)
    flags: dict = field(default_factory=dict)
    oiii_hbeta_ratio: float | None = None

    @property
    def active_flags(self) -> list:
        return sorted(k for k, v in self.flags.items() if v)

    def to_dict(self) -> dict:
        return {
            "oiii": self.oiii.to_dict(),
            "feii": self.feii.to_dict(),
            "hbeta": self.hbeta.to_dict(),
            "halpha": self.halpha.to_dict(),
            "flags": dict(self.flags),
            "oiii_hbeta_ratio": _nan_to_none(self.oiii_hbeta_ratio),
        }

    def summary(self) -> str:
        bits = []
        if _finite(self.oiii.w80_kms):
            bits.append(f"[OIII] W80={self.oiii.w80_kms:.0f} v50={self.oiii.v50_kms:+.0f} km/s")
        if _finite(self.feii.r_feii):
            bits.append(f"R_FeII={self.feii.r_feii:.2f}")
        if _finite(self.hbeta.fwhm_kms):
            bits.append(f"FWHM(Hb)={self.hbeta.fwhm_kms:.0f} km/s")
        af = self.active_flags
        bits.append("flags: " + (", ".join(af) if af else "none"))
        return " | ".join(bits)


def science_report(result, derived=None,
                   inst_fwhm_kms: float = SDSS_INST_FWHM_KMS) -> ScienceReport:
    """Run every science measurement on one decomposition."""
    oiii = oiii_outflow(result, inst_fwhm_kms=inst_fwhm_kms)
    fe = feii_strength(result)
    hb = broad_profile(result, "Hb")
    ha = broad_profile(result, "Ha")
    cls = classify(result, hb, ha, fe, oiii, derived=derived)
    return ScienceReport(oiii=oiii, feii=fe, hbeta=hb, halpha=ha,
                         flags=cls["flags"], oiii_hbeta_ratio=cls["oiii_hbeta_ratio"])
