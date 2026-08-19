"""Residual-based anomaly scoring: find the objects the model cannot describe.

The science flags in :mod:`agn_egent.science` select objects by what the fit
*found*. This module selects by what the fit *missed*. After the accepted fit,
an object whose residuals still carry coherent structure inside a line window is
either badly fit or genuinely unusual -- an extra kinematic component, a
double-peaked profile the Gaussians smoothed over, broad absorption, a
mis-subtracted host. Both cases are worth a look, and neither is visible in the
best-fit parameters.

The discriminant is *coherence*, not size. Pull (residual / error) from pure
noise averages down as 1/sqrt(N) when smoothed; a real feature spanning many
pixels does not. So we smooth the pull over a velocity kernel comparable to a
narrow line and compare the result against the sqrt(N) expectation. A score of
1 means "consistent with noise"; 3 means the coherent residual is three times
what noise alone produces.

This is deterministic and costs nothing beyond the fit, so it can rank an entire
survey. It is a *ranking* statistic for human or LLM review, not a detection
claim: a high score says "this fit does not describe this spectrum", and the
diagnostic plot says why.

Measured calibration (white-noise residuals, 15 realizations, SDSS sampling):
score = 1.10 +/- 0.11, max 1.32, and it does not drift with pixel scale.
Injected broad structure climbs clear of that and keeps climbing: a 3000 km/s
bump at 1, 2 and 4 sigma scores 1.7, 3.0 and 6.0. Two deliberate
insensitivities are worth knowing about -- a residual that is merely *large*
but incoherent (pull rms 3) still scores ~1, because amplitude is already
reduced chi^2's job; and an unresolved spike (a cosmic ray or sky residual)
scores ~1.5, so single-pixel artefacts do not dominate a survey ranking.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict

import numpy as np

from .science import C_KMS, REST_WAVE, _nan_to_none

# Windows scored independently, as (name, rest-frame lo, hi) in Angstroms.
# Line regions where a missed component actually means something, plus a
# continuum control window that should be featureless if the model is right.
DEFAULT_WINDOWS = (
    ("Hb", 4700.0, 5100.0),      # broad Hbeta + [O III]
    ("Ha", 6400.0, 6800.0),      # broad Halpha + [N II] + [S II]
    ("MgII", 2700.0, 2900.0),    # only covered at higher z
    ("continuum", 5600.0, 6200.0),   # line-free control
)

# Smoothing kernel: ~600 km/s is wider than the instrumental resolution and than
# a typical narrow line, so unresolved features are not counted as structure,
# while broad-line-scale structure survives.
SMOOTH_KMS = 600.0

# Score above which an object is worth eyeballing. Calibrated below in
# `interpret`: pure noise gives ~1.0, and real structure climbs quickly.
NOTABLE_SCORE = 2.0
STRONG_SCORE = 3.0

# Minimum usable pixels in a window before its score means anything.
MIN_PIXELS = 20


@dataclass
class WindowScore:
    name: str
    n_pixels: int = 0
    score: float = float("nan")        # coherent residual / noise expectation
    pull_rms: float = float("nan")     # raw rms of residual/err (~1 if perfect)
    max_run: int = 0                   # longest same-sign run of pull
    run_score: float = float("nan")    # that run vs the random-walk expectation
    worst_wave: float = float("nan")   # rest wavelength of the strongest excursion
    worst_velocity_kms: float = float("nan")   # ... as a velocity, if a line window

    def to_dict(self) -> dict:
        return {k: _nan_to_none(v) for k, v in asdict(self).items()}


@dataclass
class AnomalyReport:
    score: float = float("nan")        # worst line-window score
    worst_window: str = ""
    windows: dict = field(default_factory=dict)
    notable: bool = False
    strong: bool = False
    continuum_score: float = float("nan")   # control: high here means a bad continuum

    def to_dict(self) -> dict:
        return {
            "score": _nan_to_none(self.score),
            "worst_window": self.worst_window,
            "notable": self.notable,
            "strong": self.strong,
            "continuum_score": _nan_to_none(self.continuum_score),
            "windows": {k: v.to_dict() for k, v in self.windows.items()},
        }

    def summary(self) -> str:
        if not math.isfinite(self.score):
            return "anomaly: not scored"
        tag = "STRONG" if self.strong else ("notable" if self.notable else "ordinary")
        return (f"anomaly score {self.score:.2f} in {self.worst_window} [{tag}] "
                f"(continuum control {self.continuum_score:.2f})")


def _smooth(y: np.ndarray, npix: int) -> np.ndarray:
    """Boxcar smooth, edge-safe (reflected padding keeps window ends usable)."""
    if npix <= 1 or y.size < 3:
        return y.astype(float)
    npix = min(npix, y.size if y.size % 2 else y.size - 1)
    if npix % 2 == 0:
        npix -= 1
    if npix <= 1:
        return y.astype(float)
    pad = npix // 2
    padded = np.pad(y.astype(float), pad, mode="reflect")
    kern = np.ones(npix) / npix
    return np.convolve(padded, kern, mode="valid")


def _longest_run(sign: np.ndarray) -> int:
    """Longest run of a constant sign (a cheap coherence test)."""
    best = run = 0
    prev = 0
    for s in sign:
        if s != 0 and s == prev:
            run += 1
        else:
            run = 1 if s != 0 else 0
        prev = s
        best = max(best, run)
    return int(best)


# If the pixel-to-pixel scatter is a smaller fraction of the total residual
# scatter than this, the "noise" estimate is not measuring noise -- the spectrum
# is oversampled or interpolated, or its error array is wrong. Dividing by it
# produces a meaningless enormous score (a pilot run reached 1e8), so such a
# window is reported as unscored instead.
MIN_WHITE_FRACTION = 0.01


def _white_noise_level(p: np.ndarray) -> float:
    """Pixel-to-pixel noise amplitude of a series, immune to smooth structure.

    Successive differences cancel anything varying slowly across a pixel, so
    their scatter measures the white component alone: for white noise of
    dispersion s, std(diff) = s*sqrt(2). Uses the MAD so a handful of spikes
    (cosmic rays, sky residuals) do not inflate the estimate.
    """
    d = np.diff(p)
    if d.size < 2:
        return float("nan")
    mad = float(np.median(np.abs(d - np.median(d))))
    s = 1.4826 * mad / math.sqrt(2.0)
    if s > 0:
        return s
    s = float(np.std(d)) / math.sqrt(2.0)       # fall back if MAD degenerates
    return s if s > 0 else float("nan")


def score_window(wave: np.ndarray, pull: np.ndarray, lo: float, hi: float,
                 name: str, smooth_kms: float = SMOOTH_KMS) -> WindowScore:
    """Coherent-residual score inside one rest-frame window.

    Smoothing white noise over k pixels lowers its dispersion by sqrt(k). So we
    compare the smoothed pull against that expectation -- but the expectation
    must be built from the *white* part of the residual, measured from
    pixel-to-pixel differences, not from the total residual scatter.

    Normalizing by the total scatter instead (the obvious choice) silently caps
    the statistic at sqrt(k): a perfectly coherent residual smooths to its own
    rms, and the ratio can never exceed the square root of the kernel. On real
    SDSS sampling k is about 9 pixels, so every object piles up against a
    ceiling of 3 and the ranking carries no information. Dividing by the white
    level removes the ceiling: the score is unbounded above and reads directly
    as "this residual is N times more coherent than photon noise allows".
    """
    ws = WindowScore(name=name)
    sel = (wave >= lo) & (wave <= hi) & np.isfinite(pull)
    n = int(sel.sum())
    ws.n_pixels = n
    if n < MIN_PIXELS:
        return ws
    w, p = wave[sel], pull[sel]

    ws.pull_rms = float(np.sqrt(np.mean(p ** 2)))

    # pixels per smoothing kernel, from the local dispersion
    dlam = float(np.median(np.diff(w)))
    lam_mid = float(np.median(w))
    if not (dlam > 0 and lam_mid > 0):
        return ws
    kernel_pix = max(int(round((smooth_kms / C_KMS) * lam_mid / dlam)), 1)

    sm = _smooth(p, kernel_pix)
    eff = min(kernel_pix, n)
    white = _white_noise_level(p)
    usable = (math.isfinite(white) and white > 0
              and ws.pull_rms > 0 and white >= MIN_WHITE_FRACTION * ws.pull_rms)
    if usable:
        observed = float(np.sqrt(np.mean(sm ** 2)))
        ws.score = observed / (white / math.sqrt(eff))

    ws.max_run = _longest_run(np.sign(p))
    # expected longest same-sign run in n coin flips is ~log2(n)
    exp_run = max(math.log2(n), 1.0)
    ws.run_score = ws.max_run / exp_run

    i = int(np.argmax(np.abs(sm)))
    ws.worst_wave = float(w[i])
    lam0 = REST_WAVE.get(name)
    if lam0:
        ws.worst_velocity_kms = C_KMS * (ws.worst_wave - lam0) / lam0
    return ws


def anomaly_score(result, windows=DEFAULT_WINDOWS,
                  smooth_kms: float = SMOOTH_KMS) -> AnomalyReport:
    """Rank one decomposition by how much coherent residual it leaves behind.

    Only windows actually covered by the spectrum are scored. The overall score
    is the worst *line* window; the continuum window is reported separately as a
    control, because a high continuum score points at the continuum model rather
    than at anything interesting about the lines.
    """
    rep = AnomalyReport()
    wave = np.asarray(result.rest_wave, dtype=float)
    resid = result.components.get("residual")
    err = result.components.get("err")
    if resid is None or err is None or wave.size < MIN_PIXELS:
        return rep
    resid = np.asarray(resid, dtype=float)
    err = np.asarray(err, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        pull = np.where(np.isfinite(err) & (err > 0), resid / err, np.nan)

    best_name, best_score = "", float("nan")
    for name, lo, hi in windows:
        if wave.max() < lo or wave.min() > hi:
            continue                      # window not covered by this spectrum
        ws = score_window(wave, pull, lo, hi, name, smooth_kms=smooth_kms)
        if ws.n_pixels < MIN_PIXELS:
            continue
        rep.windows[name] = ws
        if name == "continuum":
            rep.continuum_score = ws.score
            continue
        if math.isfinite(ws.score) and not (math.isfinite(best_score) and best_score >= ws.score):
            best_name, best_score = name, ws.score

    rep.score, rep.worst_window = best_score, best_name
    if math.isfinite(best_score):
        rep.notable = bool(best_score >= NOTABLE_SCORE)
        rep.strong = bool(best_score >= STRONG_SCORE)
    return rep
