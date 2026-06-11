"""Visual-first continuum planning: anchor the AGN power law *before* the fit.

Why this exists
---------------
Egent (the stellar pipeline) can trust its continuum because a stellar continuum
is a *constrained inversion*: Teff/logg/[Fe/H] feed a synthetic atmosphere that
predicts the continuum shape. An AGN "continuum" has no such forward model -- it
is a superposition of physically distinct emitters (disk power law + Balmer
continuum + the Fe II pseudo-continuum + host starlight) with no single physical
law tying the shape together. The decomposition is therefore under-determined and
*prior-dependent*, and it is exactly the place where a human expert eyeballs the
spectrum first. This module injects that human structural prior.

How
---
1. Lay down candidate *continuum windows* -- rest-frame regions believed to be
   relatively line-free -- and mark which sit inside the Fe II forest.
2. A planner (deterministic :class:`RuleContinuumPlanner` here, or a vision LLM
   in :mod:`agn_egent.agent.continuum_planner`) chooses which windows to trust.
3. Robustly fit a pure power law ``f_lambda = norm * (lam/3000)^slope`` through the
   median flux of the trusted windows -- the classic lower-envelope continuum.
4. :func:`apply_plan_to_config` turns that judgement into *soft constraints* on
   the deterministic PyQSOFit fit (PL slope init + narrowed bounds, restricted
   windows). The LLM/heuristic makes the structural call; the math stays
   deterministic.

The Fe II trap: a naive "draw a line through the bottom" pass absorbs the
4434-4684 Fe II blend into the continuum and under-fits Fe II. The candidate
windows below carry an ``fe`` flag and the rule planner down-weights them; the
vision planner is told to reason about it explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .backends.qsopar_config import QsoparConfig


# -- candidate continuum windows ----------------------------------------------
# Rest-frame [A] regions used to anchor the optical continuum. ``fe`` marks
# windows that fall inside the optical Fe II pseudo-continuum (strong in Fe-loud
# AGN); they are usable but contaminated, so the planner treats them with care.
@dataclass(frozen=True)
class Window:
    lo: float
    hi: float
    label: str
    fe: bool = False          # sits in the Fe II forest
    balmer: bool = False      # blueward Balmer-continuum / small-blue-bump zone

    @property
    def center(self) -> float:
        return 0.5 * (self.lo + self.hi)


CANDIDATE_WINDOWS: list[Window] = [
    Window(3625., 3645., "pre-Balmer", balmer=True),
    Window(3775., 3832., "3800"),
    Window(4000., 4050., "4025"),
    Window(4150., 4280., "4200"),          # classic clean blue anchor
    Window(4435., 4640., "4500 (FeII)", fe=True),
    Window(5080., 5110., "5100"),          # the L5100 anchor itself
    Window(5100., 5535., "5300 (FeII)", fe=True),
    Window(5600., 5630., "5600"),
    Window(6005., 6035., "6020"),
    Window(6110., 6250., "6180"),          # clean, between Hbeta and Halpha
    Window(6800., 7000., "6900"),          # clean, redward of Halpha
]


@dataclass
class WindowStat:
    """Per-window summary the planner reasons over."""
    window: Window
    center: float
    median: float            # median flux in the window
    scatter: float           # robust fractional scatter (MAD/median)
    npix: int
    covered: bool
    used: bool = True        # selected for the anchored fit
    excess: float = np.nan   # fractional flux above the anchored PL (Fe/line)

    def to_dict(self) -> dict:
        return {"label": self.window.label, "lo": self.window.lo,
                "hi": self.window.hi, "fe": self.window.fe,
                "center": round(self.center, 1), "median": round(self.median, 3),
                "scatter": round(self.scatter, 3), "npix": self.npix,
                "used": self.used,
                "excess": None if not np.isfinite(self.excess) else round(self.excess, 3)}


def window_stats(rest_wave, flux, windows=CANDIDATE_WINDOWS,
                 min_pix: int = 4) -> list[WindowStat]:
    """Median/scatter/coverage of each candidate window over the spectrum."""
    rw = np.asarray(rest_wave, dtype=float)
    fx = np.asarray(flux, dtype=float)
    out = []
    for win in windows:
        m = (rw >= win.lo) & (rw <= win.hi) & np.isfinite(fx)
        n = int(m.sum())
        if n < min_pix:
            out.append(WindowStat(win, win.center, np.nan, np.nan, n, False, used=False))
            continue
        f = fx[m]
        med = float(np.median(f))
        mad = float(np.median(np.abs(f - med)))
        scatter = (1.4826 * mad / med) if med > 0 else np.inf
        cen = float(np.median(rw[m]))
        out.append(WindowStat(win, cen, med, scatter, n, True))
    return out


# -- the anchored power-law fit ------------------------------------------------
def anchored_pl_fit(stats: list[WindowStat]) -> tuple[float, float, list[WindowStat]]:
    """Robust ``f_lambda = norm*(lam/3000)^slope`` through the *used* windows.

    Linear least squares in log-log space on (window center, window median). The
    PyQSOFit power law uses the same parametrisation, so ``slope``/``norm`` map
    straight onto its ``PL_slope``/``PL_norm`` priors.
    """
    pts = [s for s in stats if s.used and s.covered and s.median > 0]
    if len(pts) < 2:
        return np.nan, np.nan, pts
    x = np.log(np.array([s.center for s in pts]) / 3000.0)
    y = np.log(np.array([s.median for s in pts]))
    slope, intercept = np.polyfit(x, y, 1)
    norm = float(np.exp(intercept))
    return float(slope), norm, pts


def _pl_at(slope: float, norm: float, lam) -> np.ndarray:
    return norm * (np.asarray(lam, dtype=float) / 3000.0) ** slope


# -- the plan ------------------------------------------------------------------
@dataclass
class ContinuumPlan:
    """A structural decision about the continuum, ready to constrain the fit."""
    slope: float                       # anchored power-law slope (f_lambda)
    norm: float                        # anchored PL_norm at 3000 A
    used_windows: list[tuple]          # (lo, hi) actually anchored on
    slope_lo: float                    # narrowed PL_slope lower bound
    slope_hi: float                    # narrowed PL_slope upper bound
    host: bool                         # keep host decomposition on?
    classification: str                # 'blue_pl' | 'host_dominated' | 'fe_loud' | 'normal'
    rationale: str = ""
    source: str = "rule"
    stats: list = field(default_factory=list)   # list[WindowStat] for provenance

    def to_dict(self) -> dict:
        return {"slope": round(self.slope, 3), "norm": round(self.norm, 4),
                "slope_bounds": [round(self.slope_lo, 3), round(self.slope_hi, 3)],
                "used_windows": [list(w) for w in self.used_windows],
                "host": self.host, "classification": self.classification,
                "source": self.source, "rationale": self.rationale,
                "windows": [s.to_dict() for s in self.stats]}


# physical optical AGN slope range; the planner pins the fit to a narrow band
# *inside* this, but never lets the band escape it.
SLOPE_FLOOR = -3.0
SLOPE_CEIL = 0.0
SLOPE_HALFWIDTH = 0.5      # +/- band around the anchored slope


def _classify(slope: float, stats: list[WindowStat]) -> str:
    fe = [s for s in stats if s.window.fe and s.covered and np.isfinite(s.excess)]
    fe_loud = bool(fe) and float(np.median([s.excess for s in fe])) > 0.20
    if slope >= -0.3:
        return "host_dominated"     # ~flat/rising f_lambda => starlight wins
    if fe_loud:
        return "fe_loud"
    if slope <= -1.8:
        return "blue_pl"
    return "normal"


def make_plan(slope: float, norm: float, stats: list[WindowStat],
              source: str = "rule", rationale: str = "",
              host: bool | None = None) -> ContinuumPlan:
    """Assemble a :class:`ContinuumPlan` from an anchored fit + window stats."""
    slope = float(np.clip(slope, SLOPE_FLOOR, SLOPE_CEIL))
    lo = max(SLOPE_FLOOR, slope - SLOPE_HALFWIDTH)
    hi = min(SLOPE_CEIL, slope + SLOPE_HALFWIDTH)
    cls = _classify(slope, stats)
    if host is None:
        host = cls == "host_dominated"
    used = [(s.window.lo, s.window.hi) for s in stats if s.used and s.covered]
    return ContinuumPlan(slope=slope, norm=norm, used_windows=used,
                         slope_lo=lo, slope_hi=hi, host=host,
                         classification=cls, rationale=rationale,
                         source=source, stats=stats)


# -- deterministic baseline planner -------------------------------------------
class RuleContinuumPlanner:
    """Iterative lower-envelope window selection -- the offline baseline.

    Fit a PL through all covered windows, drop the windows lying most *above* it
    (Fe II/line contamination pushes a window up, never down), refit, repeat.
    Fe-flagged windows must clear a tighter excess tolerance to survive.
    """

    name = "rule"

    def __init__(self, excess_tol: float = 0.08, fe_excess_tol: float = 0.03,
                 scatter_tol: float = 0.25, max_iter: int = 4, min_windows: int = 3):
        self.excess_tol = excess_tol
        self.fe_excess_tol = fe_excess_tol
        self.scatter_tol = scatter_tol
        self.max_iter = max_iter
        self.min_windows = min_windows

    def plan(self, spectrum, result=None) -> ContinuumPlan:
        rw = spectrum.rest_wave
        # prefer host-subtracted QSO flux if a prior fit gave us one, else raw
        flux = spectrum.flux
        if result is not None and "qso" in result.components:
            rw = result.rest_wave
            flux = result.components["qso"]
        stats = window_stats(rw, flux)

        # drop high-scatter (line-contaminated) windows up front
        for s in stats:
            if s.covered and s.scatter > self.scatter_tol:
                s.used = False

        for _ in range(self.max_iter):
            slope, norm, pts = anchored_pl_fit(stats)
            if not np.isfinite(slope):
                break
            worst, worst_excess = None, 0.0
            for s in stats:
                if not (s.covered and s.median > 0):
                    continue
                s.excess = s.median / float(_pl_at(slope, norm, s.center)) - 1.0
                if not s.used:
                    continue
                tol = self.fe_excess_tol if s.window.fe else self.excess_tol
                if s.excess > tol and s.excess > worst_excess:
                    worst, worst_excess = s, s.excess
            n_used = sum(1 for s in stats if s.used and s.covered)
            if worst is None or n_used <= self.min_windows:
                break
            worst.used = False

        slope, norm, _ = anchored_pl_fit(stats)
        n = sum(1 for s in stats if s.used and s.covered)
        rationale = (f"anchored PL on {n} clean window(s); slope={slope:.2f}. "
                     "dropped Fe II/line-contaminated windows lying above the "
                     "lower envelope.")
        return make_plan(slope, norm, stats, source=self.name, rationale=rationale)


# -- apply a plan to the fit config -------------------------------------------
def apply_plan_to_config(config: QsoparConfig, plan: ContinuumPlan) -> QsoparConfig:
    """Turn a continuum plan into soft constraints on the deterministic fit."""
    new = config
    if np.isfinite(plan.slope):
        new = new.set_conti_param("PL_slope", initial=plan.slope,
                                  vmin=plan.slope_lo, vmax=plan.slope_hi)
    if np.isfinite(plan.norm) and plan.norm > 0:
        new = new.set_conti_param("PL_norm", initial=plan.norm)
    if plan.used_windows:
        new = new.set_conti_windows(plan.used_windows)
    new = new.set_fit_option("decompose_host", bool(plan.host))
    return new


# -- planning figure (vision input) -------------------------------------------
def render_continuum_plan(spectrum, plan: ContinuumPlan, path: str,
                          result=None) -> str:
    """Render the raw spectrum + candidate windows + anchored PL for review.

    This is the image the vision planner looks at: it shows where the windows
    fall, which were trusted vs. rejected, and the resulting power law.
    """
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    rw = spectrum.rest_wave
    flux = spectrum.flux
    if result is not None and "qso" in result.components:
        rw = result.rest_wave
        flux = result.components["qso"]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(rw, flux, color="0.25", lw=0.7, drawstyle="steps-mid", label="data")

    for s in plan.stats:
        if not s.covered:
            continue
        color = "tab:green" if s.used else "tab:red"
        if s.window.fe:
            color = "tab:orange" if s.used else "tab:red"
        ax.axvspan(s.window.lo, s.window.hi, color=color, alpha=0.18)
        ax.plot([s.center], [s.median], "o", color=color, ms=5,
                mec="k", mew=0.4, zorder=5)

    if np.isfinite(plan.slope):
        grid = np.linspace(rw.min(), rw.max(), 400)
        ax.plot(grid, _pl_at(plan.slope, plan.norm, grid), color="crimson",
                lw=1.6, label=f"anchored PL (slope={plan.slope:.2f})")

    finite = np.isfinite(flux)
    if finite.any():
        lo, hi = np.percentile(flux[finite], [1, 99])
        pad = 0.2 * (hi - lo)
        ax.set_ylim(lo - pad, hi + pad)
    ax.set_title(f"{spectrum.name}   z={spectrum.z:.4f}   "
                 f"class={plan.classification}   [{plan.source}]", fontsize=10)
    ax.set_xlabel(r"Rest wavelength [\AA]", fontsize=9)
    ax.set_ylabel(r"$f_\lambda$  [$10^{-17}$ cgs]", fontsize=9)
    ax.legend(fontsize=8, loc="upper right")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path
