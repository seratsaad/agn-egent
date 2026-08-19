"""Render a decomposition to a diagnostic PNG for the agent's vision input.

Purpose-built for line-decomposition QC: one *zoomed* column per emission-line
complex (Hbeta, Halpha), each with a flux panel (data + every fitted component:
continuum, Fe II, host, broad, narrow, total model) and a pull panel
(residual / error) with +/-3 sigma bands. The view is restricted to the line
region so the broad/narrow deblending and the residual structure at the line are
actually resolvable -- not a handful of pixels in a full-spectrum plot. The
flagged complexes from a VerdictReport are annotated.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .backends.base import DecompositionResult

# Colorblind-safe palette (Okabe-Ito), with well-separated hues for the
# components most often co-plotted. The continuum is always the orange power law.
COLORS = {
    "data": "#444444",
    "model": "#D55E00",       # vermillion -- the total model
    "pl": "#E69F00",          # orange     -- continuum (power law)
    "feii_op": "#CC79A7",     # reddish purple -- Fe II (optical)
    "feii_uv": "#56B4E9",     # sky blue       -- Fe II (UV)
    "host": "#999999",        # grey           -- host galaxy
    "broad": "#0072B2",       # blue           -- broad lines
    "narrow": "#009E73",      # bluish green    -- narrow lines
}

# Rest-frame zoom window per line complex: wide enough to show the broad wings
# and the immediate continuum on both sides, tight enough that the line profile
# fills the panel. Hbeta covers He II 4687 -> [O III] 5007; Halpha covers
# [N II] 6548/6583 and [S II] 6716/6731 around Halpha.
ZOOM = {
    "Hb": (4600., 5120.),
    "Ha": (6300., 6800.),
}
_ORDER = ["Hb", "Ha"]


def _present_complexes(result) -> list[str]:
    """Which zoom windows actually have data coverage, in canonical order."""
    w = result.rest_wave
    out = []
    for name in _ORDER:
        lo, hi = ZOOM[name]
        m = (w >= lo) & (w <= hi)
        if np.any(m & np.isfinite(result.components.get("data", w))):
            out.append(name)
    return out


def render_diagnostic(result: DecompositionResult, path: str,
                      report=None) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    w = result.rest_wave
    c = result.components
    data = c.get("data")
    model = c.get("model")

    cols = _present_complexes(result) or ["Hb"]
    ncol = len(cols)

    flagged = set()
    if report is not None:
        flagged = {ck.target for ck in report.checks
                   if ck.target and ck.status >= 1 and ck.target in result.lines}

    fig, axes = plt.subplots(
        2, ncol, figsize=(6.0 * ncol, 6.0), squeeze=False, sharex="col",
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05, "wspace": 0.22})

    comp_specs = [
        ("pl", "continuum (power law)", 1.6, "-"),
        ("feii_op", "Fe II", 1.1, "-"),
        ("feii_uv", "Fe II (UV)", 1.0, "-"),
        ("host", "host", 1.0, "-"),
        ("broad", "broad", 1.4, "-"),
        ("narrow", "narrow", 1.1, "-"),
    ]

    for j, name in enumerate(cols):
        lo, hi = ZOOM[name]
        win = (w >= lo) & (w <= hi)
        ax, axr = axes[0, j], axes[1, j]

        # -- flux panel --------------------------------------------------------
        ax.plot(w, data, color=COLORS["data"], lw=0.8, drawstyle="steps-mid",
                label="data")
        if model is not None:
            ax.plot(w, model, color=COLORS["model"], lw=1.5, label="model")
        for key, lbl, lw, ls in comp_specs:
            if key in c:
                ax.plot(w, c[key], color=COLORS[key], lw=lw, ls=ls, alpha=0.9,
                        label=lbl)
        ax.set_xlim(lo, hi)
        dwin = data[win] if data is not None else None
        if dwin is not None and np.isfinite(dwin).any():
            ylo, yhi = np.nanpercentile(dwin[np.isfinite(dwin)], [1, 99])
            pad = 0.18 * (yhi - ylo + 1e-9)
            ax.set_ylim(ylo - pad, yhi + pad)
        ax.set_title(f"{name} region", fontsize=11)
        if j == 0:
            ax.set_ylabel(r"$f_\lambda$  [$10^{-17}$ erg s$^{-1}$ cm$^{-2}$ "
                          r"$\mathrm{\AA}^{-1}$]", fontsize=9)
            ax.legend(fontsize=7, ncol=2, loc="upper right", framealpha=0.7)

        # annotate flagged complexes belonging to this region
        for comp in flagged:
            m = result.lines[comp]
            if np.isfinite(m.peak_wave) and lo <= m.peak_wave <= hi:
                ax.axvline(m.peak_wave, color="red", ls=":", lw=0.8, alpha=0.5)
                ax.text(m.peak_wave, ax.get_ylim()[1], f" {comp}!",
                        color="red", fontsize=8, va="top")

        # -- pull panel --------------------------------------------------------
        if "residual" in c and "err" in c:
            err = c["err"]
            good = np.isfinite(err) & (err > 0)
            pull = np.full_like(w, np.nan)
            pull[good] = c["residual"][good] / err[good]
            axr.plot(w, pull, color="0.3", lw=0.7, drawstyle="steps-mid")
            for y in (-3, 0, 3):
                axr.axhline(y, color="red" if y else "0.5",
                            ls="--" if y else "-", lw=0.6, alpha=0.6)
            axr.set_ylim(-8, 8)
            if j == 0:
                axr.set_ylabel("pull", fontsize=9)
        axr.set_xlim(lo, hi)
        axr.set_xlabel(r"Rest wavelength [$\mathrm{\AA}$]", fontsize=9)

    title = f"{result.name}   z={result.z:.4f}   backend={result.backend}"
    if report is not None:
        title += f"   verdict={report.overall!s}"
    fig.suptitle(title, fontsize=10, y=0.99)

    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path
