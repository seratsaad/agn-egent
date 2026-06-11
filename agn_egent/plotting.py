"""Render a decomposition to a diagnostic PNG for the agent's vision input.

Purpose-built for QC: a top panel showing the data and every fitted component
(continuum, Fe II, host, broad, narrow, total model), and a pull panel
(residual / error) with +/-3 sigma bands so correlated structure is obvious.
The flagged complexes from a VerdictReport are annotated.
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


def render_diagnostic(result: DecompositionResult, path: str,
                      report=None) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    w = result.rest_wave
    c = result.components
    data = c.get("data")
    model = c.get("model")

    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(11, 6.5), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05})

    # -- top: data + components ------------------------------------------------
    ax.plot(w, data, color=COLORS["data"], lw=0.7, drawstyle="steps-mid", label="data")
    if model is not None:
        ax.plot(w, model, color=COLORS["model"], lw=1.4, label="model")
    # The continuum IS the power law (pure, by construction) -- nothing else.
    # Fe II, host and the lines are separate additive components, drawn on their
    # own so the continuum can never be read as anything but a power law.
    for key, lbl, lw, ls in [
        ("pl", "continuum (power law)", 1.6, "-"),
        ("feii_op", "Fe II", 1.1, "-"),
        ("feii_uv", "Fe II (UV)", 1.0, "-"),
        ("host", "host", 1.0, "-"),
        ("broad", "broad", 1.2, "-"),
        ("narrow", "narrow", 1.0, "-"),
    ]:
        if key in c:
            ax.plot(w, c[key], color=COLORS[key], lw=lw, ls=ls, alpha=0.9, label=lbl)

    title = f"{result.name}   z={result.z:.4f}   backend={result.backend}"
    if report is not None:
        title += f"   verdict={report.overall!s}"
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(r"$f_\lambda$  [$10^{-17}$ erg s$^{-1}$ cm$^{-2}$ \AA$^{-1}$]",
                  fontsize=9)
    ax.legend(fontsize=7, ncol=3, loc="upper right")
    finite = np.isfinite(data)
    if finite.any():
        lo, hi = np.percentile(data[finite], [1, 99])
        pad = 0.15 * (hi - lo)
        ax.set_ylim(lo - pad, hi + pad)

    # annotate flagged complexes at their broad-line peak
    if report is not None:
        flagged = {ck.target for ck in report.checks
                   if ck.target and ck.status >= 1 and ck.target in result.lines}
        for comp in flagged:
            m = result.lines[comp]
            if np.isfinite(m.peak_wave):
                ax.axvline(m.peak_wave, color="red", ls=":", lw=0.8, alpha=0.5)
                ax.text(m.peak_wave, ax.get_ylim()[1], f" {comp}!",
                        color="red", fontsize=8, va="top")

    # -- bottom: pull (residual / err) -----------------------------------------
    if "residual" in c and "err" in c:
        err = c["err"]
        good = np.isfinite(err) & (err > 0)
        pull = np.full_like(w, np.nan)
        pull[good] = c["residual"][good] / err[good]
        axr.plot(w, pull, color="0.3", lw=0.6, drawstyle="steps-mid")
        for y in (-3, 0, 3):
            axr.axhline(y, color="red" if y else "0.5",
                        ls="--" if y else "-", lw=0.6, alpha=0.6)
        axr.set_ylim(-8, 8)
        axr.set_ylabel("pull", fontsize=9)
    axr.set_xlabel(r"Rest wavelength [\AA]", fontsize=9)

    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path
