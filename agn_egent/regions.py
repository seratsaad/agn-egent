"""Region-by-region decomposition: fit each line complex with its OWN continuum.

The global decomposition forces a single power law (+ the Fe II forest under Hbeta)
across the whole spectrum, so a continuum error in one place biases the broad-line
wings somewhere else. Here we instead fit each emission-line region independently
on a *local* continuum:

    Hbeta region  ~ rest [4150, 5600]  (Hbeta + [O III] + optical Fe II)
    Halpha region ~ rest [6000, 7000]  (Halpha + [N II] + [S II])

Each region is fit with PyQSOFit restricted to that wavelength range (``wave_range``
-- PyQSOFit's own "local fit" mode), with continuum windows that bracket the lines
on both sides and host decomposition off (a local power law absorbs the locally
~flat starlight). The L5100 luminosity comes from the Hbeta region (5100 A lives
there), so the single-epoch M_BH(Hbeta) is fully determined within one region.

Results are merged back into a single :class:`DecompositionResult` so ``verify``,
``derive`` and the agent loop work unchanged.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from .spectrum import Spectrum
from .pipeline import decompose, RUNS_DIR
from .backends.base import DecompositionResult
from .backends.qsopar_config import QsoparConfig
from .continuum import RuleContinuumPlanner, apply_plan_to_config
from .verify import verify
from .remedies import apply_remedy, describe_remedy
from .agent.inspector import RuleInspector, Decision


@dataclass(frozen=True)
class Region:
    """One contiguous fit region and the line complexes it owns."""
    name: str
    lo: float                 # rest-frame fit range [A]
    hi: float
    complexes: tuple          # compname(s) fit in this region, e.g. ("Hb",)
    fe: bool = True           # fit an optical Fe II pseudo-continuum here?


# Hbeta carries the optical Fe II forest; Halpha is essentially Fe-free.
DEFAULT_OPTICAL_REGIONS: list[Region] = [
    Region("Hb", 4150., 5600., ("Hb",), fe=True),
    Region("Ha", 6000., 7000., ("Ha",), fe=False),
]


def trim_spectrum(spectrum: Spectrum, lo: float, hi: float) -> Spectrum:
    """Return a copy of `spectrum` restricted to rest-frame [lo, hi]."""
    rw = spectrum.rest_wave
    m = (rw >= lo) & (rw <= hi)
    return Spectrum(wave=spectrum.wave[m], flux=spectrum.flux[m], err=spectrum.err[m],
                    z=spectrum.z, name=spectrum.name, ra=spectrum.ra, dec=spectrum.dec,
                    meta=dict(spectrum.meta))


def region_config(config: QsoparConfig, region: Region) -> QsoparConfig:
    """Build a local sub-config: only this region's lines + local continuum."""
    sub = config.copy()
    sub.lines = [l for l in sub.lines if l.compname in region.complexes]
    # keep only continuum windows that fall inside the region (bracket the lines)
    sub.conti_windows = [(lo, hi) for (lo, hi) in config.conti_windows
                         if lo >= region.lo and hi <= region.hi]
    sub = sub.set_fit_option("wave_range", [region.lo, region.hi])
    # host PCA needs the full spectrum; a local power law stands in within a region
    sub = sub.set_fit_option("decompose_host", False)
    sub = sub.set_fit_option("Fe_uv_op", bool(region.fe))
    return sub


class _Step:
    """Minimal history entry for RuleInspector (only needs ``.decision``)."""
    def __init__(self, decision):
        self.decision = decision


def _region_anchor(spectrum, sub, region, backend, rdir):
    """Anchor the local power-law continuum on the region's line-free windows."""
    base = decompose(spectrum, config=sub, backend=backend, make_figure=False,
                     workdir=os.path.join(rdir, "base"))
    sub_spec = trim_spectrum(spectrum, region.lo, region.hi)
    plan = RuleContinuumPlanner().plan(sub_spec, result=base)
    sub = apply_plan_to_config(sub, plan)
    # apply_plan flips host on by default; a region always stays host-off + local
    return (sub.set_fit_option("decompose_host", False)
               .set_fit_option("wave_range", [region.lo, region.hi]))


def decompose_by_region(spectrum: Spectrum,
                        config: QsoparConfig | None = None,
                        regions: list[Region] = DEFAULT_OPTICAL_REGIONS,
                        backend: str = "pyqsofit",
                        workdir: str | None = None,
                        anchor: bool = True,
                        qc: bool = True,
                        inspector=None,
                        max_iter: int = 5) -> "RegionDecomposition":
    """Fit each region independently and merge into one result.

    Pipeline per region:
      * restrict to the region (``wave_range``) with only its line complexes;
      * if ``anchor``: pin the local power-law continuum from line-free windows
        (the visual-first prior, applied locally);
      * if ``qc``: run the verify -> remedy loop *within the region* so each line
        complex self-corrects (e.g. an over-flexible 2-Gaussian Halpha that goes
        degenerate is pulled back to a single Gaussian) -- independently of the
        other region.
    """
    config = config or QsoparConfig.sdss_optical_default()
    inspector = inspector or RuleInspector()
    workdir = workdir or os.path.join(RUNS_DIR, f"region_{spectrum.name or 'obj'}")
    os.makedirs(workdir, exist_ok=True)

    per_region: dict[str, DecompositionResult] = {}
    used_regions: list[Region] = []
    for region in regions:
        sub = region_config(config, region)
        if not sub.lines:
            continue
        sub_spec = trim_spectrum(spectrum, region.lo, region.hi)
        if len(sub_spec.wave) < 120:        # PyQSOFit needs >=100 px
            continue
        rdir = os.path.join(workdir, region.name)

        if anchor:
            sub = _region_anchor(spectrum, sub, region, backend, rdir)

        # per-region QC loop
        history: list = []
        res = None
        for it in range(max_iter if qc else 1):
            res = decompose(spectrum, config=sub, backend=backend,
                            make_figure=False, workdir=os.path.join(rdir, f"it{it}"))
            if not qc:
                break
            rep = verify(res)
            decision = inspector.inspect(res, rep, history)
            history.append(_Step(decision))
            if decision.action == "apply" and decision.remedy:
                sub = apply_remedy(sub, decision.remedy)
                continue
            break
        else:
            # loop hit max_iter while still applying a remedy: the config advanced
            # but was not re-fit. Do one final fit so the result reflects it.
            if qc and history and history[-1].decision.action == "apply":
                res = decompose(spectrum, config=sub, backend=backend,
                                make_figure=False, workdir=os.path.join(rdir, "final"))

        per_region[region.name] = res
        used_regions.append(region)

    merged = merge_region_results(spectrum, per_region, used_regions, backend)
    return RegionDecomposition(merged=merged, regions=per_region,
                               order=[r.name for r in used_regions])


@dataclass
class RegionDecomposition:
    """Per-region fits plus their merged :class:`DecompositionResult`."""
    merged: DecompositionResult
    regions: dict                       # name -> DecompositionResult
    order: list                         # region names, blue -> red

    def __getattr__(self, item):        # delegate convenience to the merged result
        return getattr(self.merged, item)


def merge_region_results(spectrum, per_region: dict, regions: list[Region],
                         backend: str) -> DecompositionResult:
    """Stitch independent region fits into one combined result.

    Component arrays are concatenated in wavelength order (with a NaN gap between
    regions so plots don't bridge the void); line/continuum/quality dicts are
    merged with the bluer region taking precedence -- so L5100 comes from the
    Hbeta region that actually contains 5100 A.
    """
    parts = [per_region[r.name] for r in regions if r.name in per_region]
    if not parts:
        raise RuntimeError("no region produced a fit")

    keys: set[str] = set()
    for p in parts:
        keys |= set(p.components)

    wave_segs, idx_segs = [], []
    for p in parts:
        wave_segs.append(np.asarray(p.rest_wave, dtype=float))
        wave_segs.append(np.array([np.nan]))     # gap marker between regions
    waves = np.concatenate(wave_segs)

    comps: dict[str, np.ndarray] = {}
    for k in keys:
        segs = []
        for p in parts:
            n = len(p.rest_wave)
            v = p.components.get(k)
            segs.append(np.asarray(v, dtype=float) if v is not None and len(v) == n
                        else np.full(n, np.nan))
            segs.append(np.array([np.nan]))
        comps[k] = np.concatenate(segs)

    lines, continuum, quality, params = {}, {}, {}, {}
    for p in parts:                                # blue -> red; first wins
        lines.update(p.lines)
        quality.update(p.quality)
        for k, v in p.params.items():
            params.setdefault(k, v)
        for k, v in p.continuum.items():
            continuum.setdefault(k, v)

    cfg = {"mode": "region", "regions": [
        {"name": r.name, "lo": r.lo, "hi": r.hi,
         "complexes": list(r.complexes), "fe": r.fe} for r in regions]}

    return DecompositionResult(
        name=spectrum.name, z=spectrum.z, backend=backend,
        rest_wave=waves, components=comps, lines=lines, continuum=continuum,
        quality=quality, params=params, config=cfg, figure_path=None)


# -- visualization -------------------------------------------------------------
def render_region_diagnostic(rd: "RegionDecomposition", path: str) -> str:
    """One zoomed panel per region (data + components + pull), side by side."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    names = rd.order
    n = len(names)
    fig, axes = plt.subplots(2, n, figsize=(6.2 * n, 6.0), squeeze=False,
                             gridspec_kw={"height_ratios": [3, 1], "hspace": 0.06,
                                          "wspace": 0.18})
    # continuum = pure power law; Fe II drawn separately as its own component.
    from .plotting import COLORS
    comp_style = [("pl", COLORS["pl"], "continuum (power law)", 1.6, "-"),
                  ("feii_op", COLORS["feii_op"], "Fe II", 1.1, "-"),
                  ("feii_uv", COLORS["feii_uv"], "Fe II (UV)", 1.0, "-"),
                  ("broad", COLORS["broad"], "broad", 1.3, "-"),
                  ("narrow", COLORS["narrow"], "narrow", 1.0, "-")]
    for j, name in enumerate(names):
        r = rd.regions[name]
        w = r.rest_wave
        c = r.components
        ax, axr = axes[0][j], axes[1][j]
        ax.plot(w, c.get("data"), color=COLORS["data"], lw=0.7,
                drawstyle="steps-mid", label="data")
        if "model" in c:
            ax.plot(w, c["model"], color=COLORS["model"], lw=1.5, label="model")
        for key, color, lbl, lw, ls in comp_style:
            if key in c:
                ax.plot(w, c[key], color=color, lw=lw, ls=ls, alpha=0.9, label=lbl)
        m = r.lines.get(name)
        ttl = f"{name} region"
        if m is not None and np.isfinite(m.fwhm_kms):
            ttl += f"   broad FWHM={m.fwhm_kms:.0f} km/s"
        ax.set_title(ttl, fontsize=10)
        ax.legend(fontsize=7, ncol=2, loc="upper right")
        d = c.get("data")
        fin = np.isfinite(d)
        if fin.any():
            lo, hi = np.percentile(d[fin], [1, 99])
            pad = 0.15 * (hi - lo)
            ax.set_ylim(lo - pad, hi + pad)
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
        axr.set_xlabel(r"Rest wavelength [\AA]", fontsize=9)
        if j == 0:
            ax.set_ylabel(r"$f_\lambda$ [$10^{-17}$ cgs]", fontsize=9)
            axr.set_ylabel("pull", fontsize=9)
    fig.suptitle(f"{rd.merged.name}   z={rd.merged.z:.4f}   region-by-region "
                 f"decomposition", fontsize=11, y=0.98)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path
