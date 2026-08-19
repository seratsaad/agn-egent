"""Multi-epoch comparison: broad-line variability and changing-look AGN.

Everything else in the package measures one spectrum. This module compares the
*same object* at two or more epochs, which is where a different class of
discovery lives: broad lines that appear or vanish between visits (changing-look
AGN), and extreme continuum or line variability.

SDSS has hundreds of thousands of repeat spectra, so this is a search that can
be run over the archive at no cost beyond CPU. :func:`find_repeat_spectra`
locates the repeats for a position; :func:`compare_epochs` runs the standard
pipeline on each and diffs the results.

The comparison is deliberately conservative. A broad line "disappearing" is far
more often a low-S/N epoch, a bad fit, or a mis-placed continuum than a real
transition, so a changing-look candidate must show a *detection* in one epoch,
a clear *non-detection* in the other, and both fits must be ones the engine
trusts. Everything short of that is reported as variability, not a transition.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict

from .science import _nan_to_none, _finite, BROAD_SNR_DETECT

# A broad line must be this well detected in the "on" epoch before its absence
# in another epoch is called a changing-look event.
CL_ON_SNR = 5.0
CL_OFF_SNR = BROAD_SNR_DETECT        # 3.0
# Line-flux change, in dex, beyond which an object counts as strongly variable.
STRONG_VARIABILITY_DEX = 0.3
# Required significance of a flux change before it is believed at all.
MIN_SIGNIFICANCE = 3.0


@dataclass
class LineChange:
    """How one broad line changed between two epochs."""
    line: str = ""
    flux_a: float = float("nan")
    flux_b: float = float("nan")
    snr_a: float = float("nan")
    snr_b: float = float("nan")
    fwhm_a: float = float("nan")
    fwhm_b: float = float("nan")
    dlog_flux: float = float("nan")       # log10(flux_b / flux_a)
    fwhm_ratio: float = float("nan")
    significance: float = float("nan")    # |flux_b - flux_a| / combined error
    disappeared: bool = False             # detected in A, not in B
    appeared: bool = False                # detected in B, not in A
    strongly_variable: bool = False

    def to_dict(self) -> dict:
        return {k: _nan_to_none(v) for k, v in asdict(self).items()}


@dataclass
class VariabilityReport:
    name: str = ""
    epoch_a: str = ""
    epoch_b: str = ""
    quality_a: str = ""
    quality_b: str = ""
    lines: dict = field(default_factory=dict)      # line -> LineChange
    dlog_L5100: float = float("nan")
    changing_look: bool = False
    variable: bool = False
    trustworthy: bool = False        # both epochs are fits the engine trusts
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name, "epoch_a": self.epoch_a, "epoch_b": self.epoch_b,
            "quality_a": self.quality_a, "quality_b": self.quality_b,
            "dlog_L5100": _nan_to_none(self.dlog_L5100),
            "changing_look": self.changing_look, "variable": self.variable,
            "trustworthy": self.trustworthy, "note": self.note,
            "lines": {k: v.to_dict() for k, v in self.lines.items()},
        }

    def summary(self) -> str:
        bits = [f"{self.name}: {self.epoch_a} vs {self.epoch_b}"]
        for nm, c in self.lines.items():
            if _finite(c.dlog_flux):
                bits.append(f"{nm} dlogF={c.dlog_flux:+.2f} "
                            f"({c.significance:.1f} sigma)")
            if c.disappeared:
                bits.append(f"{nm} DISAPPEARED")
            if c.appeared:
                bits.append(f"{nm} APPEARED")
        if _finite(self.dlog_L5100):
            bits.append(f"dlogL5100={self.dlog_L5100:+.2f}")
        if self.changing_look:
            bits.append("CHANGING-LOOK CANDIDATE"
                        + ("" if self.trustworthy else " (untrusted fits)"))
        elif self.variable:
            bits.append("variable")
        return " | ".join(bits)


def _flux_error(m) -> float:
    """1-sigma flux error: the Monte-Carlo one if present, else flux / S/N."""
    if m is None:
        return float("nan")
    if _finite(getattr(m, "flux_err", None)):
        return float(m.flux_err)
    if _finite(m.flux) and _finite(m.snr) and m.snr > 0:
        return abs(float(m.flux)) / float(m.snr)
    return float("nan")


def compare_line(name: str, ma, mb) -> LineChange:
    """Diff one broad-line measurement between two epochs."""
    c = LineChange(line=name)
    if ma is None or mb is None:
        return c
    c.flux_a, c.flux_b = float(ma.flux), float(mb.flux)
    c.snr_a, c.snr_b = float(ma.snr), float(mb.snr)
    c.fwhm_a, c.fwhm_b = float(ma.fwhm_kms), float(mb.fwhm_kms)

    if _finite(c.flux_a) and _finite(c.flux_b) and c.flux_a > 0 and c.flux_b > 0:
        c.dlog_flux = math.log10(c.flux_b / c.flux_a)
    if _finite(c.fwhm_a) and _finite(c.fwhm_b) and c.fwhm_a > 0:
        c.fwhm_ratio = c.fwhm_b / c.fwhm_a

    ea, eb = _flux_error(ma), _flux_error(mb)
    if _finite(ea) and _finite(eb):
        denom = math.hypot(ea, eb)
        if denom > 0 and _finite(c.flux_a) and _finite(c.flux_b):
            c.significance = abs(c.flux_b - c.flux_a) / denom

    det_a = _finite(c.snr_a) and c.snr_a >= CL_ON_SNR
    det_b = _finite(c.snr_b) and c.snr_b >= CL_ON_SNR
    gone_a = _finite(c.snr_a) and c.snr_a < CL_OFF_SNR
    gone_b = _finite(c.snr_b) and c.snr_b < CL_OFF_SNR
    c.disappeared = bool(det_a and gone_b)
    c.appeared = bool(det_b and gone_a)
    c.strongly_variable = bool(
        _finite(c.dlog_flux) and abs(c.dlog_flux) >= STRONG_VARIABILITY_DEX
        and _finite(c.significance) and c.significance >= MIN_SIGNIFICANCE)
    return c


def compare_outcomes(outcome_a, outcome_b, name: str = "",
                     epoch_a: str = "A", epoch_b: str = "B") -> VariabilityReport:
    """Compare two completed agent runs of the same object.

    Takes :class:`~agn_egent.agent.loop.AgentOutcome` objects so the quality
    flags travel with the measurements -- a transition claimed from two fits the
    engine does not trust is not a transition, and the report says so rather
    than hiding it.
    """
    rep = VariabilityReport(name=name or outcome_a.name,
                            epoch_a=epoch_a, epoch_b=epoch_b,
                            quality_a=outcome_a.quality_flag,
                            quality_b=outcome_b.quality_flag)
    ra_, rb_ = outcome_a.final_result, outcome_b.final_result
    if ra_ is None or rb_ is None:
        rep.note = "one epoch has no usable fit"
        return rep

    rep.trustworthy = (outcome_a.quality_flag in ("clean", "reviewed")
                       and outcome_b.quality_flag in ("clean", "reviewed"))

    for nm in sorted(set(ra_.lines) | set(rb_.lines)):
        rep.lines[nm] = compare_line(nm, ra_.lines.get(nm), rb_.lines.get(nm))

    la = ra_.continuum.get("LogL5100")
    lb = rb_.continuum.get("LogL5100")
    if _finite(la) and _finite(lb):
        rep.dlog_L5100 = float(lb) - float(la)

    rep.changing_look = any(c.disappeared or c.appeared for c in rep.lines.values())
    rep.variable = bool(rep.changing_look
                        or any(c.strongly_variable for c in rep.lines.values())
                        or (_finite(rep.dlog_L5100)
                            and abs(rep.dlog_L5100) >= STRONG_VARIABILITY_DEX))
    if rep.changing_look and not rep.trustworthy:
        rep.note = ("broad-line change seen, but at least one epoch is a fit the "
                    "engine does not trust -- verify before believing it")
    return rep


def find_repeat_spectra(ra: float, dec: float, radius_arcsec: float = 2.0) -> list:
    """Find every SDSS spectrum at a position: the multi-epoch search input.

    Returns a list of ``(plate, mjd, fiberID)``. More than one entry means the
    object was observed more than once and can be tested for variability.
    Requires network.
    """
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astroquery.sdss import SDSS

    pos = SkyCoord(ra, dec, unit="deg")
    tab = SDSS.query_region(pos, radius=radius_arcsec * u.arcsec, spectro=True)
    if tab is None or len(tab) == 0:
        return []
    out, seen = [], set()
    for row in tab:
        try:
            key = (int(row["plate"]), int(row["mjd"]), int(row["fiberID"]))
        except (KeyError, TypeError, ValueError):
            continue
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def compare_epochs(epochs, inspector=None, workdir: str | None = None,
                   name: str = "", max_iterations: int = 4) -> VariabilityReport:
    """Fetch and fit two SDSS epochs of one object, then diff them.

    `epochs` is a sequence of at least two ``(plate, mjd, fiber)`` tuples; the
    first and last are compared, so passing a full list of repeats compares the
    widest baseline available.
    """
    import os
    from .catalog import fetch_sdss_spectrum
    from .agent.loop import run_agent
    from .agent.inspector import RuleInspector

    epochs = list(epochs)
    if len(epochs) < 2:
        return VariabilityReport(name=name, note="fewer than two epochs available")
    inspector = inspector or RuleInspector()
    workdir = workdir or os.path.join("agn_egent_out", name or "variability")

    outs, labels = [], []
    for plate, mjd, fiber in (epochs[0], epochs[-1]):
        label = f"{plate}-{mjd}-{fiber}"
        spec = fetch_sdss_spectrum(plate, mjd, fiber, name=label)
        outs.append(run_agent(spec, inspector=inspector,
                              max_iterations=max_iterations,
                              workdir=os.path.join(workdir, label), verbose=False))
        labels.append(label)
    return compare_outcomes(outs[0], outs[1], name=name or labels[0],
                            epoch_a=labels[0], epoch_b=labels[1])
