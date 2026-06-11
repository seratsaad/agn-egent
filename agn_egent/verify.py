"""Phase 2: deterministic per-stage verification of a decomposition.

Encodes "is this fit trustworthy?" as a set of objective checks over a
:class:`DecompositionResult`. Each :class:`Check` carries a status and, when it
fails, a *structured remedy* — a machine-applicable config edit. In Phase 3 the
LLM reads the rendered residuals plus this report and chooses among the
remedies; here the checks and remedies are fully deterministic so the loop can
already run without any model in the seat.

Thresholds live as module constants so they are easy to tune and so the agent
layer can reference the same numbers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

from .backends.base import DecompositionResult


class Status(IntEnum):
    PASS = 0
    WARN = 1
    FAIL = 2

    def __str__(self):
        return self.name


# -- thresholds ----------------------------------------------------------------
BROAD_SNR_FAIL = 3.0          # below this, broad line is not detected
BROAD_SNR_WARN = 6.0          # below this, broad params are poorly constrained
FWHM_MIN = 1000.0             # km/s; narrower => likely a narrow line mislabeled
FWHM_MAX_FAIL = 20000.0       # km/s; unphysically broad
FWHM_MAX_WARN = 10000.0       # km/s; suspiciously broad (often a degenerate fit)
NARROW_FWHM_MAX = 1500.0      # km/s; narrow components should be below this
REDCHI_WARN = 1.5            # per-complex reduced chi^2 upper warn
# A high per-complex chi^2 is a *warning*, not a rejection: with a pure power-law
# continuum the formal chi^2 runs higher (real continua have mild curvature), and
# continuum/narrow-line residuals inflate it without invalidating the broad-line
# mass. Only a catastrophic chi^2 fails.
REDCHI_FAIL = 5.0
REDCHI_OVERFIT = 0.25        # very low => model too flexible for the data
HOST_FRAC_WARN = 0.8         # host-dominated => AGN lines unreliable
# Systematic residual: fraction of pixels in a line window beyond 3 sigma.
# A good fit leaves ~0.3%; >10% indicates correlated/missed structure. Using a
# fraction (not max |pull|) avoids firing on one or two sharp narrow-line peaks.
CORE_RESID_FRAC_WARN = 0.10
PL_SLOPE_WARN = 0.0          # positive f_lambda slope is unusual (host-dominated)
# Narrow-line peak SNR (peak amplitude / local noise). Below NOISE the component
# is consistent with a noise spike rather than a real line.
NARROW_SNR_NOISE = 3.0
NARROW_OVERFLEX_SNR = 8.0    # >1 Gaussian on a narrow line weaker than this is over-flexible


@dataclass
class Check:
    name: str                       # short check id, e.g. "Hb.broad_snr"
    stage: str                      # ingest | continuum | lines | measure
    status: Status
    message: str
    value: float | None = None
    target: str | None = None       # complex/line the check is about
    remedy: dict | None = None      # structured, agent-applicable fix

    def __str__(self):
        v = "" if self.value is None else f" ({self.value:.4g})"
        return f"[{self.status!s:4}] {self.name}{v}: {self.message}"


@dataclass
class VerdictReport:
    checks: list[Check] = field(default_factory=list)

    @property
    def overall(self) -> Status:
        return max((c.status for c in self.checks), default=Status.PASS)

    def by_status(self, status: Status) -> list[Check]:
        return [c for c in self.checks if c.status == status]

    @property
    def failures(self) -> list[Check]:
        return self.by_status(Status.FAIL)

    @property
    def warnings(self) -> list[Check]:
        return self.by_status(Status.WARN)

    def remedies(self) -> list[dict]:
        """Distinct suggested fixes, worst-status first, de-duplicated."""
        seen, out = set(), []
        for c in sorted(self.checks, key=lambda c: -c.status):
            if c.remedy:
                key = tuple(sorted(c.remedy.items()))
                if key not in seen:
                    seen.add(key)
                    out.append(c.remedy)
        return out

    def summary(self) -> str:
        lines = [f"Verdict: {self.overall!s}  "
                 f"({len(self.failures)} fail, {len(self.warnings)} warn, "
                 f"{len(self.by_status(Status.PASS))} pass)"]
        for c in sorted(self.checks, key=lambda c: -c.status):
            lines.append("  " + str(c))
        rem = self.remedies()
        if rem:
            lines.append("  suggested remedies:")
            for r in rem:
                lines.append(f"    - {r}")
        return "\n".join(lines)


# -- the verifier --------------------------------------------------------------
def verify(result: DecompositionResult) -> VerdictReport:
    rep = VerdictReport()
    _check_continuum(result, rep)
    for comp in result.lines:
        _check_complex(result, comp, rep)
    _check_narrow_lines(result, rep)
    _check_measure(result, rep)
    return rep


def _check_narrow_lines(result: DecompositionResult, rep: VerdictReport):
    """Per narrow-line-component checks: noise rejection + Gaussian-count control.

    Mirrors the broad-line knobs for narrow lines. Two actionable cases:
      * a narrow component sitting at the noise level (peak ~ local 1-sigma) is
        likely fitting a noise spike, not a real line -> offer to remove it;
      * more than one Gaussian on a weak narrow line is over-flexible -> reduce.
    Anchors of a tie group and the Balmer narrow cores are never offered for
    removal (removing an anchor breaks the tie; the Balmer narrow core blends
    with the broad line and is owned by the broad checks).
    """
    for linename, m in (result.narrow_lines or {}).items():
        snr = m.get("snr", np.nan)
        if not np.isfinite(snr):
            continue
        ng = int(m.get("ngauss", 1))
        comp = m.get("compname", "")
        # never offer to remove: a tie anchor (breaks the tie), the Balmer narrow
        # core (blends with the broad line), or a flux-tied multiplet member
        # ([N II]/[S II]/tied [O III]) -- those are real lines, not noise.
        protected = (m.get("anchor", False) or linename.endswith("_na")
                     or m.get("tied", False))

        if snr < NARROW_SNR_NOISE and not protected:
            rep.checks.append(Check(
                f"{linename}.narrow_noise", "lines", Status.WARN,
                "narrow component is consistent with noise (peak ~ local 1-sigma); "
                "likely a noise spike, not a real line", value=snr, target=comp,
                remedy={"action": "remove_line", "linename": linename}))
        elif ng > 1 and snr < NARROW_OVERFLEX_SNR:
            rep.checks.append(Check(
                f"{linename}.narrow_flexibility", "lines", Status.WARN,
                f"{ng} Gaussians on a weak narrow line (SNR<{NARROW_OVERFLEX_SNR:.0f}); "
                "narrow profile likely over-flexible", value=float(ng), target=comp,
                remedy={"action": "set_ngauss", "linename": linename, "value": 1}))


def _broad_ngauss(result: DecompositionResult, comp: str) -> int | None:
    """How many Gaussians the broad component of `comp` was given."""
    for l in result.config.get("lines", []):
        if l.get("compname") == comp and "_br" in str(l.get("linename", "")):
            return int(l.get("ngauss", 1))
    return None


def _broad_linename(result: DecompositionResult, comp: str) -> str | None:
    for l in result.config.get("lines", []):
        if l.get("compname") == comp and "_br" in str(l.get("linename", "")):
            return str(l.get("linename"))
    return None


def _check_continuum(result: DecompositionResult, rep: VerdictReport):
    cont = result.continuum
    slope = cont.get("PL_slope")
    if slope is not None:
        # railed against the physical bounds (-3.0 .. 0.0) => the power-law
        # continuum could not fit the data and is likely mis-placed.
        railed = slope <= -2.95 or slope >= -0.05
        st = Status.WARN if railed else Status.PASS
        rep.checks.append(Check(
            "continuum.pl_slope", "continuum", st,
            "power-law continuum slope is at its physical bound; the continuum "
            "may be mis-placed (host-dominated, or the fitting windows need "
            "adjustment)" if railed else "power-law continuum slope physical",
            value=slope))

    host = cont.get("frac_host_5100")
    if host is not None:
        st = Status.WARN if host > HOST_FRAC_WARN else Status.PASS
        rep.checks.append(Check(
            "continuum.host_fraction", "continuum", st,
            f"host dominates the 5100A flux (frac={host:.2f}); broad lines may be "
            "unreliable" if st else "host fraction acceptable",
            value=host, target="host"))


def _check_complex(result: DecompositionResult, comp: str, rep: VerdictReport):
    m = result.lines[comp]
    ng = _broad_ngauss(result, comp)
    brname = _broad_linename(result, comp)
    has_broad = brname is not None
    reduce_remedy = ({"action": "set_ngauss", "linename": brname, "value": 1}
                     if brname and ng and ng > 1 else None)

    # If the model has no broad component for this complex (e.g. it was dropped,
    # or this is a narrow-line object), the backend's whole_br_* numbers are
    # meaningless -- skip the broad checks and only judge overall fit quality.
    if not has_broad:
        rep.checks.append(Check(
            f"{comp}.no_broad", "lines", Status.PASS,
            "no broad component fit for this complex (narrow-line only)",
            target=comp))
        _check_complex_quality(result, comp, rep)
        _check_core_residual(result, comp, rep)
        return

    # broad-line SNR
    snr = m.snr
    if np.isfinite(snr):
        if snr < BROAD_SNR_FAIL:
            rep.checks.append(Check(
                f"{comp}.broad_snr", "lines", Status.FAIL,
                "broad component not detected", value=snr, target=comp,
                remedy={"action": "drop_broad", "compname": comp}))
        elif snr < BROAD_SNR_WARN:
            rep.checks.append(Check(
                f"{comp}.broad_snr", "lines", Status.WARN,
                "broad component weak / poorly constrained", value=snr,
                target=comp, remedy=reduce_remedy))
        else:
            rep.checks.append(Check(f"{comp}.broad_snr", "lines", Status.PASS,
                                    "broad component well detected", value=snr,
                                    target=comp))

    # broad FWHM physicality
    fwhm = m.fwhm_kms
    if np.isfinite(fwhm):
        if fwhm < FWHM_MIN or fwhm > FWHM_MAX_FAIL:
            rep.checks.append(Check(
                f"{comp}.broad_fwhm", "lines", Status.FAIL,
                "broad FWHM outside physical range", value=fwhm, target=comp,
                remedy=reduce_remedy))
        elif fwhm > FWHM_MAX_WARN:
            rep.checks.append(Check(
                f"{comp}.broad_fwhm", "lines", Status.WARN,
                "broad FWHM suspiciously large (often a degenerate multi-Gaussian fit)",
                value=fwhm, target=comp, remedy=reduce_remedy))
        else:
            rep.checks.append(Check(f"{comp}.broad_fwhm", "lines", Status.PASS,
                                    "broad FWHM physical", value=fwhm, target=comp))

    # over-flexible model: multiple broad Gaussians on a weak line
    if ng and ng > 1 and np.isfinite(snr) and snr < BROAD_SNR_WARN:
        rep.checks.append(Check(
            f"{comp}.model_flexibility", "lines", Status.WARN,
            f"{ng} broad Gaussians on a weak line (SNR<{BROAD_SNR_WARN:.0f}); "
            "model likely over-flexible", value=float(ng), target=comp,
            remedy=reduce_remedy))

    # under-fit broad line: a single Gaussian leaving elevated chi^2 at high S/N
    # -> the broad profile likely needs an extra component (Egent: "add a blend")
    rchi = result.quality.get(comp)
    if (has_broad and ng == 1 and np.isfinite(snr) and snr > 2 * BROAD_SNR_WARN
            and rchi is not None and np.isfinite(rchi) and rchi > REDCHI_WARN):
        rep.checks.append(Check(
            f"{comp}.underfit", "lines", Status.WARN,
            "single broad Gaussian leaves an elevated reduced chi^2 at high S/N; "
            "the broad profile may need another component", value=rchi, target=comp,
            remedy={"action": "set_ngauss", "linename": brname, "value": 2}))

    # per-complex reduced chi^2 + residual structure
    _check_complex_quality(result, comp, rep, reduce_remedy)
    _check_core_residual(result, comp, rep)


def _check_complex_quality(result, comp, rep, reduce_remedy=None):
    """Per-complex reduced chi^2 check (applies with or without a broad line)."""
    rchi = result.quality.get(comp)
    if rchi is None or not np.isfinite(rchi):
        return
    if rchi > REDCHI_FAIL:
        rep.checks.append(Check(f"{comp}.redchi2", "lines", Status.FAIL,
                                "poor fit (high reduced chi^2)", value=rchi,
                                target=comp))
    elif rchi > REDCHI_WARN:
        rep.checks.append(Check(f"{comp}.redchi2", "lines", Status.WARN,
                                "elevated reduced chi^2", value=rchi, target=comp))
    elif rchi < REDCHI_OVERFIT:
        rep.checks.append(Check(f"{comp}.redchi2", "lines", Status.WARN,
                                "very low reduced chi^2 (model may be overfitting)",
                                value=rchi, target=comp, remedy=reduce_remedy))
    else:
        rep.checks.append(Check(f"{comp}.redchi2", "lines", Status.PASS,
                                "reduced chi^2 acceptable", value=rchi, target=comp))


def _check_core_residual(result: DecompositionResult, comp: str, rep: VerdictReport):
    c = result.components
    if "residual" not in c or "err" not in c:
        return
    win = None
    for l in result.config.get("lines", []):
        if l.get("compname") == comp:
            win = (float(l["minwav"]), float(l["maxwav"]))
            break
    if win is None:
        return
    w = result.rest_wave
    mask = (w >= win[0]) & (w <= win[1])
    err = c["err"][mask]
    resid = c["residual"][mask]
    good = np.isfinite(err) & (err > 0) & np.isfinite(resid)
    if good.sum() < 5:
        return
    pull = np.abs(resid[good] / err[good])
    frac = float(np.mean(pull > 3.0))      # fraction of pixels beyond 3 sigma
    st = Status.WARN if frac > CORE_RESID_FRAC_WARN else Status.PASS
    rep.checks.append(Check(
        f"{comp}.core_residual", "lines", st,
        f"correlated residual in line window ({100*frac:.0f}% of pixels >3sigma); "
        "possible missed component or continuum error" if st
        else "residuals consistent with noise",
        value=frac, target=comp))


def _check_measure(result: DecompositionResult, rep: VerdictReport):
    for comp, m in result.lines.items():
        if _broad_linename(result, comp) is None:
            continue  # no broad component -> nothing to measure
        if not np.isfinite(m.fwhm_kms) or not np.isfinite(m.flux):
            rep.checks.append(Check(
                f"{comp}.measure_finite", "measure", Status.FAIL,
                "broad-line measurement is not finite", target=comp))
