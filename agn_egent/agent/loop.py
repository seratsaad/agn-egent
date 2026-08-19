"""The agentic QC loop: decompose -> verify -> render -> inspect -> remedy -> repeat.

Backend-agnostic and inspector-agnostic. Every iteration is recorded as an
``AgentStep`` so the whole trajectory (configs tried, verdicts, decisions,
rationales, figures) is reproducible and auditable.

Two entry points:

* :func:`run_agent` -- the primitive: one inspector, one pass.
* :func:`run_agent_escalate` -- the **recommended** path for an LLM inspector
  (rule-first gating): the deterministic rules run to completion and the LLM is
  consulted *only* on the objects they leave flagged. This keeps the LLM from
  perturbing already-trustworthy masses, halves its footprint, and is the default
  used throughout the paper pipeline. Use :func:`run_agent` directly only for the
  rule-only baseline or the legacy triage ablation.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from ..pipeline import decompose, RUNS_DIR
from ..verify import verify, VerdictReport, Status
from ..remedies import apply_remedy, describe_remedy
from ..plotting import render_diagnostic
from ..continuum import apply_plan_to_config, render_continuum_plan
from ..backends.qsopar_config import QsoparConfig
from .inspector import Inspector, RuleInspector, Decision


@dataclass
class AgentStep:
    iteration: int
    config: dict                 # the QsoparConfig used this iteration (serialized)
    summary: str                 # result.summary()
    verdict: VerdictReport
    decision: Decision
    figure_path: str | None = None

    def to_dict(self) -> dict:
        return {
            "iteration": self.iteration,
            "verdict": str(self.verdict.overall),
            "checks": [{"name": c.name, "status": str(c.status),
                        "value": c.value, "message": c.message}
                       for c in self.verdict.checks],
            "decision": {"action": self.decision.action,
                         "remedy": self.decision.remedy,
                         "rationale": self.decision.rationale,
                         "source": self.decision.source},
            "figure_path": self.figure_path,
        }


@dataclass
class AgentOutcome:
    name: str
    status: str                  # "accepted" | "rejected" | "max_iterations"
    steps: list[AgentStep] = field(default_factory=list)
    final_result: object = None  # DecompositionResult
    final_verdict: VerdictReport = None
    continuum_plan: object = None  # ContinuumPlan from the pre-fit stage (if any)

    @property
    def n_iterations(self) -> int:
        return len(self.steps)

    # WARN-level checks that, on their own, make the single-epoch mass
    # untrustworthy (as opposed to benign warnings like an elevated reduced
    # chi^2 or a narrow-line residual, which do not invalidate the broad-line
    # mass). The flag is defined on these physical grounds, then tested for
    # calibration against the literature -- it is never tuned to the residuals.
    _MASS_CRITICAL = ("broad_snr", "broad_fwhm", "pl_slope", "host_fraction")

    @property
    def quality_flag(self) -> str:
        """Confidence tier for the measurement: 'clean' | 'reviewed' | 'flagged'.

        flagged  = the single-epoch mass should not be trusted: the fit was
                   rejected, a check FAILED, or a mass-critical condition warns
                   (weak/undetected or degenerate broad line, a railed/mis-placed
                   power-law continuum, or a host-dominated continuum).
        clean    = no warnings of any kind.
        reviewed = accepted with only benign warnings (e.g. elevated chi^2 or a
                   narrow-line residual) that do not invalidate the broad-line mass.
        """
        if self.status == "rejected" or self.final_verdict is None:
            return "flagged"
        checks = self.final_verdict.checks
        if any(c.status >= Status.FAIL for c in checks):
            return "flagged"
        if any(c.status >= Status.WARN and c.name.endswith(self._MASS_CRITICAL)
               for c in checks):
            return "flagged"
        if self.final_verdict.overall <= Status.PASS:
            return "clean"
        return "reviewed"

    @property
    def reviewed(self) -> bool:
        """Whether the loop went past triage (applied a remedy or used an LLM)."""
        return any(s.decision.remedy is not None for s in self.steps) or \
            any(s.decision.source not in ("triage", "") for s in self.steps)

    # Science + anomaly are derived from the *accepted* fit, so they are computed
    # lazily on first access and cached: a batch run that only wants masses never
    # pays for them, while a discovery campaign gets them without a second pass.
    _science = None
    _anomaly = None

    @property
    def science(self):
        """Line-shape science for the accepted fit (outflows, R_FeII, profiles)."""
        if self._science is None and self.final_result is not None:
            from ..measure import derive
            from ..science import science_report
            self._science = science_report(self.final_result,
                                           derived=derive(self.final_result, "Hb"))
        return self._science

    @property
    def anomaly(self):
        """Coherent-residual anomaly score for the accepted fit."""
        if self._anomaly is None and self.final_result is not None:
            from ..anomaly import anomaly_score
            self._anomaly = anomaly_score(self.final_result)
        return self._anomaly

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status,
            "n_iterations": self.n_iterations,
            "final_verdict": str(self.final_verdict.overall) if self.final_verdict else None,
            "continuum_plan": (self.continuum_plan.to_dict()
                               if self.continuum_plan is not None else None),
            "quality_flag": self.quality_flag,
            "science": self.science.to_dict() if self.science is not None else None,
            "anomaly": self.anomaly.to_dict() if self.anomaly is not None else None,
            "steps": [s.to_dict() for s in self.steps],
        }

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    def summary(self) -> str:
        lines = [f"AGENT OUTCOME [{self.name}]: {self.status} "
                 f"after {self.n_iterations} iteration(s)"]
        for s in self.steps:
            act = s.decision.action
            rem = (" -> " + describe_remedy(s.decision.remedy)) if s.decision.remedy else ""
            lines.append(f"  iter {s.iteration}: verdict={s.verdict.overall!s}  "
                         f"[{s.decision.source}] {act}{rem}")
            if s.decision.rationale:
                lines.append(f"           {s.decision.rationale}")
        return "\n".join(lines)


def plan_continuum(spectrum, config, planner, backend="pyqsofit",
                   workdir=None, verbose=False):
    """Pre-fit continuum anchoring (two-stage).

    Run a quick baseline fit to get the host-subtracted QSO flux, let the planner
    select line-/Fe II-free continuum windows on it, and fold that into the config
    as soft constraints (PL slope init + narrowed bounds + restricted windows).
    Planning on the raw flux instead would conflate host+power-law, so the
    baseline fit is what makes the anchored slope trustworthy.

    Returns ``(new_config, plan)``.
    """
    base = decompose(spectrum, config=config, backend=backend, make_figure=False,
                     workdir=os.path.join(workdir or ".", "continuum_base"))
    plan = planner.plan(spectrum, result=base)
    if workdir:
        try:
            render_continuum_plan(spectrum, plan,
                                  os.path.join(workdir, "continuum_plan.png"),
                                  result=base)
        except Exception:
            pass
    if verbose:
        print(f"[continuum] class={plan.classification} slope={plan.slope:.2f} "
              f"({plan.source}); {plan.rationale}")
    return apply_plan_to_config(config, plan), plan


def run_agent(spectrum,
              inspector: Inspector | None = None,
              config: QsoparConfig | None = None,
              backend: str = "pyqsofit",
              max_iterations: int = 4,
              workdir: str | None = None,
              finalize_mc: bool = False,
              nsamp: int = 25,
              continuum_planner=None,
              render: str = "all",
              verbose: bool = False) -> AgentOutcome:
    """Run the QC loop until the inspector accepts/rejects or iterations run out.

    If ``finalize_mc`` is set, the accepted fit is re-run once with Monte-Carlo
    resampling so the reported measurements carry 1-sigma uncertainties. The fast
    MLE fits are used during iteration; MC is paid only on the final config.

    If ``continuum_planner`` is given, a visual-first continuum-anchoring stage
    runs *before* the QC loop: it pins the power-law continuum from line-/Fe II-
    free windows so the loop starts from a trustworthy continuum (see
    :func:`plan_continuum`).

    ``render`` controls diagnostic figures: ``"all"`` (default) renders one per
    iteration -- REQUIRED for any vision-LLM inspector, which reads them;
    ``"final"`` renders only the last (kept) fit's figure, which is all a
    rule-only survey campaign needs and saves ~1-2 s of matplotlib plus a PNG
    per iteration at scale; ``"none"`` renders nothing.
    """
    inspector = inspector or RuleInspector()
    config = config or QsoparConfig.sdss_optical_default()
    workdir = workdir or os.path.join(RUNS_DIR, f"agent_{spectrum.name or 'obj'}")
    os.makedirs(workdir, exist_ok=True)

    continuum_plan = None
    if continuum_planner is not None:
        config, continuum_plan = plan_continuum(
            spectrum, config, continuum_planner, backend=backend,
            workdir=workdir, verbose=verbose)

    steps: list[AgentStep] = []
    result = verdict = None
    status = "max_iterations"

    for it in range(max_iterations):
        it_dir = os.path.join(workdir, f"iter{it}")
        result = decompose(spectrum, config=config, backend=backend,
                           make_figure=False, workdir=it_dir)
        verdict = verify(result)
        fig = None
        if render == "all":
            fig = render_diagnostic(result, os.path.join(it_dir, "diagnostic.png"),
                                    report=verdict)
        result.figure_path = fig

        decision = inspector.inspect(result, verdict, steps)
        steps.append(AgentStep(it, result.config, result.summary(), verdict,
                               decision, fig))
        if verbose:
            print(f"[iter {it}] verdict={verdict.overall!s} -> "
                  f"{decision.action} ({decision.source}): {decision.rationale}")

        if decision.action == "accept":
            status = "accepted"
            break
        if decision.action == "reject":
            status = "rejected"
            break
        if decision.action == "apply" and decision.remedy:
            config = apply_remedy(config, decision.remedy)
        else:
            status = "accepted"  # no actionable move
            break

    # the kept fit still gets its diagnostic -- galleries and human vetting
    # need the picture even when the loop itself ran blind
    if render == "final" and result is not None and steps:
        it_dir = os.path.join(workdir, f"iter{steps[-1].iteration}")
        fig = render_diagnostic(result, os.path.join(it_dir, "diagnostic.png"),
                                report=verdict)
        result.figure_path = fig
        steps[-1].figure_path = fig

    # finalize accepted fits with Monte-Carlo uncertainties
    if finalize_mc and status == "accepted":
        mc_cfg = config.set_fit_option("MC", True).set_fit_option("nsamp", nsamp)
        result = decompose(spectrum, config=mc_cfg, backend=backend,
                           make_figure=False,
                           workdir=os.path.join(workdir, "final_mc"))
        verdict = verify(result)

    return AgentOutcome(spectrum.name or "obj", status, steps, result, verdict,
                        continuum_plan=continuum_plan)


def run_agent_escalate(spectrum,
                       llm_inspector: Inspector,
                       rule_inspector: Inspector | None = None,
                       escalate_on=("flagged",),
                       workdir: str | None = None,
                       **kwargs):
    """Two-pass 'rule-first' QC: deterministic rules run to completion, and the
    LLM is consulted *only* on the objects the rule pass leaves untrustworthy.

    Pass 1 fits the object with the deterministic ``RuleInspector``. If the rule
    outcome is acceptable (``quality_flag`` not in ``escalate_on`` -- i.e. clean
    or reviewed-with-benign-warnings) it is returned as-is: the LLM never sees it,
    so it cannot perturb an already-good mass. Only when the rule pass is
    ``flagged`` (rejected, a FAILed check, or a mass-critical warning) does Pass 2
    re-run the full loop with the LLM inspector, which gets a fresh crack with the
    complete remedy vocabulary.

    This concentrates the LLM exactly where the deterministic baseline fails --
    minimising both cost and the LLM's footprint on the science quantity.

    Returns ``(outcome, escalated, rule_outcome)``: ``outcome`` is the final result
    (the rule outcome if not escalated, else the LLM's), ``escalated`` is True iff
    the LLM ran, and ``rule_outcome`` is always the deterministic rule pass -- so a
    single call yields both the rule baseline and the agent result, perfectly paired.
    """
    rule_inspector = rule_inspector or RuleInspector()
    base = workdir or os.path.join(RUNS_DIR, f"agent_{spectrum.name or 'obj'}")

    out_rule = run_agent(spectrum, inspector=rule_inspector,
                         workdir=os.path.join(base, "rule"), **kwargs)
    if out_rule.quality_flag not in escalate_on:
        return out_rule, False, out_rule

    out_llm = run_agent(spectrum, inspector=llm_inspector,
                        workdir=os.path.join(base, "llm"), **kwargs)
    return out_llm, True, out_rule
