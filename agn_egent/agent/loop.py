"""The agentic QC loop: decompose -> verify -> render -> inspect -> remedy -> repeat.

Backend-agnostic and inspector-agnostic. Every iteration is recorded as an
``AgentStep`` so the whole trajectory (configs tried, verdicts, decisions,
rationales, figures) is reproducible and auditable.
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

    @property
    def quality_flag(self) -> str:
        """Egent-style confidence tier: 'clean' | 'reviewed' | 'flagged'.

        clean    = accepted with an all-PASS verdict (no human/LLM attention needed)
        reviewed = accepted but with residual warnings (best-effort, data-limited)
        flagged  = rejected, or any FAIL remains (needs attention)
        """
        if self.status == "rejected":
            return "flagged"
        if self.final_verdict is None:
            return "flagged"
        if any(c.status >= Status.FAIL for c in self.final_verdict.checks):
            return "flagged"
        if self.final_verdict.overall <= Status.PASS:
            return "clean"
        return "reviewed"

    @property
    def reviewed(self) -> bool:
        """Whether the loop went past triage (applied a remedy or used an LLM)."""
        return any(s.decision.remedy is not None for s in self.steps) or \
            any(s.decision.source not in ("triage", "") for s in self.steps)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status,
            "n_iterations": self.n_iterations,
            "final_verdict": str(self.final_verdict.overall) if self.final_verdict else None,
            "continuum_plan": (self.continuum_plan.to_dict()
                               if self.continuum_plan is not None else None),
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
              verbose: bool = False) -> AgentOutcome:
    """Run the QC loop until the inspector accepts/rejects or iterations run out.

    If ``finalize_mc`` is set, the accepted fit is re-run once with Monte-Carlo
    resampling so the reported measurements carry 1-sigma uncertainties. The fast
    MLE fits are used during iteration; MC is paid only on the final config.

    If ``continuum_planner`` is given, a visual-first continuum-anchoring stage
    runs *before* the QC loop: it pins the power-law continuum from line-/Fe II-
    free windows so the loop starts from a trustworthy continuum (see
    :func:`plan_continuum`).
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

    # finalize accepted fits with Monte-Carlo uncertainties
    if finalize_mc and status == "accepted":
        mc_cfg = config.set_fit_option("MC", True).set_fit_option("nsamp", nsamp)
        result = decompose(spectrum, config=mc_cfg, backend=backend,
                           make_figure=False,
                           workdir=os.path.join(workdir, "final_mc"))
        verdict = verify(result)

    return AgentOutcome(spectrum.name or "obj", status, steps, result, verdict,
                        continuum_plan=continuum_plan)
