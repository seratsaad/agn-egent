"""Vision-first continuum planner: the LLM *selects windows*, the math fits.

This is the pre-fit counterpart to the post-fit :mod:`~agn_egent.agent.inspector`.
Egent uses the LLM only as a post-hoc reviewer; for AGN the continuum is the
under-determined, prior-dependent part, so here the LLM acts up front -- it looks
at the raw spectrum with candidate continuum windows marked and decides which
windows are genuinely line-/Fe II-free, classifies the object, and says whether
host decomposition is needed.

Crucially the LLM does NOT output continuum numbers (vision models read flux
levels off a plot imprecisely). It only *selects and classifies*; the power-law
slope/norm are then fit deterministically through the windows it endorsed
(:func:`agn_egent.continuum.anchored_pl_fit`). Same closed, auditable surface as
the remedy vocabulary.
"""
from __future__ import annotations

import base64

from ..continuum import (CANDIDATE_WINDOWS, window_stats, anchored_pl_fit,
                         make_plan, render_continuum_plan, RuleContinuumPlanner)


def make_continuum_planner(name: str = "rule", api_key: str | None = None,
                           model: str | None = None):
    """Build a continuum planner by name: 'rule', 'claude'/'anthropic'."""
    name = (name or "rule").lower()
    if name in ("claude", "anthropic"):
        kw = {}
        if api_key:
            kw["api_key"] = api_key
        if model:
            kw["model"] = model
        return ClaudeContinuumPlanner(**kw)
    return RuleContinuumPlanner()


_SYSTEM_PROMPT = """\
You are an expert AGN/quasar spectroscopist setting up the *continuum* for a \
spectral decomposition. Unlike a stellar continuum (predictable from Teff/logg), \
an AGN continuum is a blend of a disk power law, the Fe II pseudo-continuum, the \
Balmer continuum, and host-galaxy starlight, with no single physical model -- so \
a human picks the continuum by eye first. That is your job here.

You are shown a figure of one rest-frame spectrum with numbered candidate \
continuum windows shaded, each with its median flux marked, plus a table of the \
windows. Decide which windows lie on the *true underlying continuum* (line-free, \
and NOT riding on top of the Fe II pseudo-continuum), and classify the object.

Principles:
  * The continuum is a LOWER envelope. Reject any window whose flux sits visibly \
    ABOVE its neighbours -- that excess is Fe II or an emission line, not continuum.
  * The Fe II forest (~4434-4684 and ~5100-5500 A) masquerades as continuum. \
    Windows flagged fe=true are suspect; keep them only if the object is clearly \
    Fe-weak and they sit on the envelope.
  * A flat or rising f_lambda toward the red means host starlight dominates \
    (classification host_dominated) -- keep host decomposition on.
  * Keep at least 3 windows spanning a wide wavelength baseline so the slope is \
    well determined.
  * Do not output slope or flux numbers; only choose windows and classify.

Always finish by calling submit_continuum_plan exactly once."""

_PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "use_windows": {
            "type": "array", "items": {"type": "integer"},
            "description": "0-based indices (from the table) of the windows that "
                           "lie on the true continuum and should anchor the fit.",
        },
        "classification": {
            "type": "string",
            "enum": ["blue_pl", "host_dominated", "fe_loud", "normal"],
            "description": "blue_pl=steep blue power law; host_dominated=flat/red, "
                           "starlight wins; fe_loud=strong Fe II forest; normal.",
        },
        "host": {"type": "boolean",
                 "description": "keep host-galaxy decomposition on?"},
        "rationale": {"type": "string",
                      "description": "One or two sentences citing the evidence "
                                     "(which windows ride on Fe II / lines)."},
    },
    "required": ["use_windows", "classification", "rationale"],
}
_ANTHROPIC_TOOL = {"name": "submit_continuum_plan",
                   "description": "Submit the continuum-window selection and "
                                  "object classification.",
                   "input_schema": _PLAN_SCHEMA}


def _window_table(stats) -> str:
    rows = []
    for i, s in enumerate(stats):
        if not s.covered:
            rows.append(f"  [{i}] {s.window.label:14s} {s.window.lo:.0f}-{s.window.hi:.0f}A  "
                        "(not covered)")
            continue
        rows.append(f"  [{i}] {s.window.label:14s} {s.window.lo:.0f}-{s.window.hi:.0f}A  "
                    f"median={s.median:8.2f}  scatter={s.scatter:.2f}  "
                    f"fe={str(s.window.fe).lower()}")
    return "\n".join(rows)


class ClaudeContinuumPlanner:
    """Claude selects the trustworthy continuum windows; we fit the PL through them."""

    name = "claude"

    def __init__(self, model: str = "claude-opus-4-8", client=None,
                 api_key: str | None = None, max_tokens: int = 1536,
                 fig_dir: str | None = None):
        self.model = model
        self.max_tokens = max_tokens
        self.api_key = api_key
        self._client = client
        self.fig_dir = fig_dir

    def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = (anthropic.Anthropic(api_key=self.api_key)
                            if self.api_key else anthropic.Anthropic())
        return self._client

    def plan(self, spectrum, result=None):
        import os
        import tempfile

        rw = spectrum.rest_wave
        flux = spectrum.flux
        if result is not None and "qso" in result.components:
            rw = result.rest_wave
            flux = result.components["qso"]
        stats = window_stats(rw, flux)

        # render a provisional planning figure (rule selection) for the model
        provisional = RuleContinuumPlanner().plan(spectrum, result=result)
        fig_dir = self.fig_dir or tempfile.mkdtemp(prefix="contplan_")
        fig_path = os.path.join(fig_dir, f"{spectrum.name or 'obj'}_plan.png")
        render_continuum_plan(spectrum, provisional, fig_path, result=result)

        user_text = (f"OBJECT: {spectrum.name}  z={spectrum.z:.4f}\n\n"
                     f"CANDIDATE WINDOWS:\n{_window_table(stats)}\n\n"
                     "The figure shows these windows shaded (green/orange=tentatively "
                     "kept, red=rejected by a first pass) with the first-pass power "
                     "law in red. Decide the final window selection and classify the "
                     "object, then submit your plan.")
        with open(fig_path, "rb") as f:
            b64 = base64.standard_b64encode(f.read()).decode("utf-8")
        content = [{"type": "text", "text": user_text},
                   {"type": "image", "source": {"type": "base64",
                    "media_type": "image/png", "data": b64}}]

        resp = self._get_client().messages.create(
            model=self.model, max_tokens=self.max_tokens,
            thinking={"type": "adaptive"},
            system=[{"type": "text", "text": _SYSTEM_PROMPT,
                     "cache_control": {"type": "ephemeral"}}],
            tools=[_ANTHROPIC_TOOL],
            messages=[{"role": "user", "content": content}])

        tool_use = next((b for b in resp.content
                         if getattr(b, "type", None) == "tool_use"
                         and b.name == "submit_continuum_plan"), None)
        if tool_use is None:
            return provisional  # fall back to the deterministic plan
        inp = tool_use.input
        chosen = set(int(i) for i in inp.get("use_windows", []))
        for i, s in enumerate(stats):
            s.used = (i in chosen) and s.covered
        slope, norm, _ = anchored_pl_fit(stats)
        host = inp.get("host")
        return make_plan(slope, norm, stats, source=self.name,
                         rationale=inp.get("rationale", ""),
                         host=host if host is not None else None)
