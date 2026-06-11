"""Inspectors: decide what to do with a fit given its verdict + diagnostic plot.

Two implementations, same interface:

* ``RuleInspector`` — deterministic, no LLM. Applies the worst un-tried remedy
  until none remain, then accepts (or rejects on an unfixable failure). This is
  the offline default and the baseline the LLM must beat.
* ``ClaudeInspector`` — a vision LLM (Claude) reads the rendered residual plot
  plus the verdict report and *chooses among the offered remedies* (or accepts
  / rejects). It picks from the same closed vocabulary the RuleInspector uses,
  so the loop, provenance, and safety properties are identical.

The LLM does structural judgment (which edit, or stop); the math stays
deterministic in the pipeline.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from ..verify import VerdictReport, Status
from ..remedies import describe_remedy


@dataclass
class Decision:
    action: str                       # "accept" | "apply" | "reject"
    remedy: dict | None = None        # set when action == "apply"
    rationale: str = ""
    source: str = ""                  # which inspector produced it
    extra: dict = field(default_factory=dict)


@runtime_checkable
class Inspector(Protocol):
    name: str

    def inspect(self, result, report: VerdictReport, history: list) -> Decision:
        ...


def make_inspector(name: str = "rule", api_key: str | None = None,
                   model: str | None = None):
    """Build an inspector by name: 'rule', 'claude'/'anthropic', or 'openai'/'gpt'."""
    name = (name or "rule").lower()
    kw = {}
    if api_key:
        kw["api_key"] = api_key
    if model:
        kw["model"] = model
    if name in ("claude", "anthropic"):
        return ClaudeInspector(**kw)
    if name in ("openai", "gpt"):
        return OpenAIInspector(**kw)
    return RuleInspector()


def _tried_remedies(history: list) -> list[dict]:
    out = []
    for step in history:
        d = getattr(step, "decision", None)
        if d is not None and d.remedy:
            out.append(d.remedy)
    return out


# -- deterministic baseline ----------------------------------------------------
class RuleInspector:
    """Pick the worst un-tried remedy; accept when clean / out of moves."""

    name = "rule"

    def inspect(self, result, report: VerdictReport, history: list) -> Decision:
        tried = _tried_remedies(history)
        pending = [r for r in report.remedies() if r not in tried]

        if report.overall <= Status.PASS:
            return Decision("accept", None, "all checks pass", self.name)
        if pending:
            r = pending[0]
            return Decision("apply", r, f"apply remedy: {describe_remedy(r)}", self.name)
        if report.failures:
            return Decision("reject", None,
                            "unresolved failures and no remaining remedy", self.name)
        return Decision("accept", None,
                        "remaining warnings reflect data limits, not fixable by "
                        "a model edit", self.name)


# -- triage wrapper ------------------------------------------------------------
class TriageInspector:
    """Auto-accept clean fits without consulting the delegate.

    This is the throughput lever for batch mode: a passing verdict is accepted
    immediately (no LLM call), so the expensive reviewer only ever sees the
    WARN/FAIL cases. Picklable, so it ships to worker processes.
    """

    def __init__(self, delegate: Inspector | None = None):
        self.delegate = delegate or RuleInspector()
        self.name = f"triage+{self.delegate.name}"

    def inspect(self, result, report: VerdictReport, history: list) -> Decision:
        if report.overall <= Status.PASS:
            return Decision("accept", None, "clean fit; no review needed", "triage")
        return self.delegate.inspect(result, report, history)


# -- vision LLM ----------------------------------------------------------------
_SYSTEM_PROMPT = """\
You are an expert AGN/quasar spectroscopist performing automated quality control \
on optical spectral decompositions (PyQSOFit-style: power-law + Fe II + host \
galaxy continuum, with tied narrow lines and one or more broad Gaussians per \
Balmer line). You are given, for one object:
  * a diagnostic figure: the data, each fitted component, the total model, and a \
    pull panel (residual / error) with +/-3 sigma bands, and
  * a structured verdict report of deterministic checks (PASS/WARN/FAIL), and
  * a numbered menu of remedies you may apply (a closed vocabulary).

Your job: decide whether the fit is scientifically trustworthy, and if not, pick \
the single best remedy from the menu to improve it, or stop.

You can change the number of BROAD Gaussians and the number of NARROW Gaussians, \
and you can remove a component entirely. Narrow lines get the same scrutiny as \
broad ones: a narrow "line" whose peak sits at the local noise level is a noise \
spike, not a real line — remove it. More than one Gaussian on a weak narrow line \
is over-flexible — reduce it.

Principles:
  * Prefer the simplest model that fits. A broad OR narrow line at low S/N fit with \
    multiple Gaussians is usually over-flexible — reducing the Gaussian count \
    stabilises it.
  * Treat a narrow component flagged as "consistent with noise" as spurious and \
    remove it, unless the figure clearly shows a real, resolved emission peak there.
  * Distinguish *fixable* problems (over-flexible model, missing/spurious component, \
    noise mistaken for a line, bad continuum window) from *intrinsic data limits* \
    (genuinely weak broad line, host-dominated continuum) that no model edit can \
    repair — ACCEPT those rather than thrashing.
  * Look at the pull panel: correlated runs of same-sign residual at a line core \
    mean a missed component or wrong continuum; scattered noise within 3 sigma is fine.
  * Only choose a remedy from the offered menu, by its index. If none would help, \
    ACCEPT (best-effort) or REJECT (if the fit is unusable).

Always finish by calling the submit_review tool exactly once."""

# The structured decision schema, shared by both LLM backends.
_REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["accept", "apply_remedy", "reject"],
            "description": "accept = fit is trustworthy (or best-effort); "
                           "apply_remedy = apply one offered remedy and refit; "
                           "reject = fit is unusable and no remedy helps.",
        },
        "remedy_index": {
            "type": "integer",
            "description": "0-based index into the offered remedies; required "
                           "only when action is apply_remedy.",
        },
        "rationale": {
            "type": "string",
            "description": "One or two sentences citing the specific evidence "
                           "(which check, what the residuals show).",
        },
    },
    "required": ["action", "rationale"],
}
_ANTHROPIC_TOOL = {"name": "submit_review",
                   "description": "Submit your QC decision for this AGN spectral fit.",
                   "input_schema": _REVIEW_SCHEMA}
_OPENAI_TOOL = {"type": "function", "function": {
    "name": "submit_review",
    "description": "Submit your QC decision for this AGN spectral fit.",
    "parameters": _REVIEW_SCHEMA}}


def _build_user_text(result, report: VerdictReport, history: list):
    """The shared review prompt body; returns (text, remedies)."""
    remedies = report.remedies()
    menu = "\n".join(f"  [{i}] {describe_remedy(r)}   {r}"
                     for i, r in enumerate(remedies)) or "  (none offered)"
    hist = "\n".join(
        f"  iter {i}: {getattr(s.decision, 'action', '?')} "
        f"{describe_remedy(s.decision.remedy) if getattr(s.decision, 'remedy', None) else ''}"
        for i, s in enumerate(history)) or "  (first iteration)"
    text = (f"OBJECT: {result.name}  z={result.z:.4f}\n\n"
            f"MEASUREMENTS:\n{result.summary()}\n\n"
            f"VERDICT REPORT:\n{report.summary()}\n\n"
            f"REMEDY MENU (choose by index):\n{menu}\n\n"
            f"HISTORY SO FAR:\n{hist}\n\n"
            "Review the attached diagnostic figure and submit your decision.")
    return text, remedies


def _image_b64(result) -> str | None:
    if not result.figure_path:
        return None
    with open(result.figure_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def _decision_from(action, remedy_index, rationale, remedies, source) -> Decision:
    """Map a parsed (action, remedy_index, rationale) to a Decision."""
    if action == "apply_remedy":
        if remedy_index is None or not (0 <= remedy_index < len(remedies)):
            return Decision("accept", None,
                            f"invalid remedy_index {remedy_index}; accepting. {rationale}",
                            source)
        return Decision("apply", remedies[remedy_index], rationale, source)
    if action == "reject":
        return Decision("reject", None, rationale, source)
    return Decision("accept", None, rationale, source)


class ClaudeInspector:
    """Claude (Anthropic) reads the plot + report and chooses an action/remedy."""

    name = "claude"

    def __init__(self, model: str = "claude-opus-4-8", client=None,
                 api_key: str | None = None, max_tokens: int = 2048):
        self.model = model
        self.max_tokens = max_tokens
        self.api_key = api_key
        self._client = client  # injectable for testing

    def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = (anthropic.Anthropic(api_key=self.api_key)
                            if self.api_key else anthropic.Anthropic())
        return self._client

    def inspect(self, result, report: VerdictReport, history: list) -> Decision:
        user_text, remedies = _build_user_text(result, report, history)
        content = [{"type": "text", "text": user_text}]
        b64 = _image_b64(result)
        if b64:
            content.append({"type": "image", "source": {
                "type": "base64", "media_type": "image/png", "data": b64}})

        resp = self._get_client().messages.create(
            model=self.model, max_tokens=self.max_tokens,
            thinking={"type": "adaptive"},
            system=[{"type": "text", "text": _SYSTEM_PROMPT,
                     "cache_control": {"type": "ephemeral"}}],
            tools=[_ANTHROPIC_TOOL],
            messages=[{"role": "user", "content": content}])

        tool_use = next((b for b in resp.content
                         if getattr(b, "type", None) == "tool_use"
                         and b.name == "submit_review"), None)
        if tool_use is None:
            return Decision("accept", None,
                            "no structured decision returned; defaulting to accept.",
                            self.name)
        inp = tool_use.input
        return _decision_from(inp.get("action", "accept"), inp.get("remedy_index"),
                              inp.get("rationale", ""), remedies, self.name)


class OpenAIInspector:
    """OpenAI (GPT vision) reviewer — the same role as ClaudeInspector.

    Uses chat-completions with a base64 image and the submit_review function so
    the model must return the structured decision. Default model is the
    cost-effective vision model, matching Egent's OpenAI backend.
    """

    name = "openai"

    def __init__(self, model: str = "gpt-4o-mini", client=None,
                 api_key: str | None = None, max_tokens: int = 1024):
        self.model = model
        self.max_tokens = max_tokens
        self.api_key = api_key
        self._client = client  # injectable for testing

    def _get_client(self):
        if self._client is None:
            import openai
            self._client = (openai.OpenAI(api_key=self.api_key)
                            if self.api_key else openai.OpenAI())
        return self._client

    def inspect(self, result, report: VerdictReport, history: list) -> Decision:
        import json
        user_text, remedies = _build_user_text(result, report, history)
        content = [{"type": "text", "text": user_text}]
        b64 = _image_b64(result)
        if b64:
            content.append({"type": "image_url", "image_url": {
                "url": f"data:image/png;base64,{b64}"}})

        resp = self._get_client().chat.completions.create(
            model=self.model, max_tokens=self.max_tokens,
            messages=[{"role": "system", "content": _SYSTEM_PROMPT},
                      {"role": "user", "content": content}],
            tools=[_OPENAI_TOOL],
            tool_choice={"type": "function", "function": {"name": "submit_review"}})

        msg = resp.choices[0].message
        calls = getattr(msg, "tool_calls", None)
        if not calls:
            return Decision("accept", None,
                            "no structured decision returned; defaulting to accept.",
                            self.name)
        try:
            args = json.loads(calls[0].function.arguments)
        except (ValueError, AttributeError):
            return Decision("accept", None, "unparseable decision; accepting.", self.name)
        return _decision_from(args.get("action", "accept"), args.get("remedy_index"),
                              args.get("rationale", ""), remedies, self.name)
