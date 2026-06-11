"""Phase 3 check: the LLM-in-the-seat QC loop.

Part A (offline, always runs): drive the loop with the deterministic
RuleInspector on the degenerate example object and confirm the agent converges,
applies the over-flexibility remedy, and records a full, auditable trajectory.

Part B (offline, always runs): drive the loop with ClaudeInspector backed by a
*mock* Anthropic client, exercising the real vision-message construction and
tool-call parsing path without a network/key. Asserts the agent applies the
remedy the mock "Claude" chose.

Part C (live, only if ANTHROPIC_API_KEY is set): one real ClaudeInspector call.

Run thread-pinned (see README) for reproducibility.
"""
import os
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)

import agn_egent  # noqa: E402  (import first -> pin threads before numpy)
from agn_egent import (load_sdss, run_agent, RuleInspector, ClaudeInspector,  # noqa: E402
                       Status, QsoparConfig)

EXAMPLE_SPEC = os.path.join(
    PROJ, "external", "PyQSOFit", "example", "data", "spec-0332-52367-0639.fits")


# --- a mock Anthropic client that mimics a Claude tool-call response ----------
class _Block:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _Resp:
    def __init__(self, content):
        self.content = content


class MockClaude:
    """Returns a submit_review tool call: apply remedy 0 if any, else accept."""

    def __init__(self):
        self.calls = 0
        self.messages = self

    def create(self, **kwargs):
        self.calls += 1
        # Did the user content include the remedy menu with at least one item?
        user = kwargs["messages"][0]["content"]
        text = next(b["text"] for b in user if b["type"] == "text")
        has_image = any(b.get("type") == "image" for b in user)
        assert has_image, "ClaudeInspector did not attach the diagnostic image"
        offered = "[0]" in text.split("REMEDY MENU")[1] if "REMEDY MENU" in text else False
        if offered:
            inp = {"action": "apply_remedy", "remedy_index": 0,
                   "rationale": "mock: broad line over-flexible per pull panel"}
        else:
            inp = {"action": "accept", "rationale": "mock: no fixable issue remains"}
        return _Resp([_Block(type="tool_use", name="submit_review", input=inp)])


def part_a():
    print("=== Part A: RuleInspector (deterministic) ===")
    spec = load_sdss(EXAMPLE_SPEC, name="phase3a")
    outcome = run_agent(spec, inspector=RuleInspector(), max_iterations=4,
                        verbose=True)
    print("\n" + outcome.summary())

    assert outcome.status in ("accepted", "rejected"), "agent did not converge"
    assert outcome.n_iterations >= 2, "expected at least one remedy iteration"
    # the first decision should apply the ngauss remedy
    first = outcome.steps[0].decision
    assert first.action == "apply" and first.remedy and \
        first.remedy.get("action") == "set_ngauss", \
        f"expected first action to apply set_ngauss, got {first}"
    # final verdict should be no worse, and Hb over-flexibility flag gone
    final_hb_overflex = [c for c in outcome.final_verdict.checks
                         if c.name == "Hb.model_flexibility" and c.status >= Status.WARN]
    assert not final_hb_overflex, "over-flexibility flag should be cleared at the end"

    p = outcome.save(os.path.join(PROJ, "data", "runs", "agent_phase3a", "provenance.json"))
    print(f"[ok] provenance saved -> {p}")
    print("[PASS] Part A")
    return outcome


def part_b():
    print("\n=== Part B: ClaudeInspector with a mock client (offline) ===")
    spec = load_sdss(EXAMPLE_SPEC, name="phase3b")
    mock = MockClaude()
    inspector = ClaudeInspector(client=mock)
    outcome = run_agent(spec, inspector=inspector, max_iterations=4, verbose=True)
    print("\n" + outcome.summary())

    assert mock.calls >= 1, "mock Claude was never called"
    assert outcome.steps[0].decision.source == "claude"
    assert outcome.steps[0].decision.action == "apply", \
        "mock Claude should have applied a remedy on the flagged fit"
    assert outcome.status in ("accepted", "rejected")
    print("[PASS] Part B")


def part_c():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("\n=== Part C: live ClaudeInspector — SKIPPED (no ANTHROPIC_API_KEY) ===")
        return
    print("\n=== Part C: live ClaudeInspector (claude-opus-4-8) ===")
    spec = load_sdss(EXAMPLE_SPEC, name="phase3c")
    outcome = run_agent(spec, inspector=ClaudeInspector(), max_iterations=4,
                        verbose=True)
    print("\n" + outcome.summary())
    assert outcome.status in ("accepted", "rejected")
    print("[PASS] Part C (live)")


if __name__ == "__main__":
    part_a()
    part_b()
    part_c()
    print("\n[PASS] Phase 3: agentic QC loop works (deterministic + LLM paths).")
