"""Apply a structured remedy (from a :class:`~agn_egent.verify.Check`) to a config.

Remedies are plain dicts so they are easy to log, diff, and (in Phase 3) emit
from an LLM as JSON. ``apply_remedy`` is the deterministic interpreter that
turns one into a new :class:`QsoparConfig`. Keeping this mapping in one place
gives a closed vocabulary of edits the agent is allowed to make.
"""
from __future__ import annotations

from .backends.qsopar_config import QsoparConfig


def apply_remedy(config: QsoparConfig, remedy: dict) -> QsoparConfig:
    """Return a new config with `remedy` applied. Unknown actions are no-ops."""
    action = remedy.get("action")
    if action == "set_ngauss":
        return config.set_ngauss(remedy["linename"], remedy["value"])
    if action == "drop_broad":
        return config.drop_broad(remedy["compname"])
    if action == "remove_line":
        return config.remove_line(remedy["linename"])
    if action == "set_fit_option":
        return config.set_fit_option(remedy["key"], remedy["value"])
    if action == "add_wave_mask":
        lo, hi = remedy["range"]
        existing = config.fit_options.get("wave_mask")
        masks = [list(m) for m in existing] if existing is not None else []
        masks.append([float(lo), float(hi)])
        return config.set_fit_option("wave_mask", masks)
    return config  # unknown / no-op (e.g. {"action": "none"})


def describe_remedy(remedy: dict) -> str:
    a = remedy.get("action")
    if a == "set_ngauss":
        return f"set {remedy['linename']} broad Gaussians -> {remedy['value']}"
    if a == "drop_broad":
        return f"drop broad component of {remedy['compname']}"
    if a == "remove_line":
        return f"remove line {remedy['linename']}"
    if a == "set_fit_option":
        return f"set fit option {remedy['key']} = {remedy['value']}"
    if a == "add_wave_mask":
        return f"mask rest-frame {remedy['range']} A"
    return str(remedy)
