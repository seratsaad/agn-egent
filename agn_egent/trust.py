"""Per-object trust statement for a measured mass.

The 3-tier quality flag is regime-dependent, so a bare "flagged"/"clean" is not, by
itself, an honest reliability statement. Here we attach a *calibrated* probability
that the broad-line mass is recovered to better than a threshold (default 0.3 dex),
learned from the realistic injection-recovery (known truth; see
scripts/paper/trust_calibration.py). If the calibration file is unavailable we fall
back to conservative built-in values.

Usage:
    from agn_egent.trust import trust_statement
    print(trust_statement("flagged"))
"""
from __future__ import annotations

import json
import os

# Conservative built-in calibration (fraction of fits recovered to <0.3 dex), used
# if paper/results/trust_calibration.json is absent. Updated from the realistic
# injection-recovery run.
_FALLBACK = {
    "reliable_dex": 0.3,
    "by_flag": {   # from realistic injection-recovery (N=125, known truth)
        "clean": {"p_reliable": 1.00, "median_err_dex": 0.05},
        "reviewed": {"p_reliable": 0.89, "median_err_dex": 0.05},
        "flagged": {"p_reliable": 0.83, "median_err_dex": 0.12},
    },
}


def _load():
    """Load the calibration that ships inside the package.

    It lives in ``agn_egent/data/`` rather than alongside the analysis scripts
    that produced it, so an installed copy of the package can still report
    calibrated reliability instead of silently dropping to the fallback.
    """
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "data", "trust_calibration.json")
    if os.path.exists(p):
        try:
            with open(p) as fh:
                return json.load(fh)
        except Exception:
            pass
    return _FALLBACK


def reliability(quality_flag: str, cal: dict | None = None) -> dict | None:
    """Return {p_reliable, median_err_dex, reliable_dex} for a flag, or None."""
    cal = cal or _load()
    rec = cal.get("by_flag", {}).get(quality_flag)
    if rec is None:
        return None
    return {"p_reliable": rec["p_reliable"],
            "median_err_dex": rec["median_err_dex"],
            "reliable_dex": cal.get("reliable_dex", 0.3)}


def trust_statement(quality_flag: str) -> str:
    r = reliability(quality_flag)
    if r is None:
        return f"quality={quality_flag} (uncalibrated)"
    return (f"quality={quality_flag}: P(mass within {r['reliable_dex']} dex) "
            f"= {r['p_reliable']:.0%}, typical error ~{r['median_err_dex']:.2f} dex")
