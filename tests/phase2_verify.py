"""Phase 2 check: deterministic verify -> remedy -> re-fit loop.

On the known-degenerate example object (weak 2-Gaussian broad Hbeta), the
verifier should flag Hbeta and propose a structured remedy; applying it and
re-fitting should clear the flag. This is the exact control loop the Phase 3
agent will drive, here with deterministic logic in the seat.

Run thread-pinned (see README) for reproducibility.
"""
import os
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)

import agn_egent  # noqa: E402  (import first -> pin threads before numpy)
from agn_egent import (load_sdss, decompose, QsoparConfig, verify,  # noqa: E402
                       apply_remedy, describe_remedy, Status)

EXAMPLE_SPEC = os.path.join(
    PROJ, "external", "PyQSOFit", "example", "data", "spec-0332-52367-0639.fits")


def main():
    spec = load_sdss(EXAMPLE_SPEC, name="phase2")
    config = QsoparConfig.sdss_optical_default()

    # --- initial fit + verify -------------------------------------------------
    res0 = decompose(spec, config=config, make_figure=False,
                     workdir=os.path.join(PROJ, "data", "runs", "phase2_0"))
    rep0 = verify(res0)
    print("=== initial decomposition ===")
    print(res0.summary())
    print("\n" + rep0.summary())

    assert rep0.overall >= Status.WARN, "expected the degenerate Hbeta to be flagged"
    hb_flags = [c for c in rep0.checks
                if c.target == "Hb" and c.status >= Status.WARN]
    assert hb_flags, "expected at least one Hbeta warning/failure"
    print(f"\n[check] Hbeta flagged by {len(hb_flags)} check(s): "
          f"{[c.name for c in hb_flags]}")

    # --- pick the worst remedy and apply it -----------------------------------
    remedies = rep0.remedies()
    assert remedies, "verifier produced no actionable remedy"
    remedy = remedies[0]
    print(f"\n[action] applying remedy: {describe_remedy(remedy)}  {remedy}")
    config2 = apply_remedy(config, remedy)

    # --- re-fit + re-verify ---------------------------------------------------
    res1 = decompose(spec, config=config2, make_figure=False,
                     workdir=os.path.join(PROJ, "data", "runs", "phase2_1"))
    rep1 = verify(res1)
    print("\n=== after remedy ===")
    print(res1.summary())
    print("\n" + rep1.summary())

    # the Hbeta situation must improve: fewer/lower-severity Hb flags
    def hb_severity(rep):
        sev = [c.status for c in rep.checks if c.target == "Hb"]
        return max(sev) if sev else Status.PASS

    before, after = hb_severity(rep0), hb_severity(rep1)
    print(f"\n[check] Hbeta worst severity: {before!s} -> {after!s}")
    assert after < before or len(
        [c for c in rep1.checks if c.target == "Hb" and c.status >= Status.WARN]
    ) < len(hb_flags), "remedy did not improve the Hbeta verdict"

    print("\n[PASS] Phase 2: verify -> remedy -> re-fit loop improves the fit.")


if __name__ == "__main__":
    main()
