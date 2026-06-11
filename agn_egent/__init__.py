"""AGN-Egent: agentic AGN spectral decomposition.

Phase 1 exposes the deterministic pipeline:

    from agn_egent import decompose, load_sdss, QsoparConfig

The agentic QC layer (Phase 3) is built on top of this surface.
"""
# Pin BLAS threads BEFORE numpy is imported anywhere below, for reproducible
# fits. See agn_egent.repro for why this must precede the numpy import.
from . import repro as _repro
_repro.pin_threads()

from .spectrum import Spectrum
from .backends.base import DecompositionResult, LineMeasurement, FitBackend
from .backends.qsopar_config import QsoparConfig, LinePrior
from .pipeline import decompose, get_backend
from .io_sdss import load_sdss
from .verify import verify, VerdictReport, Check, Status
from .remedies import apply_remedy, describe_remedy
from .plotting import render_diagnostic
from .agent import (Decision, Inspector, RuleInspector, ClaudeInspector,
                    OpenAIInspector, TriageInspector, make_inspector,
                    AgentStep, AgentOutcome, run_agent,
                    ClaudeContinuumPlanner, make_continuum_planner)
from .continuum import (ContinuumPlan, RuleContinuumPlanner, CANDIDATE_WINDOWS,
                        window_stats, anchored_pl_fit, apply_plan_to_config,
                        render_continuum_plan)
from .regions import (Region, DEFAULT_OPTICAL_REGIONS, decompose_by_region,
                      RegionDecomposition, merge_region_results,
                      render_region_diagnostic, trim_spectrum)
from .batch import run_batch, BatchReport, BatchRow
from .io_generic import load_row_fits
from .io_sdss import spectrum_from_hdulist
from .synthetic import make_synthetic_spectrum
from .measure import derive, DerivedQuantities, black_hole_mass_hbeta
from .catalog import (query_shen, find_shen_quasars, fetch_sdss_spectrum,
                      ShenRecord, Comparison)

__all__ = [
    "Spectrum",
    "DecompositionResult",
    "LineMeasurement",
    "FitBackend",
    "QsoparConfig",
    "LinePrior",
    "decompose",
    "get_backend",
    "load_sdss",
    "verify",
    "VerdictReport",
    "Check",
    "Status",
    "apply_remedy",
    "describe_remedy",
    "render_diagnostic",
    "Decision",
    "Inspector",
    "RuleInspector",
    "ClaudeInspector",
    "OpenAIInspector",
    "TriageInspector",
    "make_inspector",
    "AgentStep",
    "AgentOutcome",
    "run_agent",
    "run_batch",
    "BatchReport",
    "BatchRow",
    "load_row_fits",
    "make_synthetic_spectrum",
    "derive",
    "DerivedQuantities",
    "black_hole_mass_hbeta",
    "spectrum_from_hdulist",
    "query_shen",
    "find_shen_quasars",
    "fetch_sdss_spectrum",
    "ShenRecord",
    "Comparison",
    "ContinuumPlan",
    "RuleContinuumPlanner",
    "ClaudeContinuumPlanner",
    "make_continuum_planner",
    "CANDIDATE_WINDOWS",
    "window_stats",
    "anchored_pl_fit",
    "apply_plan_to_config",
    "render_continuum_plan",
    "Region",
    "DEFAULT_OPTICAL_REGIONS",
    "decompose_by_region",
    "RegionDecomposition",
    "merge_region_results",
    "render_region_diagnostic",
    "trim_spectrum",
]
