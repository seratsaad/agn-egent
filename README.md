# AGN-Egent: automated AGN spectral decomposition and discovery

Separate the **broad** and **narrow** emission lines of quasar / AGN spectra,
measure black-hole masses and line kinematics, and rank a whole survey by which
objects are unusual. A deterministic engine ([PyQSOFit](https://github.com/legolason/PyQSOFit))
does the fitting; an agentic QC layer decides which fits to trust and which to
fix. Inspired by [Egent](https://github.com/tingyuansen/Egent).

**Try it in the browser:** [seratsaad.github.io/agn-egent](https://seratsaad.github.io/agn-egent).
Upload a spectrum CSV and see Hβ and Hα separated.

---

## What it does

Three layers, each usable on its own:

1. **Fit one spectrum.** Power-law continuum + Fe II + host + tied narrow lines +
   broad Hβ / Hα, then QC checks (S/N, FWHM, χ², residuals). Out comes
   `log M_BH ± err`, `L_bol`, Eddington ratio, and a calibrated trust statement.
2. **Fix the bad fits.** Flagged fits go to an inspector that reads the
   diagnostic plot and picks one edit from a fixed menu. Refit, recheck, repeat.
   Every decision is logged.
3. **Search a survey.** Run thousands of objects, measure line kinematics for
   each, score how badly the model failed, and rank the result into candidate
   lists: disk emitters, outflows, NLS1s, oddities. Repeat spectra of the same
   object can be diffed for variability and changing-look transitions.

```
Spectrum ─► Decompose ─► Verify ─┐
              ▲                   │ flagged?
              │  apply edit ◄── Inspect ◄┘
              └──────────────────────► accept ─► Measure ─► Science ─► Rank
```

---

## Install

```bash
git clone https://github.com/seratsaad/agn-egent.git
cd agn-egent
python3 -m venv .venv

# the PyQSOFit engine (vendored, not committed)
git clone https://github.com/legolason/PyQSOFit.git external/PyQSOFit
./.venv/bin/pip install -U pip setuptools wheel
./.venv/bin/pip install -e external/PyQSOFit
./.venv/bin/pip install -e .
```

Python 3.10+. Optional extras: `pip install -e ".[llm]"` for the LLM inspectors,
`".[app]"` for the Streamlit app.

Pin BLAS to one thread before numpy loads, or fits are not reproducible
(importing `agn_egent` does this for you; a batch launcher should set the env
vars itself):

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
```

---

## Quick start

One object, fully deterministic, no API key, no cost:

```bash
agn-egent 650-52143-166                 # SDSS plate-mjd-fiber (downloads it)
agn-egent --fits spec.fits --z 0.2144   # any SDSS-style FITS
```

```
=== AGN-Egent: 650-52143-166  (z=0.2942) ===
  quality flag : clean
  trust        : quality=clean: P(mass within 0.3 dex) = 100%, typical error ~0.05 dex
  log M_BH     : 8.819 +/- 0.289  [Msun]
  FWHM(Hbeta)  : 7555 km/s
  log L5100    : 44.304   log L_bol: 45.271
  Eddington    : 0.0225
  science      : [OIII] W80=501 v50=-21 km/s | R_FeII=0.02 | flags: broad_hbeta_detected
  anomaly score 1.50 in Hb [ordinary] (continuum control 1.43)
```

Writes `result.json`, `provenance.json` (every config, verdict and decision) and
`diagnostic.png`.

In Python:

```python
from agn_egent import load_sdss, run_agent, RuleInspector, derive

spec = load_sdss("spec-0332-52367-0639.fits")
out = run_agent(spec, inspector=RuleInspector(), finalize_mc=True)
print(derive(out.final_result, "Hb"))   # M_BH +/- err, L_bol, lambda_Edd
print(out.science.summary())            # outflow, R_FeII, profile shape, flags
print(out.anomaly.summary())            # how much the model failed to describe
```

Also: a [Colab notebook](https://colab.research.google.com/github/seratsaad/agn-egent/blob/main/notebooks/run_on_your_fits.ipynb)
and a local Streamlit app ([`app.py`](app.py)).

---

## Survey campaigns

```bash
# fit 300 SDSS quasars, measure everything, rank the odd ones
python scripts/campaign.py --n 300 --workers 8 --outdir campaigns/pilot

# re-rank without refitting; add literature context to the shortlist
python scripts/campaign.py --outdir campaigns/pilot --rank-only --vet
```

Produces a value-added `catalog.csv` (mass, luminosity, Eddington ratio, [O III]
kinematics, R_FeII, profile shape, quality tier, anomaly score for every object)
plus ranked candidate lists.

| Candidate class | Selection |
|---|---|
| `double_peaked` | two resolved peaks in a broad line — disk emitters |
| `extreme_outflow` | [O III] W80 > 1000 km/s or centroid blueshift > 300 km/s |
| `nls1` | FWHM(Hβ) < 2000 km/s with weak [O III] |
| `strong_feii` | R_FeII > 1 (the Eigenvector-1 extreme) |
| `type1_9` | broad Hα but no broad Hβ |
| `offset_broad_line` | broad-line centroid offset > 1000 km/s |
| `anomaly` | coherent residual the model could not fit — unclassified oddities |

Runs are checkpointed per object, so an interrupted campaign resumes instead of
refitting. Every class except `anomaly` is restricted to fits the engine trusts:
an unusual measurement from an untrusted fit is nearly always a bad fit, not a
discovery. For `anomaly` that filter is deliberately off — there, the failed fit
*is* the selection.

Two things a survey search gets wrong unless it is built not to, both found by
running a 300-quasar pilot and looking at what came back:

- **Disk emitters cannot be found in the fitted model.** Given two Gaussians the
  optimizer prefers a narrow core plus a broad pedestal over two offset
  components, so the model profile is single-peaked even when the data are not
  — zero of 300 pilot objects had a two-peaked model. The peak search therefore
  runs on the continuum-, host-, Fe II- and narrow-subtracted **data**.
- **Narrow-line residuals imitate disk horns.** [O III] 5007 sits +8950 km/s
  from Hβ, inside the real horn-separation range, so leftovers from an imperfect
  narrow subtraction read as a second peak; a loose version of the test flagged
  61% of the pilot on exactly that. The narrow-line positions are interpolated
  over before peaks are counted, and the horns must additionally be comparable
  in height with the trough straddling systemic.

---

## Inspector backends

| | `RuleInspector` (default) | `ClaudeInspector` | `OpenAIInspector` | `GeminiInspector` |
|---|---|---|---|---|
| **Type** | deterministic rules | vision LLM | vision LLM | vision LLM |
| **API key** | none | `ANTHROPIC_API_KEY` | `OPENAI_API_KEY` | `GEMINI_API_KEY` |
| **Cost** | free | ~1 request / flagged fit | ~1 request / flagged fit | ~1 request / flagged fit |

All pick from the same fixed edit menu, so results stay reproducible and the
reviewer never writes code or touches the numerics.

**Use rule-first gating** (`run_agent_escalate`, the default for LLM inspectors):
rules run to completion and the LLM sees only the objects they leave flagged.
This matters — see below.

---

## What we know about the agent (measured, not assumed)

Findings from our own ablations, including the ones that did not go our way:

- **Gating is decisive.** Letting the LLM see every non-PASS fit *degrades*
  masses (scatter 0.35 → 0.47 dex; it breaks twice as many fits as it rescues,
  by narrowing genuinely broad lines). Rule-first gating removes that harm and
  halves the LLM's footprint. This is why it is the default.
- **The LLM does not improve per-object mass accuracy at low S/N.** Two
  independent samples agree: the agent is equal to or slightly worse than the
  rule baseline on scatter and outliers.
- **Its value is reliability, not accuracy.** It declines genuine broad-line
  non-detections and flags untrustworthy fits, at the cost of ~25% yield.
- **A cheap model is enough.** Haiku ≈ Opus ≈ Gemini on the QC task.
- **At the population level, on survey-grade (low-S/N) spectra**, LLM review
  restores the mass–luminosity relation the rule baseline loses (N=687: slope
  0.15 → 0.87, scatter 0.49 → 0.34 dex, McNemar p=0.007). This is a
  distribution-level gain, not per-object accuracy.

---

## Validation

| Test | Result |
|---|---|
| **Synthetic injection-recovery** (known truth, N=54, S/N 4–40) | \|Δlog M_BH\| median 0.007 dex, max 0.050 dex |
| **Realistic injection** (known broad line on real host+Fe II backgrounds, N=125) | FWHM recovery median 0.10 dex |
| **Reverberation-mapped AGN** (Grier+2017 SDSS-RM, N=24) | bias **+0.02 dex**, robust scatter 0.27 dex |
| **Hβ vs Hα internal consistency** (N=27) | FWHM ratio 0.92 (cf. 0.90 published), scatter 0.17 dex |
| **Shen DR7 clean sample** (N=108) | ~0.2 dex scatter |
| **Anomaly score null** (white noise, 12 realizations) | 1.09 ± 0.07, max 1.19 (threshold 2.0) |

The RM comparison is the one that matters most: it is the only test against a
mass that is *not* another single-epoch estimate, and it comes out unbiased.

**Trust tiers** are calibrated against the realistic injection run (N=125):

| Tier | P(mass within 0.3 dex) | Typical error |
|---|---|---|
| `clean` | 100% | 0.05 dex |
| `reviewed` | 89% | 0.05 dex |
| `flagged` | 83% | 0.12 dex |

`flagged` is still 83% reliable because the broad-line FWHM is robust — the flag
is deliberately conservative, not a sharp error separator. It selects a more
reliable subsample at high S/N; at low S/N it works as a triage router rather
than a reliability ranking.

---

## Quality thresholds

Deterministic checks (tunable in [`agn_egent/verify.py`](agn_egent/verify.py)):

| Check | FAIL | WARN | Remedy on flag |
|---|---|---|---|
| broad-line S/N | < 3 (not detected) | < 6 (poorly constrained) | drop broad / reduce Gaussians |
| broad FWHM | < 1000 or > 20000 km/s | > 10000 km/s | reduce Gaussians |
| model over-flexibility | n/a | >1 broad Gaussian at S/N < 6 | reduce to 1 Gaussian |
| under-fit broad line | n/a | 1 Gaussian, high S/N, χ² > 1.5 | add a Gaussian |
| reduced χ² | > 5.0 | > 1.5, or < 0.25 (overfit) | reduce Gaussians |
| core residual | n/a | >10% of pixels beyond 3σ | flag (missed component) |
| narrow line is noise | n/a | peak S/N < 3 | remove the component |
| continuum PL slope | n/a | railed at its physical bound | flag (mis-placed continuum) |
| host fraction | n/a | > 0.8 | flag |

Protected lines (tie-anchors, Balmer narrow cores, flux-tied multiplets) are
never removed.

---

## API reference

| Function / class | Purpose |
|---|---|
| `load_sdss`, `load_row_fits`, `fetch_sdss_spectrum` | build a `Spectrum` |
| `QsoparConfig` | the decomposition model as data |
| `decompose(spec, config)` | one deterministic fit |
| `decompose_by_region(spec)` | fit Hβ / Hα separately on local continua |
| `verify(result)` | QC checks → `VerdictReport` |
| `RuleInspector` / `ClaudeInspector` / `TriageInspector` | the QC reviewer |
| `run_agent` / `run_agent_escalate` | the agentic loop (plain / rule-first) |
| `run_batch(sources, …)` | parallel, checkpointed batch → `BatchReport` |
| `derive(result, "Hb")` | `M_BH`, `L_bol`, Eddington ratio |
| `science_report(result)` | [O III] outflow, R_FeII, profile shape, class flags |
| `anomaly_score(result)` | coherent-residual score |
| `trust_statement(flag)` | calibrated reliability of a measurement |
| `select_sdss_targets` / `run_campaign` / `shortlist` | survey campaigns |
| `find_repeat_spectra` / `compare_epochs` | multi-epoch variability, changing-look AGN |
| `simbad_lookup` / `vet_shortlist` | literature context for candidates |
| `query_shen` / `Comparison` | Shen DR7 cross-match |

---

## Notes and limits

- Pin threads and run one fit per process for reproducibility. Bit-identical
  refits require a BLAS that honors thread pinning (OpenBLAS/MKL — all Linux
  wheels). The numpy ≥ 2.0 **macOS** wheels link Accelerate, which does not,
  so a *degenerate* fit (the kind the QC flags anyway) can drift a few percent
  between runs on a Mac; well-constrained quantities are stable everywhere.
- The continuum is **always a pure power law** (`poly=False`); the polynomial
  term is the biggest source of bad L5100 / M_BH and is off for that reason.
  Fe II, Balmer continuum and host are separate additive components.
- Optical only (Hβ / Hα), so z ≲ 0.8. A UV backend (Mg II, C IV) is the main
  thing needed to make this a full survey tool, and is not built yet.
- Line shapes are measured on the fitted model, so they are only as good as the
  fit. [O III] shapes carry a `reliable` flag that goes false when a component
  drifts off-systemic or sits at noise level — without it, an outflow search
  mostly returns bad fits.
- Selection caveat: campaigns built on the Shen catalog inherit its requirement
  of a detected broad line, so they cannot find objects with no broad line at
  all (true Type 2s, or the "off" state of a changing-look AGN).
- The masking A/B test — where a vision LLM should genuinely beat rules, by
  seeing sky residuals and cosmic rays a rule cannot localize — is built
  (`add_wave_mask`) but not yet run.

---

## Citation

AGN-Egent adapts **Egent** (Ting, Mahmud Saad, Liu & Shen 2025) to AGN spectra:

```bibtex
@ARTICLE{2025arXiv251201270T,
       author = {{Ting}, Yuan-Sen and {Mahmud Saad}, Serat and {Liu}, Fan and {Shen}, Yuting},
        title = "{Egent: An Autonomous Agent for Equivalent Width Measurement}",
      journal = {arXiv e-prints},
         year = 2025,
        eprint = {2512.01270},
 primaryClass = {astro-ph.IM}}
```

Built on **PyQSOFit** (Guo, Shen & Wang 2018) and the **Shen et al. (2011)** SDSS
DR7 catalog.

## License

[MIT](LICENSE).
