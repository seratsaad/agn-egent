# AGN-Egent: LLM-Powered AGN Spectral Decomposition

Separate the **broad** and **narrow** emission lines of quasar / AGN spectra and
measure black-hole masses. A deterministic engine ([PyQSOFit](https://github.com/legolason/PyQSOFit))
does the fitting; an LLM inspects every fit and fixes bad ones. Inspired by
[Egent](https://github.com/tingyuansen/Egent).

**Try it in the browser:** [seratsaad.github.io/agn-egent](https://seratsaad.github.io/agn-egent).
Upload a spectrum CSV and see Hβ and Hα separated.

---

## How it works

Two stages, per object:

1. **Fit + check.** Fit the continuum, narrow lines, and broad Hβ / Hα. Run QC
   checks (S/N, FWHM, χ², residuals).
2. **Agentic QC.** If a fit is flagged, an inspector reads the diagnostic plot
   and picks one fix from a fixed menu. Refit and recheck until it passes. Every
   decision is logged.

```
Spectrum ─► Decompose ─► Verify ─┐
              ▲                   │ flagged?
              │  apply edit ◄── Inspect ◄┘
              └──────────────────────► accept ─► Measure (M_BH ± err)
```

The inspector is a rule baseline (no API key), Claude, or OpenAI. All pick from
the same fixed edit menu, so results stay reproducible.

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
./.venv/bin/pip install -r requirements.txt
```

Python 3.10+. For reproducible fits, pin threads before numpy loads (the
env-vars below; `run_agn.py` also does this for you). For the LLM inspector, set
`ANTHROPIC_API_KEY` or `OPENAI_API_KEY`.

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
```

---

## Quick start

```bash
# an SDSS spec FITS on disk
python run_agn.py external/PyQSOFit/example/data/spec-0332-52367-0639.fits

# download an SDSS quasar by plate-mjd-fiber, with the LLM reviewer
python run_agn.py 388-51793-445 --inspector claude

# a non-SDSS spectrum, give the redshift
python run_agn.py j0950_hbeta.fits --z 0.2144 --flux-scale 1e17

# batch
python run_agn.py --list targets.txt --workers 4
```

In Python:

```python
import config; config.pin_threads()          # before numpy
from agn_egent import load_sdss, run_agent, RuleInspector, derive

spec = load_sdss("external/PyQSOFit/example/data/spec-0332-52367-0639.fits")
outcome = run_agent(spec, inspector=RuleInspector(), finalize_mc=True)
print(derive(outcome.final_result, "Hb"))     # M_BH +/- error, L_bol, lambda_Edd
```

Also available: a [Colab notebook](https://colab.research.google.com/github/seratsaad/agn-egent/blob/main/notebooks/run_on_your_fits.ipynb)
(full pipeline, no install) and a local Streamlit app ([`app.py`](app.py)).

---

## Inspector backends

| | `RuleInspector` (default) | `ClaudeInspector` | `OpenAIInspector` |
|---|---|---|---|
| **Type** | deterministic rules | vision LLM | vision LLM |
| **Model** | n/a | `claude-opus-4-8` | `gpt-4o-mini` |
| **API key** | none | `ANTHROPIC_API_KEY` | `OPENAI_API_KEY` |
| **Cost** | free | ~1 request / flagged fit | ~1 request / flagged fit |
| **Best for** | batch, CI, baseline | hard cases | hard cases |

```python
from agn_egent import make_inspector
insp = make_inspector("openai", api_key="sk-...")   # or "claude", or "rule"
```

`TriageInspector` wraps any of them so only flagged fits reach the LLM.

---

## Input formats

| Input | How |
|---|---|
| **SDSS spec FITS** | `load_sdss(path)` |
| **SDSS by plate-mjd-fiber** | `fetch_sdss_spectrum(plate, mjd, fiber)` (downloads via astroquery) |
| **Generic instrument** | `load_row_fits(path, z=..., flux_scale=...)`, a 2-D `[wavelength, flux, err]` array |

---

## Output

Per object, under `data/runs/<name>/`:

- **`provenance.json`**: every config, verdict, and decision.
- **`diagnostic.png`**: data, fitted components, total model, residual panel.
- **Measurements**: broad FWHM, flux, EW, S/N (with MC errors), L5100, and
  derived `log M_BH ± err`, `log L_bol`, Eddington ratio.

Batch mode also writes `results.csv` / `results.json` with quality tiers: `clean`
(passed), `reviewed` (accepted with warnings), `flagged` (needs attention).

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
| narrow over-flexibility | n/a | >1 Gaussian on a weak narrow line | reduce to 1 Gaussian |
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
| `decompose_by_region(spec)` | fit Hβ / Hα separately |
| `verify(result)` | QC checks → `VerdictReport` |
| `RuleInspector` / `ClaudeInspector` / `TriageInspector` | the QC reviewer |
| `run_agent(spec, inspector, …)` | the full agentic loop |
| `run_batch(sources, …)` | parallel batch → `BatchReport` |
| `derive(result, "Hb")` | `M_BH`, `L_bol`, Eddington ratio |
| `query_shen` / `Comparison` | Shen DR7 literature cross-match |

---

## Validation

| Test | Result |
|---|---|
| **Synthetic injection-recovery** | broad-line FWHM recovered to 3–4%, flux to ~9% |
| **Benchmark vs Shen DR7** (N=20) | median Δlog M_BH +0.10 dex, scatter 0.10 dex, slope 0.88 |
| **Quality tiers** | clean 20% · reviewed 75% · flagged 5% |
| **Heterogeneous instrument** (HET/LRS2) | runs end-to-end; broad Hβ matches the manual fit |

```bash
python scripts/benchmark_shen.py --n 30                    # baseline
python scripts/benchmark_shen.py --n 30 --inspector claude # with LLM reviewer
```

---

## Notes

- Pin threads and run one fit per process for reproducibility.
- The reviewer only edits the model from a fixed vocabulary; it never writes code
  or touches the numerics.
- The continuum is a pure power law by default (`poly=False`); the polynomial
  term is the biggest source of bad L5100/M_BH and is off for that reason.
- Optical only (Hβ / Hα). A UV backend is future work.

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

---

## License

[MIT](LICENSE).
