"""AGN-Egent — interactive web app (Streamlit).

The live functional twin of the static GitHub Pages showcase: pick or download a
spectrum, run the agentic decomposition, and see the broad/narrow separation,
the measurements, the single-epoch black-hole mass, and the agent's QC decisions.

Deploy on Streamlit Community Cloud (GitHub Pages cannot run it — it needs a
Python server). Locally:

    pip install streamlit
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
    VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 streamlit run app.py
"""
import os
import tempfile

import config
config.pin_threads()                      # before numpy

import streamlit as st

st.set_page_config(page_title="AGN-Egent", page_icon="🌌", layout="wide")


@st.cache_resource
def _api():
    import agn_egent
    from agn_egent import (load_sdss, load_row_fits, fetch_sdss_spectrum,
                           QsoparConfig, run_agent, make_inspector,
                           derive, render_diagnostic)
    return dict(load_sdss=load_sdss, load_row_fits=load_row_fits,
                fetch=fetch_sdss_spectrum, Cfg=QsoparConfig, run_agent=run_agent,
                make_inspector=make_inspector, derive=derive, render=render_diagnostic)


EX = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                  "external", "PyQSOFit", "example", "data")
EXAMPLES = {
    "SDSS example (spec-0332)": os.path.join(EX, "spec-0332-52367-0639.fits"),
    "SDSS example (spec-0266a)": os.path.join(EX, "spec-0266-51602-0013.fits"),
    "SDSS example (spec-0266b)": os.path.join(EX, "spec-0266-51602-0107.fits"),
}

st.title("AGN-Egent 🌌")
st.caption("LLM-powered AGN spectral decomposition — separate broad & narrow lines, "
           "measure black-hole masses, with agentic quality control.")

api = _api()

with st.sidebar:
    st.header("Input")
    source = st.radio("Source", ["Example object", "SDSS plate-mjd-fiber", "Upload FITS"])
    spec = None
    if source == "Example object":
        name = st.selectbox("Object", list(EXAMPLES))
        spec_path = EXAMPLES[name]
    elif source == "SDSS plate-mjd-fiber":
        pmf = st.text_input("plate-mjd-fiber", "388-51793-445")
    else:
        up = st.file_uploader("FITS file", type=["fits"])
        zz = st.number_input("redshift (generic FITS only)", value=0.0, format="%.4f")
        fs = st.number_input("flux scale", value=1.0, format="%.0e")

    st.header("QC reviewer")
    backend = st.selectbox("Reviewer", ["Rule (no API key)", "Claude (Anthropic)", "OpenAI (GPT)"])
    api_key, model = None, None
    if backend.startswith("Claude"):
        api_key = st.text_input("Anthropic API key", type="password",
                                value=os.environ.get("ANTHROPIC_API_KEY", ""))
        model = st.text_input("model", value="claude-opus-4-8")
    elif backend.startswith("OpenAI"):
        api_key = st.text_input("OpenAI API key", type="password",
                                value=os.environ.get("OPENAI_API_KEY", ""))
        model = st.text_input("model", value="gpt-4o-mini")

    st.header("Pipeline")
    mc = st.toggle("Monte-Carlo error bars", value=True)
    ngauss = st.select_slider("Broad Hβ Gaussians", [1, 2, 3], value=2)
    go = st.button("Decompose", type="primary")

if go:
    try:
        if source == "Example object":
            spec = api["load_sdss"](spec_path)
        elif source == "SDSS plate-mjd-fiber":
            p, m, f = (int(x) for x in pmf.split("-"))
            with st.spinner("Downloading SDSS spectrum…"):
                spec = api["fetch"](p, m, f)
        elif up is not None:
            tmp = os.path.join(tempfile.gettempdir(), up.name)
            open(tmp, "wb").write(up.getbuffer())
            try:
                spec = api["load_sdss"](tmp)
            except Exception:
                spec = api["load_row_fits"](tmp, z=zz, flux_scale=fs)
        if spec is None:
            st.warning("Provide an input spectrum.")
            st.stop()

        cfg = api["Cfg"].sdss_optical_default().set_ngauss("Hb_br", ngauss)
        name = {"R": "rule", "C": "claude", "O": "openai"}[backend[0]]
        if name != "rule" and not api_key:
            st.info("No API key entered — using the deterministic rule reviewer.")
            name = "rule"
        inspector = api["make_inspector"](name, api_key=api_key or None, model=model)

        with st.spinner("Decomposing + agentic QC…"):
            outcome = api["run_agent"](spec, inspector=inspector, config=cfg,
                                       finalize_mc=mc, nsamp=25,
                                       workdir=os.path.join("data", "runs", f"app_{spec.name}"))
        res = outcome.final_result
        fig = api["render"](res, os.path.join("data", "runs", f"app_{spec.name}", "diag.png"))

        st.subheader(f"{spec.name}  ·  z = {spec.z:.4f}  ·  {outcome.status}")
        st.image(fig)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Line measurements**")
            rows = []
            for comp, mm in res.lines.items():
                rows.append({"line": comp, "FWHM [km/s]": round(mm.fwhm_kms),
                             "± err": None if mm.fwhm_err != mm.fwhm_err else round(mm.fwhm_err),
                             "flux": round(mm.flux, 1), "S/N": round(mm.snr, 1)})
            st.dataframe(rows, hide_index=True)
        with c2:
            st.markdown("**Derived quantities**")
            dq = api["derive"](res, "Hb")
            if dq:
                err = "" if dq.log_MBH_err != dq.log_MBH_err else f" ± {dq.log_MBH_err:.2f}"
                st.metric("log M_BH / M☉", f"{dq.log_MBH:.2f}{err}")
                st.write(f"log L₅₁₀₀ = {dq.log_L5100:.2f} · log L_bol = {dq.log_Lbol:.2f} "
                         f"· λ_Edd = {dq.eddington_ratio:.3f}")
            else:
                st.write("No broad Hβ detected (narrow-line only).")

        st.markdown("**Agent QC trajectory**")
        for s in outcome.steps:
            rem = f" → {s.decision.remedy}" if s.decision.remedy else ""
            st.write(f"- iter {s.iteration}: **{s.verdict.overall}** → "
                     f"`{s.decision.action}`{rem} — {s.decision.rationale}")
    except Exception as e:
        st.error(f"{type(e).__name__}: {e}")
else:
    st.info("Choose an input in the sidebar and click **Decompose**.")
