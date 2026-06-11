"""PyQSOFit backend: wraps the engine and extracts a backend-agnostic result.

This is the only module that imports PyQSOFit. It turns a :class:`Spectrum` +
:class:`QsoparConfig` into a :class:`DecompositionResult` with faithfully
reconstructed component arrays (host, continuum, Fe II, broad, narrow).
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import numpy as np

from .base import DecompositionResult, LineMeasurement
from .qsopar_config import QsoparConfig


def _to_float(x, default=np.nan):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


class PyQSOFitBackend:
    """Optical AGN decomposition via PyQSOFit (host+FeII+PL+tied line complexes).

    Reproducibility: PyQSOFit's MLE fit is numerically sensitive for weak,
    under-constrained components (e.g. a low-SNR broad Hbeta with 2 Gaussians).
    Multithreaded BLAS makes the reduction order non-deterministic, which tips
    such degenerate fits into different local minima across processes. We pin
    BLAS to a single thread and seed numpy so every fit is byte-reproducible.
    Parallelism for batch runs should be at the object level (one process each).
    """

    name = "pyqsofit"

    def __init__(self, seed: int = 0):
        self.seed = seed

    def decompose(self, spectrum, config: QsoparConfig, workdir: str,
                  make_figure: bool = True) -> DecompositionResult:
        import numpy as _np
        from pyqsofit.PyQSOFit import QSOFit

        os.makedirs(workdir, exist_ok=True)
        config.write_qsopar(workdir)  # PyQSOFit reads qsopar.fits from `path`

        fit_kwargs = dict(config.fit_options)
        # wave_range / wave_mask are Fit() positional options, not fit kwargs we
        # can splat blindly (they would collide with the explicit args below).
        # Pull them out so region-restricted ("local") fits and mask remedies work.
        wave_range = fit_kwargs.pop("wave_range", None)
        wave_mask = fit_kwargs.pop("wave_mask", None)
        plot_kwargs = {"save_fig_path": workdir} if make_figure else {}
        fig_path = os.path.join(workdir, f"{spectrum.name or 'fit'}.pdf") if make_figure else None

        # Thread pinning for reproducibility is done at process launch (env vars
        # before numpy import; see agn_egent.repro). We do NOT use a runtime
        # threadpool context here -- entering/exiting it perturbs pool state
        # between consecutive in-process fits and breaks reproducibility.
        _np.random.seed(self.seed)
        q = QSOFit(spectrum.wave, spectrum.flux, spectrum.err, spectrum.z,
                   ra=spectrum.ra if spectrum.ra is not None else -999,
                   dec=spectrum.dec if spectrum.dec is not None else -999,
                   path=workdir)
        q.Fit(name=spectrum.name or None,
              wave_range=np.asarray(wave_range, dtype=float) if wave_range is not None else None,
              wave_mask=np.asarray(wave_mask, dtype=float) if wave_mask is not None else None,
              save_result=False, plot_fig=make_figure,
              kwargs_plot=plot_kwargs, save_fits_name=None, **fit_kwargs)

        return self._extract(q, spectrum, config,
                             fig_path if (make_figure and os.path.exists(fig_path)) else None)

    # -- result extraction -----------------------------------------------------
    def _extract(self, q, spectrum, config, fig_path) -> DecompositionResult:
        wave = np.asarray(q.wave, dtype=float)            # rest-frame wavelength
        data = np.asarray(q.flux, dtype=float)            # host-subtracted QSO flux

        comps: dict[str, np.ndarray] = {"data": data}
        if hasattr(q, "err") and np.ndim(q.err) == 1 and len(q.err) == len(wave):
            comps["err"] = np.asarray(q.err, dtype=float)

        def add(key, attr):
            if hasattr(q, attr):
                v = getattr(q, attr)
                if v is not None and np.ndim(v) == 1 and len(v) == len(wave):
                    comps[key] = np.asarray(v, dtype=float)

        add("host", "host")
        add("qso", "qso")
        add("conti", "f_conti_model")
        add("pl", "f_pl_model")
        add("feii_uv", "f_fe_mgii_model")
        add("feii_op", "f_fe_balmer_model")
        add("balmer", "f_bc_model")
        add("poly", "f_poly_model")
        add("line_total", "f_line_model")

        broad, narrow = self._reconstruct_lines(q, wave)
        if broad is not None:
            comps["broad"] = broad
            comps["narrow"] = narrow

        conti = comps.get("conti", np.zeros_like(wave))
        line_total = comps.get("line_total", np.zeros_like(wave))
        comps["model"] = conti + line_total
        comps["residual"] = data - comps["model"]

        lines = self._line_measurements(q)
        continuum = self._continuum_dict(q)
        quality = self._quality_dict(q)
        params = self._param_dict(q)
        err = comps.get("err")
        narrow_lines = self._narrow_line_measurements(q, wave, err, config)

        return DecompositionResult(
            name=spectrum.name, z=spectrum.z, backend=self.name,
            rest_wave=wave, components=comps, lines=lines,
            continuum=continuum, quality=quality, params=params,
            config=config.to_dict(), figure_path=fig_path,
            narrow_lines=narrow_lines,
        )

    def _narrow_line_measurements(self, q, wave, err, config) -> dict:
        """Per narrow-line-component peak amplitude + a noise SNR estimate.

        SNR is the fitted line's peak amplitude divided by the local 1-sigma
        noise (median err within +/-25 A of the peak). It needs no Monte-Carlo,
        so it is available during the fast MLE loop -- a narrow Gaussian that has
        latched onto a noise spike has amplitude ~ the noise (SNR ~ 1), while a
        real forbidden line stands many sigma above it.
        """
        names = getattr(q, "gauss_result_name", None)
        gres = getattr(q, "gauss_result", None)
        if names is None or gres is None or len(names) == 0 or err is None:
            return {}
        names = np.asarray(names)
        gres = np.asarray(gres, dtype=float)
        lookup = {n: gres[i] for i, n in enumerate(names)}
        lnwave = np.log(wave)

        # which linenames are velocity/width anchors of a tie group (vindex/windex
        # shared): the first line carrying a given nonzero index is the free one
        # others lock to. Removing an anchor would break the tie, so we mark them.
        anchors, seen_v, comp_of, ng_of = set(), {}, {}, {}
        findex_count: dict = {}
        for l in config.lines:
            comp_of[l.linename] = l.compname
            ng_of[l.linename] = l.ngauss
            # tying is within a complex, so the anchor of a vindex group is keyed
            # by (compname, vindex) -- the first line carrying it in that complex.
            key = (l.compname, l.vindex)
            if l.vindex and key not in seen_v:
                seen_v[key] = l.linename
                anchors.add(l.linename)
            if l.findex:
                findex_count[(l.compname, l.findex)] = \
                    findex_count.get((l.compname, l.findex), 0) + 1
        # a line is "tied" if it shares a nonzero findex (fixed flux ratio) with a
        # sibling -- i.e. a real multiplet member ([N II], [S II], tied [O III]),
        # never a noise spike, so it must not be removed as noise.
        tied_of = {l.linename: bool(l.findex and
                                    findex_count.get((l.compname, l.findex), 0) > 1)
                   for l in config.lines}

        comp_models: dict[str, np.ndarray] = {}
        for n in names:
            if not str(n).endswith("_scale"):
                continue
            base = str(n)[:-len("_scale")]          # e.g. "OIII5007c_1"
            if "_br" in base:
                continue                             # narrow only
            linename = base.rsplit("_", 1)[0]        # strip Gaussian index
            try:
                pp = [lookup[base + "_scale"], lookup[base + "_centerwave"],
                      lookup[base + "_sigma"]]
            except KeyError:
                continue
            g = q.Onegauss(lnwave, pp)
            comp_models.setdefault(linename, np.zeros_like(wave))
            comp_models[linename] += g

        out = {}
        for linename, model in comp_models.items():
            ipk = int(np.argmax(np.abs(model)))
            amp = float(model[ipk])
            cw = float(wave[ipk])
            win = (wave > cw - 25) & (wave < cw + 25)
            e = err[win]
            e = e[np.isfinite(e) & (e > 0)]
            eloc = float(np.median(e)) if len(e) else np.nan
            snr = abs(amp) / eloc if (np.isfinite(eloc) and eloc > 0) else np.nan
            out[linename] = {
                "peak_wave": cw, "amp": amp, "snr": snr,
                "compname": comp_of.get(linename, ""),
                "ngauss": int(ng_of.get(linename, 1)),
                "anchor": linename in anchors,
                "tied": bool(tied_of.get(linename, False)),
            }
        return out

    def _reconstruct_lines(self, q, wave):
        """Sum per-Gaussian models into broad/narrow arrays.

        Broad = component name contains '_br'; everything else is narrow
        (includes '_na' Balmer narrow and forbidden [OIII]/[NII]/[SII]).
        """
        names = getattr(q, "gauss_result_name", None)
        gres = getattr(q, "gauss_result", None)
        if names is None or gres is None or len(names) == 0:
            return None, None
        names = np.asarray(names)
        gres = np.asarray(gres, dtype=float)
        lookup = {n: gres[i] for i, n in enumerate(names)}
        lnwave = np.log(wave)
        broad = np.zeros_like(wave)
        narrow = np.zeros_like(wave)
        for n in names:
            if not n.endswith("_scale"):
                continue
            base = n[:-len("_scale")]
            try:
                pp = [lookup[base + "_scale"], lookup[base + "_centerwave"],
                      lookup[base + "_sigma"]]
            except KeyError:
                continue
            g = q.Onegauss(lnwave, pp)
            if "_br" in base:
                broad += g
            else:
                narrow += g
        return broad, narrow

    def _line_measurements(self, q) -> dict[str, LineMeasurement]:
        fur_names = np.asarray(getattr(q, "fur_result_name", []))
        fur_vals = np.asarray(getattr(q, "fur_result", []))
        fd = {n: _to_float(v) for n, v in zip(fur_names, fur_vals)}
        complexes = list(getattr(q, "uniq_linecomp_sort", []))
        out: dict[str, LineMeasurement] = {}
        for c in complexes:
            c = str(c)
            out[c] = LineMeasurement(
                name=c,
                fwhm_kms=fd.get(f"{c}_whole_br_fwhm", np.nan),
                sigma_kms=fd.get(f"{c}_whole_br_sigma", np.nan),
                flux=fd.get(f"{c}_whole_br_area", np.nan),
                ew_aa=fd.get(f"{c}_whole_br_ew", np.nan),
                peak_wave=fd.get(f"{c}_whole_br_peak", np.nan),
                snr=fd.get(f"{c}_whole_br_snr", np.nan),
                fwhm_err=fd.get(f"{c}_whole_br_fwhm_err", np.nan),
                flux_err=fd.get(f"{c}_whole_br_area_err", np.nan),
                ew_err=fd.get(f"{c}_whole_br_ew_err", np.nan),
                sigma_err=fd.get(f"{c}_whole_br_sigma_err", np.nan),
            )
        return out

    def _continuum_dict(self, q) -> dict[str, float]:
        names = np.asarray(getattr(q, "conti_result_name", []))
        vals = np.asarray(getattr(q, "conti_result", []))
        keep_substr = ("L1350", "L1450", "L3000", "L4200", "L5100",
                       "PL_norm", "PL_slope", "Fe_op_norm", "Fe_uv_norm",
                       "Fe_op_FWHM", "Fe_flux",
                       "frac_host", "SN_host", "Dn4000")
        out = {}
        for n, v in zip(names, vals):
            n = str(n)
            # keep both the value and its MC error (e.g. L5100 and L5100_err)
            base = n[:-4] if n.endswith("_err") else n
            if any(s in base for s in keep_substr):
                fv = _to_float(v)
                if np.isfinite(fv):
                    out[n] = fv
        # PyQSOFit stores L5100 as log10 luminosity; expose both names
        if "L5100" in out:
            out.setdefault("LogL5100", out["L5100"])
        return out

    def _quality_dict(self, q) -> dict[str, float]:
        """Per-complex reduced chi^2.

        PyQSOFit names these ``{i}_line_red_chi2`` where ``i`` is the 1-based
        complex index in ``uniq_linecomp_sort`` (not the complex name).
        """
        names = np.asarray(getattr(q, "line_result_name", []))
        vals = np.asarray(getattr(q, "line_result", []))
        d = {str(n): v for n, v in zip(names, vals)}
        complexes = [str(x) for x in getattr(q, "uniq_linecomp_sort", [])]
        out = {}
        for i, comp in enumerate(complexes, start=1):
            fv = _to_float(d.get(f"{i}_line_red_chi2"))
            if np.isfinite(fv):
                out[comp] = fv
        return out

    def _param_dict(self, q) -> dict[str, float]:
        out = {}
        for src_n, src_v in (("conti_result_name", "conti_result"),
                             ("line_result_name", "line_result")):
            names = np.asarray(getattr(q, src_n, []))
            vals = np.asarray(getattr(q, src_v, []))
            for n, v in zip(names, vals):
                fv = _to_float(v)
                if np.isfinite(fv):
                    out[str(n)] = fv
        return out
