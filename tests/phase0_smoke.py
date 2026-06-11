"""
Phase 0 smoke test: prove the full PyQSOFit decomposition path end-to-end on a
known SDSS quasar (PyQSOFit's bundled example, spec-0332-52367-0639, z~0.146).

Generates a minimal optical-focused qsopar.fits, runs an MLE fit with host
decomposition + Fe II + power-law + line complexes, and reports the headline
broad-Hbeta measurement. Success criterion: a finite Hbeta broad FWHM and a
finite L5100, plus a saved diagnostic figure.
"""
import os
import matplotlib
matplotlib.use("Agg")
import numpy as np
from astropy.io import fits
from astropy.table import Table

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
EXT = os.path.join(PROJ, "external", "PyQSOFit")
EXAMPLE_SPEC = os.path.join(EXT, "example", "data", "spec-0332-52367-0639.fits")
WORKDIR = os.path.join(PROJ, "data", "phase0")
os.makedirs(WORKDIR, exist_ok=True)


def build_qsopar(path):
    """Write the line + continuum prior tables PyQSOFit reads as qsopar.fits.

    Columns follow the canonical PyQSOFit schema. The agent will later mutate
    `ngauss` (broad component count) and the v/w/f-index tying columns.
    """
    line_priors = np.rec.array([
        (6564.61, 'Ha', 6400, 6800, 'Ha_br',   2, 0.0, 0.0, 1e10, 5e-3, 0.004,  0.05,    0.015, 0, 0, 0, 0.05,  1),
        (6564.61, 'Ha', 6400, 6800, 'Ha_na',   1, 0.0, 0.0, 1e10, 1e-3, 5e-4,   0.00169, 0.01,  1, 1, 0, 0.002, 1),
        (6549.85, 'Ha', 6400, 6800, 'NII6549', 1, 0.0, 0.0, 1e10, 1e-3, 2.3e-4, 0.00169, 5e-3,  1, 1, 1, 0.001, 1),
        (6585.28, 'Ha', 6400, 6800, 'NII6585', 1, 0.0, 0.0, 1e10, 1e-3, 2.3e-4, 0.00169, 5e-3,  1, 1, 1, 0.003, 1),
        (6718.29, 'Ha', 6400, 6800, 'SII6718', 1, 0.0, 0.0, 1e10, 1e-3, 2.3e-4, 0.00169, 5e-3,  1, 1, 2, 0.001, 1),
        (6732.67, 'Ha', 6400, 6800, 'SII6732', 1, 0.0, 0.0, 1e10, 1e-3, 2.3e-4, 0.00169, 5e-3,  1, 1, 2, 0.001, 1),
        (4862.68, 'Hb', 4640, 5100, 'Hb_br',     2, 0.0, 0.0, 1e10, 5e-3, 0.004,  0.05,    0.01,  0, 0, 0, 0.01,  1),
        (4862.68, 'Hb', 4640, 5100, 'Hb_na',     1, 0.0, 0.0, 1e10, 1e-3, 2.3e-4, 0.00169, 0.01,  1, 1, 0, 0.002, 1),
        (4960.30, 'Hb', 4640, 5100, 'OIII4959c', 1, 0.0, 0.0, 1e10, 1e-3, 2.3e-4, 0.00169, 0.01,  1, 1, 0, 0.002, 1),
        (5008.24, 'Hb', 4640, 5100, 'OIII5007c', 1, 0.0, 0.0, 1e10, 1e-3, 2.3e-4, 0.00169, 0.01,  1, 1, 0, 0.004, 1),
        (4960.30, 'Hb', 4640, 5100, 'OIII4959w', 1, 0.0, 0.0, 1e10, 3e-3, 2.3e-4, 0.004,   0.01,  2, 2, 0, 0.001, 1),
        (5008.24, 'Hb', 4640, 5100, 'OIII5007w', 1, 0.0, 0.0, 1e10, 3e-3, 2.3e-4, 0.004,   0.01,  2, 2, 0, 0.002, 1),
        ],
        formats='float32, a20, float32, float32, a20, int32, float32, float32, float32, '
                'float32, float32, float32, float32, int32, int32, int32, float32, int32',
        names='lambda, compname, minwav, maxwav, linename, ngauss, inisca, minsca, maxsca, '
              'inisig, minsig, maxsig, voff, vindex, windex, findex, fvalue, vary')
    hdr1 = fits.Header()
    hdu1 = fits.BinTableHDU(data=line_priors, header=hdr1, name='line_priors')

    conti_windows = np.rec.array([
        (3775., 3832.), (4000., 4050.), (4200., 4230.), (4435., 4640.),
        (5100., 5535.), (6005., 6035.), (6110., 6250.), (6800., 7000.),
        ], formats='float32, float32', names='min, max')
    hdu2 = fits.BinTableHDU(data=conti_windows, name='conti_windows')

    conti_priors = np.rec.array([
        ('Fe_uv_norm',  0.0,   0.0,   1e10,  1),
        ('Fe_uv_FWHM',  3000,  1200,  18000, 1),
        ('Fe_uv_shift', 0.0,   -0.01, 0.01,  1),
        ('Fe_op_norm',  0.0,   0.0,   1e10,  1),
        ('Fe_op_FWHM',  3000,  1200,  18000, 1),
        ('Fe_op_shift', 0.0,   -0.01, 0.01,  1),
        ('PL_norm',     1.0,   0.0,   1e10,  1),
        ('PL_slope',    -1.5,  -5.0,  3.0,   1),
        ('Blamer_norm', 0.0,   0.0,   1e10,  1),
        ('Balmer_Te',   15000, 10000, 50000, 1),
        ('Balmer_Tau',  0.5,   0.1,   2.0,   1),
        ('conti_a_0',   0.0,   None,  None,  1),
        ('conti_a_1',   0.0,   None,  None,  1),
        ('conti_a_2',   0.0,   None,  None,  1),
        ], formats='a20, float32, float32, float32, int32',
        names='parname, initial, min, max, vary')
    hdu3 = fits.BinTableHDU(data=conti_priors, name='conti_priors')

    measure_info = Table(
        [[[1350, 1450, 3000, 4200, 5100]], [[[4435, 4685]]]],
        names=('cont_loc', 'Fe_flux_range'), dtype=('float32', 'float32'))
    hdu4 = fits.BinTableHDU(data=measure_info, name='measure_info')

    fits.HDUList([fits.PrimaryHDU(), hdu1, hdu2, hdu3, hdu4]).writeto(
        os.path.join(path, 'qsopar.fits'), overwrite=True)
    print(f"[ok] wrote qsopar.fits -> {path}")


def main():
    from pyqsofit.PyQSOFit import QSOFit
    QSOFit.set_mpl_style()

    build_qsopar(WORKDIR)

    data = fits.open(EXAMPLE_SPEC)
    lam = 10 ** data[1].data['loglam']
    flux = data[1].data['flux']
    err = 1 / np.sqrt(data[1].data['ivar'])
    z = data[2].data['z'][0]
    ra = data[0].header['plug_ra']
    dec = data[0].header['plug_dec']
    print(f"[ok] loaded {os.path.basename(EXAMPLE_SPEC)}: z={z:.4f}, "
          f"{len(lam)} pix, {lam.min():.0f}-{lam.max():.0f} A")

    q = QSOFit(lam, flux, err, z, ra=ra, dec=dec, path=WORKDIR)
    print(f"[ok] template install_path: {q.install_path}")

    q.Fit(name='phase0', nsmooth=1, deredden=True, reject_badpix=True,
          wave_range=None, wave_mask=None,
          decompose_host=True, host_prior=False, decomp_na_mask=True,
          npca_gal=5, npca_qso=10, qso_type='CZBIN1',
          Fe_uv_op=True, poly=True, rej_abs_conti=False, rej_abs_line=True,
          MC=False, linefit=True, save_result=True,
          kwargs_plot={'save_fig_path': WORKDIR}, save_fits_path=WORKDIR,
          save_fits_name='phase0_result', verbose=True)

    print("\n=== RESULTS ===")
    # Continuum luminosity at 5100A
    names = list(q.conti_result_name)
    vals = list(q.conti_result)
    cd = dict(zip(names, vals))
    for k in cd:
        if 'L5100' in k or 'LogL5100' in k:
            print(f"  {k} = {cd[k]}")
    # Broad Hbeta measurements
    lnames = list(q.line_result_name)
    lvals = list(q.line_result)
    ld = dict(zip(lnames, lvals))
    for k in lnames:
        if 'Hb' in k and ('whole_br_fwhm' in k or 'whole_br_area' in k or 'br_fwhm' in k):
            print(f"  {k} = {ld[k]}")
    # Headline check
    fwhm_keys = [k for k in lnames if 'Hb_whole_br_fwhm' in k]
    if fwhm_keys:
        fwhm = float(ld[fwhm_keys[0]])
        print(f"\n[PASS] Hbeta broad FWHM = {fwhm:.0f} km/s" if np.isfinite(fwhm) and fwhm > 0
              else f"\n[FAIL] Hbeta broad FWHM not finite: {fwhm}")
    else:
        print("\n[WARN] no Hb_whole_br_fwhm key; available line keys:")
        print("  ", [k for k in lnames if 'Hb' in k])


if __name__ == "__main__":
    main()
