#!/usr/bin/env python
"""Render a static Hbeta/Halpha per-region example for the static site.

Faithfully reproduces the in-browser fit in docs/index.html (power-law continuum
from line-free side-bands, one broad Gaussian on the masked wings, shared-width
narrow lines) so the showcase figure matches what a visitor gets live. Output:
docs/assets/example-regions.png plus the measured broad FWHMs on stdout.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

C = 299792.458  # km/s
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# a clear broad-line quasar, different from the site's default example
CSV = os.path.join(PROJ, "examples", "agn_388-51793-445_z0.3157.csv")
Z = 0.3157
OBJ = "SDSS 388-51793-445"
OUT = os.path.join(PROJ, "docs", "assets", "example-regions.png")

LINES = {
    "Hβ": dict(lam=4862.68, region=(4720, 5110), blue=(4720, 4790), red=(5035, 5105),
               mask=[(4853, 4873), (4951, 4969), (4998, 5016)],
               narrow=[4862.68, 4960.30, 5008.24]),
    "Hα": dict(lam=6562.80, region=(6300, 6850), blue=(6300, 6420), red=(6760, 6850),
               mask=[(6553, 6573), (6543, 6556), (6579, 6591)],
               narrow=[6562.80, 6549.85, 6585.28, 6718.29, 6732.67]),
}
COL = dict(data="#444444", cont="#E69F00", broad="#0072B2", narrow="#009E73", pull="#666666")


def band_med(w, f, lo, hi):
    m = (w >= lo) & (w <= hi) & np.isfinite(f)
    if m.sum() == 0:
        return np.nan, np.nan, 0
    return np.median(w[m]), np.median(f[m]), int(m.sum())


def powerlaw(w, f, blue, red, lam0):
    wb, fb, nb = band_med(w, f, *blue)
    wr, fr, nr = band_med(w, f, *red)
    if nb < 3 or nr < 3:
        return None
    if fb > 0 and fr > 0:
        k = np.log(fr / fb) / np.log(wr / wb)
        A = fb / (wb / lam0) ** k
        return lambda x: A * (x / lam0) ** k
    m = (fr - fb) / (wr - wb)
    b = fb - m * wb
    return lambda x: m * x + b


def in_mask(x, masks):
    return any(lo <= x <= hi for lo, hi in masks)


def gauss(x, A, c, s):
    return A * np.exp(-((x - c) ** 2) / (2 * s * s))


def robust_std(a):
    a = np.asarray(a)
    if a.size < 5:
        return np.nan
    m = np.median(a)
    mad = np.median(np.abs(a - m))
    return 1.4826 * mad if mad > 0 else np.nan


def fit_broad(w, r, region, lam, masks):
    keep = np.array([region[0] <= x <= region[1] and np.isfinite(rr) and not in_mask(x, masks)
                     for x, rr in zip(w, r)])
    xs, ys = w[keep], r[keep]
    if xs.size < 12:
        return None
    best = dict(sse=np.inf)
    for s in np.arange(6, 111, 2):
        for c in np.arange(lam - 20, lam + 20 + 1, 2):
            g = np.exp(-((xs - c) ** 2) / (2 * s * s))
            sgg = (g * g).sum()
            A = max(0.0, (ys * g).sum() / sgg) if sgg > 0 else 0.0
            sse = ((ys - A * g) ** 2).sum()
            if sse < best["sse"]:
                best = dict(sse=sse, s=s, c=c, A=A)
    best["fwhm_kms"] = 2.35482 * best["s"] / best["c"] * C
    return best


def fit_narrow(w, r2, region, centers):
    keep = (w >= region[0]) & (w <= region[1]) & np.isfinite(r2)
    xs, ys = w[keep], r2[keep]
    if xs.size < 10:
        return None
    best = dict(sse=np.inf)
    for s in np.arange(1.0, 7.01, 0.5):
        amps = []
        for c in centers:
            g = np.exp(-((xs - c) ** 2) / (2 * s * s))
            sgg = (g * g).sum()
            amps.append(max(0.0, (ys * g).sum() / sgg) if sgg > 0 else 0.0)
        model = np.zeros_like(xs)
        for c, A in zip(centers, amps):
            model += A * np.exp(-((xs - c) ** 2) / (2 * s * s))
        sse = ((ys - model) ** 2).sum()
        if sse < best["sse"]:
            best = dict(sse=sse, s=s, amps=amps)
    return dict(sig=best["s"], lines=list(zip(centers, best["amps"])))


def decompose(name, cfg, w, f, z):
    rest = w / (1 + z)
    cont = powerlaw(rest, f, cfg["blue"], cfg["red"], cfg["lam"])
    resid = f - cont(rest)
    broad = fit_broad(rest, resid, cfg["region"], cfg["lam"], cfg["mask"])
    bvec = gauss(rest, broad["A"], broad["c"], broad["s"]) if broad else 0
    narrow = fit_narrow(rest, resid - bvec, cfg["region"], cfg["narrow"])

    sel = (rest >= cfg["region"][0]) & (rest <= cfg["region"][1])
    x = rest[sel]
    cc = cont(x)
    by = cc + (gauss(x, broad["A"], broad["c"], broad["s"]) if broad else 0)
    ny = cc.copy()
    if narrow:
        for c, A in narrow["lines"]:
            ny = ny + gauss(x, A, c, narrow["sig"])
    fy = f[sel]
    nm = ny - cc
    res = fy - (by + nm)   # data - total model (continuum + broad + narrow)
    inband = ((x >= cfg["blue"][0]) & (x <= cfg["blue"][1])) | ((x >= cfg["red"][0]) & (x <= cfg["red"][1]))
    sig = robust_std(res[inband])
    if not np.isfinite(sig):
        sig = robust_std(res)
    if not np.isfinite(sig) or sig <= 0:
        sig = 1.0
    pull = res / sig
    return dict(x=x, data=fy, cont=cc, broad=by, narrow=ny, pull=pull,
                fwhm=broad["fwhm_kms"] if broad else None)


def main():
    arr = np.genfromtxt(CSV, delimiter=",", names=True)
    w, f = np.asarray(arr["wavelength"], float), np.asarray(arr["flux"], float)

    fig = plt.figure(figsize=(11.2, 4.6), dpi=150)
    fig.patch.set_facecolor("white")
    outer = gridspec.GridSpec(1, 2, wspace=0.22, left=0.06, right=0.985, top=0.9, bottom=0.12)
    fwhms = {}
    for col, (name, cfg) in enumerate(LINES.items()):
        d = decompose(name, cfg, w, f, Z)
        fwhms[name] = d["fwhm"]
        gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[col],
                                              height_ratios=[3.4, 1], hspace=0.06)
        ax, axp = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
        ax.plot(d["x"], d["data"], color=COL["data"], lw=0.9, label="data")
        ax.plot(d["x"], d["cont"], color=COL["cont"], lw=2, label="continuum (power law)")
        ax.plot(d["x"], d["broad"], color=COL["broad"], lw=2, label="broad")
        ax.plot(d["x"], d["narrow"], color=COL["narrow"], lw=1.6, label="narrow")
        ax.set_title("%s broad FWHM ≈ %s km/s" % (name, format(round(d["fwhm"]), ",")),
                     fontsize=13, color="#222")
        ax.set_ylabel("flux", fontsize=11)
        ax.set_xticklabels([])
        ax.legend(fontsize=8.5, loc="upper right", framealpha=0.6, edgecolor="#e5e8ee")
        ax.grid(alpha=0.25)
        axp.plot(d["x"], d["pull"], color=COL["pull"], lw=0.9)
        axp.axhline(0, color="#999", lw=1)
        axp.axhline(3, color="#D55E00", lw=1, ls="--")
        axp.axhline(-3, color="#D55E00", lw=1, ls="--")
        axp.set_ylim(-8, 8)
        axp.set_ylabel("pull", fontsize=11)
        axp.set_xlabel("rest wavelength (Å)", fontsize=11)
        axp.grid(alpha=0.25)
        for a in (ax, axp):
            a.set_xlim(cfg["region"])
            a.set_facecolor("white")

    fig.savefig(OUT, facecolor="white")
    print("wrote", OUT)
    print("object", OBJ, "z=%.4f" % Z)
    for name, v in fwhms.items():
        print("  %s broad FWHM = %d km/s" % (name, round(v)))


if __name__ == "__main__":
    main()
