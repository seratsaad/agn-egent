"""The input data model shared by every backend."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Spectrum:
    """A 1-D spectrum in the *observed* frame.

    Flux/err are in units of 1e-17 erg/s/cm^2/A (SDSS convention). Backends are
    responsible for de-redshifting and any flux rescaling they need internally.
    """

    wave: np.ndarray              # observed-frame wavelength [Angstrom]
    flux: np.ndarray              # flux density [1e-17 erg/s/cm^2/A]
    err: np.ndarray               # 1-sigma error, same units as flux
    z: float                      # redshift
    name: str = ""                # human/catalog identifier
    ra: float | None = None       # degrees
    dec: float | None = None      # degrees
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.wave = np.asarray(self.wave, dtype=float)
        self.flux = np.asarray(self.flux, dtype=float)
        self.err = np.asarray(self.err, dtype=float)
        n = len(self.wave)
        if not (len(self.flux) == len(self.err) == n):
            raise ValueError(
                f"wave/flux/err length mismatch: {n}/{len(self.flux)}/{len(self.err)}")
        if n == 0:
            raise ValueError("empty spectrum")

    @property
    def rest_wave(self) -> np.ndarray:
        return self.wave / (1.0 + self.z)

    def covers(self, rest_wavelength: float) -> bool:
        """Whether the given rest-frame wavelength falls inside the coverage."""
        rw = self.rest_wave
        return rw.min() <= rest_wavelength <= rw.max()

    def __repr__(self):
        return (f"Spectrum(name={self.name!r}, z={self.z:.4f}, n={len(self.wave)}, "
                f"obs=[{self.wave.min():.0f},{self.wave.max():.0f}]A)")
