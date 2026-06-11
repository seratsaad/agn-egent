# Example AGN spectra

Real SDSS quasar spectra as plain CSV (`wavelength`, `flux`), observed-frame
wavelength in Å, flux in SDSS units (10⁻¹⁷ erg s⁻¹ cm⁻² Å⁻¹). Use them to try the
[website](https://seratsaad.github.io/agn-egent) (upload a file and enter the
redshift) or the pipeline. The **redshift is in each filename**.

| file | z | notes |
|---|---|---|
| `agn_650-52143-166_z0.2942.csv` | 0.2942 | clean broad Hβ + Hα |
| `agn_421-51821-307_z0.1718.csv` | 0.1718 | high-S/N, very broad lines |
| `agn_651-52141-535_z0.2095.csv` | 0.2095 | strong Fe II |
| `agn_2630-54327-149_z0.3293.csv` | 0.3293 | Hβ and Hα both clear |
| `agn_388-51793-445_z0.3157.csv` | 0.3157 | luminous quasar |

All are at z ≈ 0.17–0.33 so both Hβ and Hα fall in the optical range. Above
z ≈ 0.4, Hα redshifts out of the SDSS coverage and only Hβ is shown.

```python
import config; config.pin_threads()
from agn_egent import load_row_fits  # or read the CSV yourself
import numpy as np
w, f = np.loadtxt("examples/agn_650-52143-166_z0.2942.csv",
                  delimiter=",", skiprows=1, unpack=True)
```

Files are derived from public SDSS DR7/DR16 spectra (Shen et al. 2011 sample).
