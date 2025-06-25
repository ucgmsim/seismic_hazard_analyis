"""
Example for computing fault and distributed seismicity hazard
using empirical GMMs for the 2010 NZ NSHM.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import oq_wrapper as oqw
import qcore.nhm as nhm
import seismic_hazard_analysis as sha
from qcore import coordinates as coords

# Periods to compute hazard for
PERIODS = [
    0.01,
    0.02,
    0.03,
    0.04,
    0.05,
    0.075,
    0.1,
    0.12,
    0.15,
    0.17,
    0.2,
    0.25,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.75,
    0.8,
    0.9,
    1.0,
    1.25,
    1.2,
    1.5,
    2.0,
    2.5,
    3.0,
    4.0,
    5.0,
    6.0,
    7.5,
    10.0,
]
# GMMs to use for each tectonic type
GMM_MAPPING = {
    oqw.constants.TectType.ACTIVE_SHALLOW: oqw.constants.GMMLogicTree.NSHM2022,
    oqw.constants.TectType.SUBDUCTION_SLAB: oqw.constants.GMMLogicTree.NSHM2022,
    oqw.constants.TectType.SUBDUCTION_INTERFACE: oqw.constants.GMMLogicTree.NSHM2022,
}
ims = [f"pSA_{cur_period}" for cur_period in PERIODS]


# Site details
site_coords = np.array([[-43.5381, 172.6474, 0]])
site_nztm = coords.wgs_depth_to_nztm(site_coords)[0, [1, 0, 2]]
site_properties = {
    "vs30": 180.7414,
    "vs30measured": True,
    "z1p0": 0.337,
    "z2p5": 5.75,
    "backarc": False,
}

# Load the ERF files
background_ffp = (
    Path(__file__).parent.parent.parent
    / "data/NSHM2010"
    / "NZBCK2015_Chch50yearsAftershock_OpenSHA_modType4.txt"
)
ds_erf_ffp = (
    Path(__file__).parent.parent.parent / "data/NSHM2010" / "NZ_DSmodel_2015.txt"
)
fault_erf_ffp = (
    Path(__file__).parent.parent.parent / "data/NSHM2010" / "NZ_FLTmodel_2010.txt"
)

ds_erf_df = pd.read_csv(ds_erf_ffp, index_col="rupture_name")
ds_source_df = sha.nshm_2010.get_ds_source_df(background_ffp)

flt_definitions = nhm.load_nhm(fault_erf_ffp)
flt_erf_df = nhm.load_nhm_df(str(fault_erf_ffp))

### DS Hazard
ds_hazard = sha.nshm_2010.compute_gmm_ds_hazard(
    ds_source_df,
    ds_erf_df,
    site_nztm,
    site_properties,
    GMM_MAPPING,
    ims,
)

### Fault Hazard
flt_hazard = sha.nshm_2010.compute_gmm_flt_hazard(
    site_nztm,
    site_properties,
    flt_erf_df,
    GMM_MAPPING,
    ims,
    flt_definitions=flt_definitions,
)

### Plot
plot_im = "pSA_5.0"
fig = plt.figure(figsize=(16, 10))

plt.plot(
    flt_hazard[plot_im].index.values, flt_hazard[plot_im].values, label="Fault", c="b"
)
plt.plot(ds_hazard[plot_im].index.values, ds_hazard[plot_im].values, label="DS", c="k")
plt.plot(
    ds_hazard[plot_im].index.values,
    ds_hazard[plot_im].values + flt_hazard[plot_im].values,
    label="Total",
    c="r",
)

plt.xlabel(f"{plot_im}")
plt.ylabel("Annual Exceedance Probability")

plt.legend()
plt.xscale("log")
plt.yscale("log")

fig.tight_layout()

plt.show()

