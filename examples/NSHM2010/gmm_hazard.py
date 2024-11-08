"""
Example for computing fault and distributed seismicity hazard
using empirical GMMs for the 2010 NZ NSHM.
"""
from pathlib import Path

import pandas as pd
import numpy as np
import qcore.nhm as nhm
from qcore import coordinates as coords
import seismic_hazard_analysis as sha
import matplotlib.pyplot as plt

from empirical.util.classdef import TectType, GMM

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
    TectType.ACTIVE_SHALLOW: GMM.Br_13,
    TectType.SUBDUCTION_SLAB: GMM.ZA_06,
    TectType.SUBDUCTION_INTERFACE: GMM.ZA_06,
}
ims = [f"pSA_{cur_period}" for cur_period in PERIODS]

# Site details
site_coords = np.array([[-43.53113953, 172.63661180, 0]])
site_nztm = coords.wgs_depth_to_nztm(site_coords)[0, [1, 0, 2]]
site_vs30 = 500
site_z1p0 = 0.6

# Load the ERF files
background_ffp = (
    Path(__file__).parent / "NZBCK2015_Chch50yearsAftershock_OpenSHA_modType4.txt"
)
ds_erf_ffp = Path(__file__).parent / "NZ_DSModel_2015.txt"
fault_erf_ffp = Path(__file__).parent / "NZ_FLTModel_2010.txt"

ds_erf_df = pd.read_csv(ds_erf_ffp, index_col="rupture_name")

flt_erf = nhm.load_nhm(fault_erf_ffp)
flt_erf_df = nhm.load_nhm_df(str(fault_erf_ffp))

### DS Hazard
ds_rupture_df = sha.nshm_2010.get_ds_rupture_df(
    background_ffp, site_nztm, site_vs30, site_z1p0
)
ds_gm_params_df = sha.nshm_2010.get_emp_gm_params(ds_rupture_df, GMM_MAPPING, PERIODS)
ds_hazard = sha.nshm_2010.compute_gmm_hazard(
    ds_gm_params_df, ds_erf_df.annual_rec_prob, ims
)

### Fault Hazard
# Create fault objects
faults = {
    cur_name: sha.nshm_2010.get_fault_objects(cur_fault)
    for cur_name, cur_fault in flt_erf.items()
}

flt_rupture_df = sha.nshm_2010.get_flt_rupture_df(
    faults, flt_erf_df, site_nztm, site_vs30, site_z1p0
)
flt_gm_params_df = sha.nshm_2010.get_emp_gm_params(flt_rupture_df, GMM_MAPPING, PERIODS)
flt_hazard = sha.nshm_2010.compute_gmm_hazard(
    flt_gm_params_df, 1 / flt_erf_df["recur_int_median"], ims
)

### Plot
plot_im = "pSA_1.0"
fig = plt.figure(figsize=(16, 10))

plt.plot(flt_hazard[plot_im].index.values, flt_hazard[plot_im].values, label="Fault")
plt.plot(ds_hazard[plot_im].index.values, ds_hazard[plot_im].values, label="DS")
plt.plot(
    ds_hazard[plot_im].index.values,
    ds_hazard[plot_im].values + flt_hazard[plot_im].values,
    label="Total",
)
plt.xlabel(f"{plot_im}")
plt.ylabel("Annual Exceedance Probability")

plt.legend()
plt.xscale("log")
plt.yscale("log")

fig.tight_layout()
plt.show()

print(f"wtf")
