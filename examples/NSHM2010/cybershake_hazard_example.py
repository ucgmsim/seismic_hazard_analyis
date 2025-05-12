"""
Example of computing PSHA for a single site based on
CyberShake simulation results.

If running with a complete CyberShake dataset, this script
will consume a large amount of memory (i.e. use hypocentre).
"""
from pathlib import Path

import matplotlib.pyplot as plt

import seismic_hazard_analysis as sha
from qcore import nhm

# Config
im_dir = Path("/path/to/cybershake/im/data")
fault_erf_ffp = Path(__file__).parent.parent.parent / "data/NSHM2010" / "NZ_FLTModel_2010.txt"

# Load ERF & IM data
flt_erf_df = nhm.load_nhm_df(str(fault_erf_ffp))
fault_im_data = sha.nshm_2010.load_sim_im_data(im_dir)

# Compute site hazard
site_im_df = sha.nshm_2010.get_sim_site_ims(fault_im_data, "PIPS")
im_hazard = sha.nshm_2010.compute_sim_hazard(site_im_df, flt_erf_df)

# Plot
im = "pSA_1.0"
fig = plt.figure(figsize=(16, 10))

plt.plot(im_hazard[im].index.values, im_hazard[im].values)
plt.xlabel(f"{im}")
plt.ylabel("Annual Exceedance Probability")

plt.xscale("log")
plt.yscale("log")

fig.tight_layout()
plt.show()

