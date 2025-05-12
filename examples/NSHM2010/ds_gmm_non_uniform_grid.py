"""
Example script for computing DS hazard using GMM for all non-uniform grid sites.
Breaks the set of sites into batches, and process one batch at a time.
Execution of each batch is parallelized using multiprocessing.
"""
import multiprocessing as mp
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

import seismic_hazard_analysis as sha
from empirical.util.classdef import GMM, TectType
from qcore import coordinates as coords

### Config
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

batch_size = 1000
n_procs = 32

background_ffp = Path("/Users/claudy/dev/work/data/gm_hazard/sources/18p6/archive/NZBCK211_OpenSHA.txt")
ds_erf_ffp = Path("/Users/claudy/dev/work/data/gm_hazard/sources/18p6/archive/NZ_DSmodel_2010.txt")
sites_ffp = Path("/Users/claudy/dev/work/data/gm_hazard/sites/23p1/non_uniform_whole_nz_with_real_stations-hh400_v20p3_land.ll")
vs30_ffp = Path("/Users/claudy/dev/work/data/gm_hazard/sites/23p1/non_uniform_whole_nz_with_real_stations-hh400_v20p3_land.vs30")
z_ffp = Path("/Users/claudy/dev/work/data/gm_hazard/sites/23p1/non_uniform_whole_nz_with_real_stations-hh400_v20p3_land.z")
out_dir = Path("/Users/claudy/dev/work/tmp/ds_hazard_test")

# background_ffp = Path("/home/cbs51/dev/work/data/gm_hazard/sources/18p6/archive/NZBCK211_OpenSHA.txt")
# ds_erf_ffp = Path("/home/cbs51/dev/work/data/gm_hazard/sources/18p6/archive/NZ_DSmodel_2010.txt")
# sites_ffp = "/home/cbs51/dev/work/data/gm_hazard/sites/23p1/non_uniform_whole_nz_with_real_stations-hh400_v20p3_land.ll"
# vs30_ffp = "/home/cbs51/dev/work/data/gm_hazard/sites/23p1/non_uniform_whole_nz_with_real_stations-hh400_v20p3_land.vs30"
# z_ffp = "/home/cbs51/dev/work/data/gm_hazard/sites/23p1/non_uniform_whole_nz_with_real_stations-hh400_v20p3_land.z"
# out_dir = Path("/home/cbs51/dev/work/tmp/ds_hazard_test")
###

def _process_site(site_id: str, site_df: pd.DataFrame, ds_source_df: pd.DataFrame, ds_erf_df: pd.DataFrame, gmm_mapping: dict, ims: Sequence[str]) -> dict | None:
    """Helper function to process a single site."""
    print(f"Processing site {site_id}")
    try:
        cur_ds_hazard = sha.nshm_2010.compute_gmm_ds_hazard(
            ds_source_df,
            ds_erf_df,
            site_df.loc[site_id, ["nztm_x", "nztm_y", "depth"]].values,
            site_df.loc[site_id, "vs30"],
            site_df.loc[site_id, "z1p0"],
            gmm_mapping,
            ims,
        )
    except Exception as e:
        print(f"Error processing site {site_id}: {e}")
        return None
    else:
        return cur_ds_hazard

def _process_batch(sites: np.ndarray[str], site_df: pd.DataFrame, ds_source_df: pd.DataFrame, ds_erf_df: pd.DataFrame, gmm_mapping: dict, ims: Sequence[str], out_dir: Path, batch_ix: int, n_procs: int = 4):
    """Helper function to process a batch of sites."""
    out_file_name = f"ds_hazard_batch_{batch_ix}.pkl"
    if (out_dir / out_file_name).exists():
        print(f"Skipping batch {batch_ix} as it already exists")
        return

    start = time.time()
    with mp.Pool(n_procs) as pool:
        results = pool.starmap(_process_site, [(cur_site, site_df, ds_source_df, ds_erf_df, gmm_mapping, ims) for cur_site in sites])
    print(f"Took: {time.time() - start} to run {sites.size} sites")

    # Combine the results
    comb_results = {}
    for cur_im in ims:
        cur_result = pd.concat([cur_res[cur_im] for cur_res in results if cur_res is not None], axis=1)
        cur_result.columns = sites
        comb_results[cur_im] = cur_result

    # Save the results
    pd.to_pickle(comb_results, out_dir / out_file_name)
    np.save(out_dir / f"ds_hazard_sites_{batch_ix}.npy", sites)



# Load the ERF files
ds_erf_df = pd.read_csv(ds_erf_ffp, index_col="rupture_name")
ds_source_df = sha.nshm_2010.get_ds_source_df(background_ffp)

# Load the site data
site_df = pd.read_csv(
    sites_ffp,
    sep=" ",
    header=None,
    names=["lon", "lat", "site_id"],
    index_col="site_id",
)
vs30_df = pd.read_csv(
    vs30_ffp, sep=" ", header=None, names=["site_id", "vs30"], index_col="site_id"
)
z_df = pd.read_csv(z_ffp, index_col="Station_Name")

# Combine the site information
site_df = pd.merge(site_df, vs30_df, left_index=True, right_index=True, how="inner")
site_df = pd.merge(site_df, z_df, left_index=True, right_index=True, how="inner")
site_df = site_df.rename(columns={"Z_1.0(km)": "z1p0"})

# Convert to NZTM
site_df[["nztm_x", "nztm_y"]] = coords.wgs_depth_to_nztm(site_df[["lat", "lon"]].values)[:, [1,0]]
site_df["depth"] = 0

sites = site_df.index.values.astype(str)

print("Processing batches")
n_batches = int(np.ceil(sites.size / batch_size))
for cur_batch_ix in range(n_batches):
    cur_batch_sites = sites[cur_batch_ix * batch_size : (cur_batch_ix + 1) * batch_size]
    _process_batch(cur_batch_sites, site_df, ds_source_df, ds_erf_df, GMM_MAPPING, ims, out_dir, cur_batch_ix, n_procs=32)

# Combine results
print("Combining results")
batch_result_ffps = list(out_dir.glob("ds_hazard_batch_*.pkl"))
batch_results = [pd.read_pickle(cur_ffp) for cur_ffp in batch_result_ffps]

im_results = {}
for cur_im in ims:
    im_results[cur_im] = pd.concat([cur_res[cur_im] for cur_res in batch_results], axis=1)

# Save the results
pd.to_pickle(im_results, out_dir / "ds_hazard.pkl")