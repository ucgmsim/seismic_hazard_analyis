"""
Module for computing the seismic hazard for the
New Zealand 2010 Seismic Hazard Model using
the results from physics-based GM simulations
"""
from typing import Sequence
from pathlib import Path

import pandas as pd
import numpy as np
import xarray as xr
from tqdm import tqdm

from .. import utils
from .. import hazard


def load_sim_im_data(im_data_dir: Path):
    """
    Loads the IM data for each fault

    Note: Using a dataset as the
    number of realisations and the stations
    with data varies per fault, making a 4D
    unsuitable.

    Parameters
    ----------
    im_data_dir: Path
        Directory that contains a folder for
        each fault, which contains the IM data
        csv files.

    Returns
    -------
    fault_im_dict: dict
        The IM data for each fault as a DataArray
    """
    # Available faults
    faults = [cur_dir.stem for cur_dir in im_data_dir.iterdir() if cur_dir.is_dir()]

    fault_im_dict = {}
    for cur_fault in tqdm(faults):
        cur_im_files = (im_data_dir / cur_fault).rglob("*REL*.csv")

        # Create DataArray for each fault
        cur_im_data, cur_rel_names = [], []
        cur_stations, cur_IMs = None, None
        for ix, cur_im_file in enumerate(cur_im_files):
            cur_im_df = pd.read_csv(cur_im_file, index_col=0)

            if ix == 0:
                cur_stations = cur_im_df.index
                cur_IMs = cur_im_df.columns[1:]
            else:
                assert np.all(cur_stations == cur_im_df.index)
                assert np.all(cur_IMs == cur_im_df.columns[1:])

            cur_rel_names.append(cur_im_file.stem.rsplit("_", 1)[1])
            cur_im_data.append(cur_im_df[cur_IMs].values)

        cur_im_array = xr.DataArray(
            data=np.stack(cur_im_data, axis=-1),
            dims=("station", "IM", "realisation"),
            coords={
                "station": cur_stations,
                "IM": cur_IMs,
                "realisation": cur_rel_names,
            },
        )
        fault_im_dict[cur_fault] = cur_im_array

    return fault_im_dict


def get_sim_site_ims(fault_im_dict: dict[str, xr.DataArray], site: str):
    """
    Get the IM data for a specific site

    Parameters
    ----------
    fault_im_dict: dict
        The IM data for each fault as a DataArray
    site: str

    Returns
    -------
    pd.DataFrame
        The IM data for the specified site
        with a multi-index of fault and realisation
    """
    # Get data per fault and convert to DataFrame
    cur_results = []
    for cur_fault, cur_array in fault_im_dict.items():
        if site not in cur_array.station:
            continue

        cur_df = cur_array.sel(station=site).to_dataframe(name="value").reset_index()
        cur_df["fault"] = cur_fault
        cur_df = cur_df.pivot(
            index=["fault", "realisation"], columns="IM", values="value"
        )

        cur_results.append(cur_df)

    return pd.concat(cur_results, axis=0)


def compute_sim_hazard(
    site_im_df: pd.DataFrame,
    flt_erf_df: pd.DataFrame,
    ims: Sequence[str] = None,
    im_levels: dict[str, np.ndarray[float]] = None,
):
    """
    Computes the simulation-based seismic hazard
    for a single site.

    Parameters
    ----------
    site_im_df: pd.DataFrame
        The IM data for the site.
        Index has to be a MultIndex [fault, realisation]
    flt_erf_df: pd.DataFrame
        The 2010 NSHM fault ERF data
    ims: Sequence of str
        The IMs for which to compute the hazard
    im_levels: dict
        The IM levels for each IM

    Returns
    -------
    hazard_results: dict
        The hazard curve for each IM
    """
    rec_prob = 1 / flt_erf_df["recur_int_median"]

    ims = site_im_df.columns if ims is None else ims
    if im_levels is not None:
        if any([True for cur_im in ims if cur_im not in im_levels]):
            raise ValueError(f"Not all IMs found in im_levels!")

    hazard_results = {}
    for cur_im in ims:
        cur_im_levels = utils.get_im_levels(cur_im)
        if im_levels is not None:
            cur_im_levels = im_levels.get(cur_im)

        cur_gm_prob_excd = hazard.non_parametric_gm_excd_prob(
            cur_im_levels, site_im_df[cur_im]
        )
        hazard_results[cur_im] = hazard.hazard_curve(cur_gm_prob_excd, rec_prob)

    return hazard_results
