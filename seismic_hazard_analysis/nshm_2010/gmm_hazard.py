from typing import Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm


from source_modelling import sources
from empirical.util.classdef import TectType, GMM
from empirical.util.openquake_wrapper_vectorized import oq_run

from .. import site_source
from .. import hazard
from .. import utils

TECTONIC_TYPE_MAPPING = {
    "ACTIVE_SHALLOW": TectType.ACTIVE_SHALLOW,
    "VOLCANIC": TectType.ACTIVE_SHALLOW,
    "SUBDUCTION_INTERFACE": TectType.SUBDUCTION_INTERFACE,
}


def get_flt_rupture_df(
    faults: dict[str, sources.Fault],
    flt_erf_df: pd.DataFrame,
    site_nztm: np.ndarray[float],
    site_vs30: float,
    site_z1p0: float,
):
    fault_id_mapping = {cur_name: i for i, cur_name in enumerate(faults.keys())}

    # Fault Distance calculation
    plane_nztm_coords = []
    scenario_ids = []
    scenario_section_ids = []
    segment_section_ids = []
    for cur_name, cur_fault in tqdm(faults.items()):
        plane_nztm_coords.append(
            np.stack(
                [cur_plane.bounds[:, [1, 0, 2]] for cur_plane in cur_fault.planes],
                axis=2,
            )
        )
        cur_id = fault_id_mapping[cur_name]
        scenario_ids.append(cur_id)
        # Each scenario only consists of a single fault/section
        scenario_section_ids.append(np.asarray([cur_id]))
        segment_section_ids.append(np.ones(len(cur_fault.planes), dtype=int) * cur_id)

    plane_nztm_coords = np.concatenate(plane_nztm_coords, axis=2)
    scenario_ids = np.asarray(scenario_ids)
    segment_section_ids = np.concatenate(segment_section_ids)

    assert plane_nztm_coords.shape[2] == segment_section_ids.size

    # Change the order of the corners
    plane_nztm_coords = plane_nztm_coords[[0, 3, 1, 2], :, :]

    # Compute segment strike
    segment_strike, segment_strike_vec = site_source.compute_segment_strike_nztm(
        plane_nztm_coords
    )

    # Compute rupture scenario distances
    rupture_df = site_source.get_scenario_distances(
        scenario_ids,
        scenario_section_ids,
        plane_nztm_coords,
        segment_section_ids,
        # site_nztm[0, [1, 0, 2]],
        site_nztm,
    )

    # Add fault details to the rupture_df
    rupture_df.index = list(faults.keys())
    rupture_df[["mag", "rake", "ztor", "tectonic_type", "dip", "dbottom"]] = (
        flt_erf_df.loc[
            rupture_df.index, ["mw", "rake", "dtop", "tectonic_type", "dip", "dbottom"]
        ]
    )
    rupture_df["vs30"] = site_vs30
    rupture_df["z1pt0"] = site_z1p0
    rupture_df["vs30measured"] = True

    # Use hypocentre depth at 1/2
    rupture_df["hypo_depth"] = (rupture_df["dbottom"] + rupture_df["ztor"]) / 2

    return rupture_df


def get_emp_gm_params(
    rupture_df: pd.DataFrame, gmm_mapping: dict[TectType, GMM], pSA_periods: list[float]
):
    """
    Computes the GM parameters for the given
    ruptures with the specified GMMs using
    the OpenQuake GMM wrapper.

    Currently only supports pSA.

    Parameters
    ----------
    rupture_df: pd.DataFrame
        Has to be in the correct format for the
        OpenQuake GMM wrapper.
    gmm_mapping: dict
        Specifies the GMM to use for each tectonic type
    pSA_periods:

    Returns
    -------
    gm_params_df: pd.DataFrame
        The GM parameters for the given ruptures
    """
    gm_params_df = []
    for cur_tect_type_str in rupture_df["tectonic_type"].unique():
        cur_tect_type = TECTONIC_TYPE_MAPPING[cur_tect_type_str]
        cur_gmm = gmm_mapping[cur_tect_type]

        cur_rupture_df = rupture_df.loc[
            rupture_df["tectonic_type"] == cur_tect_type_str
        ]
        cur_result = oq_run(
            cur_gmm,
            cur_tect_type,
            cur_rupture_df,
            "SA",
            periods=pSA_periods,
        )
        cur_result.index = cur_rupture_df.index
        gm_params_df.append(cur_result)

    gm_params_df = pd.concat(gm_params_df, axis=0)
    return gm_params_df


def compute_gmm_flt_hazard(
    gm_params_df: pd.DataFrame,
    flt_erf_df: pd.DataFrame,
    ims: Sequence[str],
    im_levels: dict[str, np.ndarray[float]] = None,
):
    rec_prob = 1 / flt_erf_df["recur_int_median"]

    if im_levels is not None:
        if any([True for cur_im in ims if cur_im not in im_levels]):
            raise ValueError(f"Not all IMs found in im_levels!")

    hazard_results = {}
    for cur_im in ims:
        cur_im_levels = utils.get_im_levels(cur_im)
        if im_levels is not None:
            cur_im_levels = im_levels.get(cur_im)

        gm_prob_df = hazard.parametric_gm_excd_prob(
            cur_im_levels, gm_params_df, mean_col=f"{cur_im}_mean", std_col=f"{cur_im}_std_Total"
        )
        hazard_results[cur_im] = hazard.hazard_curve(gm_prob_df, rec_prob)

    return hazard_results