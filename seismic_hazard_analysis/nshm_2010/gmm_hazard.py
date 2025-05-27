from collections.abc import Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm

import oq_wrapper as oqw
from qcore import nhm
from source_modelling import sources

from .. import hazard, utils
from . import utils as nshm_utils

TECTONIC_TYPE_MAPPING = {
    "ACTIVE_SHALLOW": oqw.constants.TectType.ACTIVE_SHALLOW,
    "VOLCANIC": oqw.constants.TectType.ACTIVE_SHALLOW,
    "SUBDUCTION_INTERFACE": oqw.constants.TectType.SUBDUCTION_INTERFACE,
    "SUBDUCTION_SLAB": oqw.constants.TectType.SUBDUCTION_SLAB,
}


def get_flt_rupture_df(
    faults: dict[str, sources.Fault],
    flt_erf_df: pd.DataFrame,
    site_nztm: np.ndarray[float],
    site_properties: dict[str, float],
):
    """
    Creates the rupture dataframe for the given
    faults, ready for GM parameter computing
    using the OpenQuake GMM wrapper.

    Parameters
    ----------
    faults: dict
        The fault objects
    flt_erf_df: pd.DataFrame
        The fault ERF dataframe
    site_nztm: np.ndarray[float]
        The site coordinates in NZTM (X, Y, Depth)
    site_properties: dict
        Dictionary containing site properties:
        - vs30: float, required
            The average shear-wave velocity in the upper 30 meters of the site.
        - vs30measured: bool, required
            Whether the Vs30 value is measured or not.
        - z1p0: float, required
            Depth to the 1.0 km/s shear-wave velocity horizon in km.
        - z2p5: float
            Depth to the 2.5 km/s shear-wave velocity horizon in km.
            Only required for some GMMs
        - backarc: bool
            Whether the site is in the backarc region.
            Only required for some GMMs

    Returns
    -------
    rupture_df: pd.DataFrame
        The rupture dataframe for the given faults
    """
    # Compute source to site distances
    rupture_df = nshm_utils.run_site_to_source_dist(
        faults, site_nztm)

    # Add fault details to the rupture_df
    rupture_df.index = list(faults.keys())
    rupture_df[["mag", "rake", "ztor", "tectonic_type", "dip", "zbot"]] = (
        flt_erf_df.loc[
            rupture_df.index, ["mw", "rake", "dtop", "tectonic_type", "dip", "dbottom"]
        ]
    )   
    # Use hypocentre depth at 1/2
    rupture_df["hypo_depth"] = (rupture_df["zbot"] + rupture_df["ztor"]) / 2

    rupture_df["vs30"] = site_properties["vs30"]
    rupture_df["z1pt0"] = site_properties["z1p0"]
    rupture_df["vs30measured"] = site_properties["vs30measured"]
    if "z2p5" in site_properties.keys():
        rupture_df["z2pt5"] = site_properties["z2p5"]
    if "backarc" in site_properties.keys():
        rupture_df["backarc"] = site_properties["backarc"]

    return rupture_df


def get_emp_gm_params(
    rupture_df: pd.DataFrame,
    gmm_mapping: dict[oqw.constants.TectType, oqw.constants.GMM],
    ims: list[str],
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
    ims: list
        The IMs to compute the GM parameters for

    Returns
    -------
    gm_params_df: pd.DataFrame
        The GM parameters for the given ruptures
    """
    pSA_periods = [
        float(im.rsplit("_", maxsplit=1)[1]) for im in ims if im.startswith("pSA")
    ]

    non_pSA_ims = [cur_im for cur_im in ims if not cur_im.startswith("pSA")]
    if len(non_pSA_ims) > 0:
        raise ValueError(
            f"The IMs {non_pSA_ims} are not supported. Only pSA is currently supported!"
        )

    gm_params_df = []
    for cur_tect_type_str in rupture_df["tectonic_type"].unique():
        cur_tect_type = TECTONIC_TYPE_MAPPING[cur_tect_type_str]
        cur_gmm = gmm_mapping[cur_tect_type]

        cur_rupture_df = rupture_df.loc[
            rupture_df["tectonic_type"] == cur_tect_type_str
        ]
        if isinstance(cur_gmm, oqw.constants.GMM):
            cur_result = oqw.run_gmm(
                cur_gmm,
                cur_tect_type,
                cur_rupture_df,
                "pSA",
                periods=pSA_periods,
            )
        else:
            cur_result = oqw.run_gmm_lt(
                cur_gmm,
                cur_tect_type,
                cur_rupture_df,
                "pSA",
                periods=pSA_periods,
            )
        cur_result.index = cur_rupture_df.index
        gm_params_df.append(cur_result)

    gm_params_df = pd.concat(gm_params_df, axis=0)
    return gm_params_df


def get_oq_ds_rupture_df(
    rupture_df: pd.DataFrame,
    site_nztm: np.ndarray[float],
    site_properties: dict[str, float],
):
    """
    Creates the rupture dataframe for
    distributed seismicity for GM parameter
    computing using the OpenQuake GMM wrapper.

    Parameters
    ----------
    source: pd.DataFrame
        The source dataframe for DS
    site_nztm: np.ndarray[float]
        The site coordinates in NZTM (X, Y, Depth)
    site_properties: dict
        Dictionary containing site properties:
        - vs30: float, required
            The average shear-wave velocity in the upper 30 meters of the site.
        - vs30measured: bool, required
            Whether the Vs30 value is measured or not.
        - z1p0: float, required
            Depth to the 1.0 km/s shear-wave velocity horizon in km.
        - z2p5: float
            Depth to the 2.5 km/s shear-wave velocity horizon in km.
            Only required for some GMMs
        - backarc: bool
            Whether the site is in the backarc region.
            Only required for some GMMs

    Returns
    -------
    rupture_df: pd.DataFrame
        The rupture dataframe for DS
    """
    # Compute site distances
    rupture_df["rjb"] = (
        np.sqrt(
            (site_nztm[0] - rupture_df["nztm_x"]) ** 2
            + (site_nztm[1] - rupture_df["nztm_y"]) ** 2
        )
        / 1000
    )
    rupture_df["rrup"] = (
        np.sqrt(
            (site_nztm[0] - rupture_df["nztm_x"]) ** 2
            + (site_nztm[1] - rupture_df["nztm_y"]) ** 2
            + (rupture_df["depth"] * 1000) ** 2
        )
        / 1000
    )
    # Use Rjb for rx and ry, in the past we have used zero for this.
    # Using Rjb gives the same result (when using Br13 and ZA06)
    # as using zero, and makes more sense.
    rupture_df["rx"] = rupture_df["rjb"]
    rupture_df["ry"] = rupture_df["rjb"]

    # rupture_df["hypo_depth"] = rupture_df["depth"]
    rupture_df = rupture_df.rename(
        columns={
            "dtop": "ztor",
            "dbot": "zbot",
            "depth": "hypo_depth",
        }
    )

    rupture_df["vs30"] = site_properties["vs30"]
    rupture_df["z1pt0"] = site_properties["z1p0"]
    rupture_df["vs30measured"] = site_properties["vs30measured"]
    if "z2p5" in site_properties.keys():
        rupture_df["z2pt5"] = site_properties["z2p5"]
    if "backarc" in site_properties.keys():
        rupture_df["backarc"] = site_properties["backarc"]

    return rupture_df


def compute_gmm_hazard(
    gm_params_df: pd.DataFrame,
    rec_prob: pd.Series,
    ims: Sequence[str],
    im_levels: dict[str, np.ndarray[float]] = None,
):
    """
    Computes the hazard curves for the given
    site and GM parameters for each IM.

    Parameters
    ----------
    gm_params_df: pd.DataFrame
        The GM parameters for the ruptures
    rec_prob: pd.Series
        The recurrence probabilities of the ruptures
    ims: Sequence[str]
        The IMs to compute the hazard curves for
    im_levels: dict, optional
        The IM levels to compute the hazard curves for.

    Returns
    -------
    hazard_results: dict
        The hazard curves for each IM
    """
    if im_levels is not None:
        if any([True for cur_im in ims if cur_im not in im_levels]):
            raise ValueError("Not all IMs found in im_levels!")

    hazard_results = {}
    for cur_im in ims:
        cur_im_levels = utils.get_im_levels(cur_im)
        if im_levels is not None:
            cur_im_levels = im_levels.get(cur_im)

        gm_prob_df = hazard.parametric_gm_excd_prob(
            cur_im_levels,
            gm_params_df,
            mean_col=f"{cur_im}_mean",
            std_col=f"{cur_im}_std_Total",
        )
        hazard_results[cur_im] = hazard.hazard_curve(gm_prob_df, rec_prob)

    return hazard_results


def compute_gmm_ds_hazard(
    source_df: pd.DataFrame,
    ds_erf_df: pd.DataFrame,
    site_nztm: np.ndarray[float],
    site_properties: dict[str, float],
    gmm_mapping: dict[oqw.constants.TectType, oqw.constants.GMM],
    ims: Sequence[str],
):
    """
    Compute the seismic hazard for a given site using
    the provided rupture and empirical ground motion models (GMM).

    Parameters
    ----------
    source_df : pd.DataFrame
        DataFrame containing DS source information.
    ds_erf_df : pd.DataFrame
        DataFrame containing annual recurrence probabilities.
    site_nztm : np.ndarray[float]
        Array containing the site coordinates in NZTM projection.
        [X, Y, Depth]
    site_properties: dict
        Dictionary containing site properties:
        - vs30: float, required
            The average shear-wave velocity in the upper 30 meters of the site.
        - vs30measured: bool, required
            Whether the Vs30 value is measured or not.
        - z1p0: float, required
            Depth to the 1.0 km/s shear-wave velocity horizon in km.
        - z2p5: float
            Depth to the 2.5 km/s shear-wave velocity horizon in km.
            Only required for some GMMs
        - backarc: bool
            Whether the site is in the backarc region.
            Only required for some GMMs
    gmm_mapping : dict[TectType, GMM]
        Dictionary mapping tectonic types to their
        corresponding ground motion models (GMM).
    ims : Sequence[str]
        Sequence of intensity measures to be considered.

    Returns
    -------
    ds_hazard : pd.DataFrame
        DataFrame containing the computed seismic hazard for the given site.
    """
    oq_rupture_df = get_oq_ds_rupture_df(source_df, site_nztm, site_properties)
    ds_gm_params_df = get_emp_gm_params(oq_rupture_df, gmm_mapping, ims).sort_index()
    ds_hazard = compute_gmm_hazard(ds_gm_params_df, ds_erf_df.annual_rec_prob, ims)

    return ds_hazard


def compute_gmm_flt_hazard(
    site_nztm: np.ndarray[float],
    site_properties: dict[str, float],
    flt_erf_df: pd.DataFrame,
    gmm_mapping: dict[oqw.constants.TectType, oqw.constants.GMM],
    ims: Sequence[str],
    faults: dict[str, sources.Fault] = None,
    flt_definitions: dict[str, nhm.NHMFault] = None,
):
    """
    Compute the fault hazard for a given site.

    Parameters:
    -----------
    site_nztm : np.ndarray[float]
        The NZTM coordinates of the site.
    site_properties: dict
        Dictionary containing site properties:
        - vs30: float, required
            The average shear-wave velocity in the upper 30 meters of the site.
        - vs30measured: bool, required
            Whether the Vs30 value is measured or not.
        - z1p0: float, required
            Depth to the 1.0 km/s shear-wave velocity horizon in km.
        - z2p5: float
            Depth to the 2.5 km/s shear-wave velocity horizon in km.
            Only required for some GMMs
        - backarc: bool
            Whether the site is in the backarc region.
            Only required for some GMMs
    flt_erf_df : pd.DataFrame
        DataFrame containing fault ERF data.
    gmm_mapping : dict[TectType, GMM]
        Dictionary mapping tectonic types to GMM.
    ims : Sequence[str]
        List of intensity measures.
    faults : dict[str, sources.Fault], optional
        Dictionary of fault objects. 
        If not provided, it will be created from flt_erf.
    flt_erf : dict[str, nhm.NHMFault], optional
        Dictionary containing fault ERF data. 
        If not provided, faults must be provided.

    Returns:
    --------
    flt_hazard : pd.DataFrame
        DataFrame containing the computed fault hazard.
    
    Raises:
    -------
    ValueError
        If neither faults nor flt_erf are provided.
    """
    if faults is None and flt_definitions is None:
        raise ValueError("Faults or fault ERF must be provided!")

    # Create the fault objects
    if faults is None:
        faults = {
            cur_name: nshm_utils.get_fault_objects(cur_fault)
            for cur_name, cur_fault in flt_definitions.items()
        }

    # Get GM parameters
    flt_rupture_df = get_flt_rupture_df(
        faults, flt_erf_df, site_nztm, site_properties
    )

    # Compute hazard
    flt_gm_params_df = get_emp_gm_params(flt_rupture_df, gmm_mapping, ims)
    flt_hazard = compute_gmm_hazard(
        flt_gm_params_df, 1 / flt_erf_df["recur_int_median"], ims
    )
    return flt_hazard
