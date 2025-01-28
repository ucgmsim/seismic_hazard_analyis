"""Module for computing the seismic hazard"""

from typing import Union

import numpy as np
import pandas as pd
import scipy as sp


def non_parametric_gm_excd_prob(im_levels: np.ndarray[float], im_values: pd.Series) -> pd.DataFrame:
    """
    Computes GM  exceedance probability for each IM level over
    all ruptures based on the non-parametric GM predictions (e.g. simulations)

    Parameters
    ----------
    im_levels: np.ndarray[float]
        The IM levels for which to calculate the ground motion exceedance probability
    im_values: pd.Series
        The IM values for each rupture and for each "realisation"
         in each rupture
        format: index = MultiIndex[rupture_name, realisation_name], values = IM value

    Returns
    -------
    pd.DataFrame
        The exceedance probability for each rupture
    """
    # Count the number of realisations per rupture
    rupture_count = im_values.groupby(level=0).count().sort_index()

    # Count the number of realisation with IM values greater than the specified IM level
    greater_comp = pd.DataFrame(
        data=im_values.values[:, None] > im_levels[None, :],
        index=im_values.index,
        columns=im_levels,
    )
    greater_count = greater_comp.groupby(level=0).agg(np.count_nonzero).sort_index()

    return greater_count.div(rupture_count, axis=0)


def parametric_gm_excd_prob(
    im_levels: Union[float, np.ndarray],
    im_params: pd.DataFrame,
    mean_col: str = "mu",
    std_col: str="sigma",
):
    """
    Computes the GM exceedance probability for each IM level over all
    ruptures based on the parametric GM predictions (e.g. empirical GMM)

    Parameters
    ----------
    im_levels: float or array
        The IM level(s) for which to calculate the ground motion
        exceedance probability
    im_params: pd.DataFrame
        The IM distribution parameters for each rupture
        format: index = rupture_name
    mean_col: str, optional
        Name of the column containing the mean lnIM values
    std_col: str, optional
        Name of the column containing the standard deviation of lnIM values

    Returns
    -------
    pd.DataFrame
        The exceedance probability for each rupture at each IM level
        shape: [n_ruptures, n_im_levels]
    """
    im_levels = np.asarray(im_levels).reshape(1, -1)

    results = sp.stats.norm.sf(
        np.log(im_levels),
        im_params[mean_col].values.reshape(-1, 1),
        im_params[std_col].values.reshape(-1, 1),
    )
    return pd.DataFrame(
        index=im_params.index.values, data=results, columns=im_levels.reshape(-1)
    )


def hazard_single(gm_prob: pd.Series, rec_prob: pd.Series):
    """
    Calculates the exceedance probability given the specified
    ground motion exceedance probabilities and rupture recurrence rates

    Note: All ruptures specified in gm_prob have to exist in rec_prob

    Parameters
    ----------
    gm_prob: pd.Series
        The ground motion probabilities
        format: index = rupture_name, values = probability
    rec_prob: pd.Series
        The recurrence probabilities of the ruptures
        format: index = rupture_name, values = probability

    Returns
    -------
    float
        The exceedance probability
    """
    ruptures = gm_prob.index.values
    return np.sum(gm_prob[ruptures] * rec_prob[ruptures])


def hazard_curve(gm_prob_df: pd.DataFrame, rec_prob: pd.Series):
    """
    Calculates the exceedance probabilities for the
    specified IM values (via the gm_prob_df)

    Note: All ruptures specified in gm_prob_df have to exist
    in rec_prob

    Parameters
    ----------
    gm_prob_df: pd.DataFrame
        The ground motion probabilities for every rupture
        for every IM level.
        format: index = rupture_name, columns = IM_levels
    rec_prob: pd.Series
        The recurrence probabilities of the ruptures
        format: index = rupture_name, values = probability

    Returns
    -------
    pd.Series
        The exceedance probabilities for the different IM levels
        format: index = IM_levels, values = exceedance probability
    """
    data = np.sum(
        gm_prob_df.values * rec_prob[gm_prob_df.index.values].values.reshape(-1, 1),
        axis=0,
    )
    return pd.Series(index=gm_prob_df.columns.values, data=data)
