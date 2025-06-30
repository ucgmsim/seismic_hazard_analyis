import pandas as pd
import numpy as np

from . import utils


def compute_uhs(
    mean_hcurves: dict[str, pd.Series],
    excd_rates: list[float],
    rps: list[float] | None = None,
):
    """
    Computes the Uniform Hazard Spectrum (UHS) from the given mean hazard curves.

    Parameters
    ----------
    mean_hcurves: dict[str, pd.Series]
        A dictionary where keys are IM names, such as "pSA_0.1", and values are
        pandas Series representing the mean hazard curves for each IM level.
    excd_rates: list[float]
        A list of exceedance rates for which to compute the UHS.
    rps: list[float] | None, optional
        Return periods corresponding to the exceedance rates.
        If provided, these will be used as columns in the output DataFrame.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the UHS values, indexed by pSA period.
    """
    pSA_keys = np.array([cur_im for cur_im in mean_hcurves.keys() if "pSA" in cur_im])
    pSA_periods = np.array([utils.get_pSA_period(cur_im) for cur_im in pSA_keys])

    # Sort the pSA periods
    sort_ind = np.argsort(pSA_periods)
    pSA_keys = pSA_keys[sort_ind]
    pSA_periods = pSA_periods[sort_ind]

    results = {}
    for cur_im, cur_period in zip(pSA_keys, pSA_periods):
        cur_result = utils.exceedance_to_im(
            np.asarray(excd_rates),
            mean_hcurves[cur_im].index.values,
            mean_hcurves[cur_im].values,
        )
        results[cur_period] = cur_result

    uhs_df = pd.DataFrame.from_dict(
        results, orient="index", columns=rps if rps else excd_rates
    )
    uhs_df.index.name = "period"

    return uhs_df
