from collections.abc import Sequence

import numpy as np
import scipy as sp


def query_non_parametric_cdf_invs(
    y: np.ndarray, cdf_x: np.ndarray, cdf_y: np.ndarray
) -> np.ndarray:
    """
    Retrieve the x-values for the specified y-values given the
    non-parametric cdf function

    Note: Since this is for a discrete CDF,
    the inversion function returns the x value
    corresponding to F(x) >= y

    Parameters
    ----------
    y: array of floats
    cdf_x: array of floats
    cdf_y: array of floats
        The x and y values of the non-parametric cdf

    Returns
    -------
    y: array of floats
        The corresponding y-values
    """
    assert cdf_y[0] >= 0.0 and np.isclose(cdf_y[-1], 1.0, rtol=1e-2)
    assert np.all((y > 0.0) & (y < 1.0))

    mask, _ = cdf_y >= y[:, np.newaxis], []
    return np.asarray(
        [cdf_x[np.min(np.flatnonzero(mask[ix, :]))] for ix in range(y.size)]
    )


def query_non_parametric_multi_cdf_invs(
    y: Sequence, cdf_x: np.ndarray, cdf_y: np.ndarray
) -> list:
    """
    Retrieve the x-values for the specified y-values given a
    multidimensional array of non-parametric cdf along each row

    Note: Since this is for a discrete CDF,
    the inversion function returns the x value
    corresponding to F(x) >= y

    Parameters
    ----------
    y: Sequence of floats
        Quantiles to query
    cdf_x: 2d array of floats
        The x values of the non-parametric cdf
        With each row representing one CDF
    cdf_y: 2d array of floats
        The y values of the non-parametric cdf
        With each row representing one CDF

    Returns
    -------
    y: List
        The corresponding y-values
    """
    x_values = []
    for cur_y in y:
        diff = cdf_y - cur_y
        x_values.append(
            [
                cdf_x[ix, :][np.min(np.flatnonzero(diff[ix, :] > 0))]
                for ix in range(len(cdf_x))
            ]
        )
    return x_values


def query_non_parametric_cdf(
    x: np.ndarray, cdf_x: np.ndarray, cdf_y: np.ndarray
) -> np.ndarray:
    """
    Retrieve the y-values for the specified x-values given the
    non-parametric cdf function

    Parameters
    ----------
    x: array of floats
    cdf_x: array of floats
    cdf_y: array of floats
        The x and y values of the non-parametric cdf

    Returns
    -------
    y: array of floats
        The corresponding y-values
    """
    assert cdf_y[0] >= 0.0 and np.isclose(
        cdf_y[-1], 1.0, rtol=1e-2
    ), f"cdf_y[0] = {cdf_y[0]}, cdf_y[-1] = {cdf_y[-1]}"

    mask, y = cdf_x <= x[:, np.newaxis], []
    for ix in range(x.size):
        cur_ind = np.flatnonzero(mask[ix, :])
        y.append(cdf_y[np.max(cur_ind)] if cur_ind.size > 0 else 0.0)

    return np.asarray(y)


def nearest_pd(A: np.ndarray[float]):
    """Find the nearest positive-definite matrix to input

    From stackoverflow:
    https://stackoverflow.com/questions/43238173/python-convert-matrix-to-positive-semi-definite

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """
    B = (A + A.T) / 2
    _, s, V = sp.linalg.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if is_pd(A3):
        return A3

    spacing = np.spacing(sp.linalg.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])  # noqa: E741
    k = 1
    while not is_pd(A3):
        mineig = np.min(np.real(sp.linalg.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3


def is_pd(B: np.ndarray[float]):
    """
    Returns true when input is positive-definite, via Cholesky

    Parameters
    ----------
    B: np.ndarray[float]
        The input matrix

    Returns
    -------
    bool
        True if positive-definite, False otherwise
    """
    try:
        _ = sp.linalg.cholesky(B, lower=True)
        return True
    except sp.linalg.LinAlgError:
        return False


def get_min_max_levels_for_im(im: str):
    """Get minimum and maximum for the given im. Values for velocity are
    given on cm/s, acceleration on cm/s^2 and Ds on s
    """
    match im.upper():
        case _ if im.startswith("pSA"):
            period = float(im.rsplit("_", 1)[-1])
            periods = np.array([0.5, 1.0, 3.0, 5.0, 10.0])
            bounds = [
                (0.005, 10.0),
                (0.005, 7.5),
                (0.0005, 5.0),
                (0.0005, 4.0),
                (0.0005, 3.0),
            ]
            idx = np.searchsorted(periods, period)
            return bounds[idx]
        case "PGA":
            return 0.0001, 10.0
        case "PGV":
            return 1.0, 400.0
        case "CAV":
            return 0.0001 * 980, 20.0 * 980.0
        case "AI":
            return 0.01, 1000.0
        case "DS575" | "DS595":
            return 1.0, 400.0
        case "MMI":
            return 1.0, 12.0
        case _:
            raise ValueError("Invalid IM")


def get_im_levels(im: str, n_values: int = 200):
    """
    Create an range of values for a given
    IM according to their min, max
    as defined by get_min_max_values

    Parameters
    ----------
    im: IM
        The IM Object to get im values for
    n_values: int

    Returns
    -------
    Array of IM values
    """
    start, end = get_min_max_levels_for_im(im)
    im_values = np.logspace(
        start=np.log(start), stop=np.log(end), num=n_values, base=np.e
    )
    return im_values



def rp_to_prob(rp: float, t: float = 1.0):
    """
    Converts return period to exceedance probability
    Based on Poisson distribution

    Parameters
    ----------
    rp: float
        Return period
    t: float
        Time period of interest

    Returns
    -------
    Exceedance probability
    """
    return 1 - np.exp(-t / rp)


def prob_to_rp(prob: float, t: float = 1.0):
    """
    Converts probability of exceedance to return period
    Based on Poisson distribution

    Parameters
    ----------
    prob: float
        Exceedance probability
    t: float
        Time period of interest

    Returns
    -------
    Return Period
    """
    return -t / np.log(1 - prob)


def get_im_file_format(im: str) -> str:
    """
    Get the file format for the given IM.
    """
    if im.startswith("pSA"):
        im = im.replace(".", "p")
    return im
    
def reverse_im_file_format(im: str) -> str:
    """
    Reverse the file format for the given IM.
    """
    if im.startswith("pSA"):
        split_im = im.split("_", 1)
        return f"{split_im[0]}_{split_im[1].replace('p', '.')}"
    
    return im

def get_non_uniform_grid_site_level(site_ids: np.ndarray) -> int:
    """
    For the non-uniform grid, get the grid level of a site based on its ID.
    For more details see:
    https://ucdigitalsms.atlassian.net/wiki/spaces/QuakeCore/pages/3291694883/Non-uniform+grid+18p6
    https://ucdigitalsms.atlassian.net/wiki/spaces/QuakeCore/pages/3291700313/Non+Uniform+Grid+20.3

    Assumes any non-numeric site ID is a real site and returns -1 for those.

    Parameters
    ----------
    site_id : np.ndarray
        The site IDs.

    Returns
    -------
    np.ndarray
        The grid level of the sites.
    """
    site_grid_level = np.full(site_ids.shape, -1, dtype=int)

    real_sites_mask = ~np.char.isnumeric(site_ids)
    site_grid_level[real_sites_mask] = -1

    level_0_mask = np.char.startswith(site_ids, "0")
    site_grid_level[level_0_mask] = 0

    level_1_mask = np.char.startswith(site_ids, "1")
    site_grid_level[level_1_mask] = 1

    level_2_mask = np.char.startswith(site_ids, "2")
    site_grid_level[level_2_mask] = 2

    level_3_mask = np.char.startswith(site_ids, "3")
    site_grid_level[level_3_mask] = 3

    level_4_mask = np.char.startswith(site_ids, "4")
    site_grid_level[level_4_mask] = 4

    assert (
        np.count_nonzero(
            real_sites_mask
            | level_0_mask
            | level_1_mask
            | level_2_mask
            | level_3_mask
            | level_4_mask
        )
        == site_ids.size
    )

    return site_grid_level
