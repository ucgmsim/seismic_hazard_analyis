from typing import List, Sequence

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

    mask, x = cdf_y >= y[:, np.newaxis], []
    return np.asarray(
        [cdf_x[np.min(np.flatnonzero(mask[ix, :]))] for ix in range(y.size)]
    )


def query_non_parametric_multi_cdf_invs(
    y: Sequence, cdf_x: np.ndarray, cdf_y: np.ndarray
) -> List:
    """
    Retrieve the x-values for the specified y-values given a
    multidimensional array of non-parametric cdf along each row

    Note: Since this is for a discrete CDF,
    the inversion function returns the x value
    corresponding to F(x) >= y

    Parameters
    ----------
    y: Sequence of floats
    cdf_x: 2d array of floats
    cdf_y: 2d array of floats
        The x and y values of the non-parametric cdf
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


def nearest_pd(A):
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
    I = np.eye(A.shape[0])
    k = 1
    while not is_pd(A3):
        mineig = np.min(np.real(sp.linalg.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3


def is_pd(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = sp.linalg.cholesky(B, lower=True)
        return True
    except sp.linalg.LinAlgError:
        return False


def get_min_max_levels_for_im(im: str):
    """Get minimum and maximum for the given im. Values for velocity are
    given on cm/s, acceleration on cm/s^2 and Ds on s
    """
    if im.startswith("pSA"):
        period = float(im.rsplit("_", 1)[-1])
        if period <= 0.5:
            return 0.005, 10.0
        elif 0.5 < period <= 1.0:
            return 0.005, 7.5
        elif 1.0 < period <= 3.0:
            return 0.0005, 5.0
        elif 3.0 < period <= 5.0:
            return 0.0005, 4.0
        elif 5.0 < period <= 10.0:
            return 0.0005, 3.0
    if im.upper() == "PGA":
        return 0.0001, 10.0
    elif im.upper() == "PGV":
        return 1.0, 400.0
    elif im.upper() == "CAV":
        return 0.0001 * 980, 20.0 * 980.0
    elif im.upper() == "AI":
        return 0.01, 1000.0
    elif im.upper() == "DS575" or im.upper() == "DS595":
        return 1.0, 400.0
    elif im.upper() == "MMI":
        return 1.0, 12.0
    else:
        raise ValueError("Invalid IM")


def get_im_levels(im: str, n_values: int = 100):
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
