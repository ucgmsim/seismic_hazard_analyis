import numpy as np
import pandas as pd

import seismic_hazard_analysis as sha


def test_single_rupture():
    """Based on the 'One Rupture Scenario' example,
    using a single IM level of 0.2
    """
    gm_prob = pd.Series(index=["rupture_1"], data=[0.378])
    rec_prob = pd.Series(index=["rupture_1"], data=[0.01])

    result = sha.hazard.hazard_single(gm_prob, rec_prob)

    assert result == 0.00378


def test_single_rupture_multi_IM():
    """Based on the 'One Rupture Scenario' example,
    with the IM levels of 0.2 and 0.5
    """
    im_levels = [0.2, 0.5]
    gm_prob_df = pd.DataFrame(
        index=["rupture_1"],
        columns=im_levels,
        data=np.asarray([0.378, 0.0191]).reshape(1, -1),
    )
    rec_prob = pd.Series(index=["rupture_1"], data=0.01)

    result = sha.hazard.hazard_curve(gm_prob_df, rec_prob)

    assert np.all(np.isclose(result.values, np.asarray([0.00378, 0.000191])))


def test_two_rupture():
    """Based on the 'Two Rupture Scenarios' example using
    two ruptures at an IM level of 0.2
    """
    gm_prob = pd.Series(index=["rupture_1", "rupture_2"], data=[0.378, 0.940])
    rec_prob = pd.Series(index=["rupture_1", "rupture_2"], data=[0.01, 0.002])

    result = sha.hazard.hazard_single(gm_prob, rec_prob)

    assert result == 0.00566


def test_two_rupture_multi_IM():
    """Based on the 'Two Rupture Scenarios' example using
    two ruptures at an IM level of 0.2 and 0.5
    """
    im_levels = [0.2, 0.5]
    gm_prob_df = pd.DataFrame(
        index=["rupture_1", "rupture_2"],
        columns=im_levels,
        data=[[0.378, 0.0191], [0.940, 0.419]],
    )
    rec_prob = pd.Series(index=["rupture_1", "rupture_2"], data=[0.01, 0.002])

    result = sha.hazard.hazard_curve(gm_prob_df, rec_prob)

    assert np.all(np.isclose(result.values, np.asarray([0.00566, 0.001029])))


def test_non_parametric_gm_prob():
    im_level = np.array([2])
    index_tuples = [
        ("rupture_1", "rel_1"),
        ("rupture_1", "rel_2"),
        ("rupture_2", "rel_1"),
        ("rupture_2", "rel_2"),
        ("rupture_2", "rel_3"),
    ]
    index = pd.MultiIndex.from_tuples(index_tuples)
    values = [1, 3, 1, 3, 4]

    im_values = pd.Series(index=index, data=values)
    result = sha.hazard.non_parametric_gm_excd_prob(im_level, im_values)

    assert np.isclose(result.loc["rupture_1", 2], 0.5)
    assert np.isclose(result.loc["rupture_2", 2], 2 / 3)


def test_parametric_gm_prob():
    im_level = np.exp(1)
    im_params = pd.DataFrame(
        index=["rupture_1", "rupture_2"],
        columns=["mu", "sigma"],
        data=[[1, 1], [0.25, 0.1]],
    )

    results = sha.hazard.parametric_gm_excd_prob(im_level, im_params)

    # The IM values of the first rupture have a mean and standard deviation of 1, 1
    # which means that for a IM level of exp(1) (since lognormal distribution is used)
    # should always give an exceedance probability of 0.5 (since it
    # corresponds to the mean)
    assert float(results.loc["rupture_1"]) == 0.5

    # The IM values of the second rupture have a mean of 0.25 and standard deviation
    # of 0.1, which means that the exceedance probability for np.exp(1) should always
    # be pretty much zero
    assert np.isclose(float(results.loc["rupture_2"]), 0)
