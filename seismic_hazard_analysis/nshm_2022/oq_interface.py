import multiprocessing as mp
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from openquake.calculators.extract import Extractor
from openquake.commonlib.datastore import read
from tqdm import tqdm

from .. import utils


def get_hazard_curves_stats(
    calc_id: int,
    n_procs: int,
    output_dir: Path,
    quantiles: Sequence[float] | None = None,
):
    """
    Extract hazard curve realisation from the OQ database 
    and compute mean and quantiles for each IM and IM level.
    
    Uses multiprocessing to speed up the extraction process.

    Parameters
    ----------
    calc_id: int
        OpenQuake calculation ID to extract hazard curves from.
    n_procs: int
        Number of processes to use for parallel extraction.
    output_dir: Path
        Directory to save the extracted hazard curve statistics.
    quantiles: Sequence[float] | None, optional
        List of quantiles to compute for each IM level.
        If None, defaults to [0.05, 0.1, 0.16, 0.5, 0.84, 0.9, 0.95].   
    """
    if quantiles is None:
        quantiles = [0.05, 0.1, 0.16, 0.5, 0.84, 0.9, 0.95]

    # Get the IM levels
    with read(calc_id) as ds:
        im_levels = ds["oqparam"].hazard_imtls
    ims = list(im_levels.keys())
    n_ims = len(ims)
    n_im_levels = len(im_levels[ims[0]])

    # Get the logic tree weights
    with Extractor(calc_id) as ex:
        lt = ex.get("full_lt")
        weights_df = pd.Series(
            {cur_rel.ordinal: cur_rel.weight[0] for cur_rel in lt.get_realizations()}
        )
    assert np.isclose(weights_df.sum(), 1)
    assert np.all(np.sort(weights_df.index.values) == weights_df.index.values)
    n_rels = weights_df.shape[0]

    # Split the realizations into batches
    # for loading the hazard curve realisations
    # as loading all of them at once is extremely slow
    n_batches = int(np.ceil(n_rels / 16))
    batches = np.array_split(weights_df.index.values, n_batches)

    start = time.time()
    if n_procs == 1:
        hcurves = []
        for batch in tqdm(batches):
            hcurves.append(
                _get_hazard_curves(calc_id, batch, n_ims, n_im_levels)
            )
    else:
        with mp.Pool(n_procs) as pool:
            hcurves = pool.starmap(
                _get_hazard_curves,
                [(calc_id, batch, n_ims, n_im_levels) for batch in batches],
            )
    print(f"Took: {time.time() - start} to get {weights_df.shape[0]} hazard curves")
    hcurves = np.concatenate(hcurves, axis=0)

    # Compute the statistics
    
    weights = weights_df.values
    for i, cur_im in tqdm(enumerate(ims), total=n_ims, desc="Processing IMs"):
        cur_data = hcurves[:, i, :]

        # Calculate the mean hazard
        cur_mean = np.average(cur_data, weights=weights, axis=0)

        # Calculate the quantiles
        sort_ind = np.argsort(cur_data, axis=0)
        cur_data = np.take_along_axis(cur_data, sort_ind, axis=0)
        cur_weights = np.tile(weights[:, None], (1, n_im_levels))
        cur_weights = np.take_along_axis(cur_weights, sort_ind, axis=0)

        cur_quantile_results = utils.query_non_parametric_multi_cdf_invs(
            quantiles, cur_data.T, np.cumsum(cur_weights.T, axis=1)
        )

        # Combine and write
        cur_df = pd.DataFrame(
            data=np.stack([cur_mean] + list(cur_quantile_results), axis=1),
            columns=["mean"] + [f"quantile_{cur_qt}" for cur_qt in quantiles],
            index=im_levels[cur_im],
        )

        cur_df.to_csv(
            output_dir / f"{utils.get_im_file_format(get_im_name(cur_im))}_statistics.csv",
            index_label="im_level",
        )

def _get_hazard_curves(
    calc_id: int,
    rel_ordinals: np.ndarray[int],
    n_ims: int,
    n_im_levels: int,
) -> np.ndarray:
    """Helper function to get hazard curves for a batch of realizations."""
    with Extractor(calc_id) as ex:
        hcurves = np.full(
            (rel_ordinals.size, n_ims, n_im_levels), np.nan, dtype=np.float32
        )
        for i, cur_rel_ordinal in enumerate(rel_ordinals):
            hcurves[i, :, :] = ex.get(f"hcurves?kind=rlz-{cur_rel_ordinal}&site_id=0")[
                f"rlz-{cur_rel_ordinal:03}"
            ][0, :, :]

    return hcurves


def get_im_name(oq_im: str) -> str:
    """
    Get the name of the IM from the OpenQuake IM name.

    Parameters
    ----------
    im: str
        The OpenQuake IM name.

    Returns
    -------
    str
        The name of the IM.
    """
    if oq_im.startswith("SA"):
        return oq_im.replace("SA", "pSA").replace("(", "_").replace(")", "")

    return oq_im


