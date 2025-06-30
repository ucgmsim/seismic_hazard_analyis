import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
import typer

import seismic_hazard_analysis as sha

app = typer.Typer()


@app.command("get-hcurve-stats")
def get_hcurve_stats(
    calc_id: int = typer.Argument(..., help="OpenQuake calculation ID"),
    n_procs: int = typer.Argument(
        mp.cpu_count(), help="Number of processes to use for parallel extraction"
    ),
    output_dir: Path = typer.Argument(
        ..., help="Directory to save the extracted hazard curve statistics"
    ),
):
    """
    Extract hazard curve realisation from the OQ database,
    and computes mean and quantiles for each IM and IM level.
    """
    sha.nshm_2022.get_hazard_curves_stats(calc_id, n_procs, output_dir)


@app.command("compute-uhs")
def compute_uhs(
    hcurve_stats_dir: Path = typer.Argument(
        ..., help="Directory containing the hazard curve statistics"
    ),
    output_ffp: Path = typer.Argument(
        ..., help="File path to save the computed UHS DataFrame"
    ),
    excd_rates: list[float] = typer.Option(
        None,
        help="List of exceedance rates to compute UHS for. "
        "One of `excd_rates` or `rps` must be provided.",
    ),
    rps: list[float] = typer.Option(
        None,
        help="List of return periods to compute UHS for. "
        "One of `excd_rates` or `rps` must be provided.",
    ),
):
    """Compute the Uniform Hazard Spectrum (UHS) from the hazard curve statistics."""
    if not (excd_rates or rps):
        raise ValueError("One of `excd_rates` or `rps` must be provided.")
    if excd_rates and rps:
        raise ValueError("Only one of `excd_rates` or `rps` can be provided.")

    # Ensure we have both RPs and Exceedance rates
    excd_rates = (
        excd_rates
        if excd_rates is not None
        else [sha.utils.rp_to_prob(cur_rp) for cur_rp in rps]
    )
    rps = (
        rps
        if rps is not None
        else [int(np.round(sha.utils.prob_to_rp(cur_excd))) for cur_excd in excd_rates]
    )

    # Load the hazard curve statistics
    hcurve_stats_ffps = hcurve_stats_dir.glob("*statistics.csv")
    mean_hcurves = {
        sha.utils.reverse_im_file_format(ffp.name.rsplit("_", 1)[0]): pd.read_csv(
            ffp, index_col=0
        )["mean"].squeeze()
        for ffp in hcurve_stats_ffps
    }

    # Compute and save UHS
    uhs_df = sha.uhs.compute_uhs(mean_hcurves, excd_rates, rps=rps)
    uhs_df.to_csv(output_ffp)


if __name__ == "__main__":
    app()
