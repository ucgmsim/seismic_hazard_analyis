"""Example code to generate a 3D disaggregation plot for NSHM2022 data (from the NSHM website)."""

from pathlib import Path

import numpy as np
import pandas as pd

from pygmt_helper.plots import disagg_plot

disagg_ffp = Path(__file__).parent.parent.parent / "data/NSHM2022/disagg_chch_nshm.csv"

# Load the data & rename columns
disagg_df = pd.read_csv(disagg_ffp, skiprows=1)
disagg_df = disagg_df.rename(
    columns={
        "distance (km)": "dist",
        "epsilon (sigma)": "epsilon",
        "% contribution to hazard": "contribution",
        "magnitude": "mag",
    }
)

# Define the magnitude and distance bin edges
mag_bin_edges = np.arange(5.0, 10.2, 0.2)
dist_bin_edges = np.array(
    [0, 5, 10, 15, 20, 30, 40, 50, 60, 80, 100, 140, 180, 220, 260, 320, 380, 500]
)
dist_bin_centres = np.array(
    [
        (dist_bin_edges[i] + dist_bin_edges[i + 1]) / 2
        for i in range(dist_bin_edges.size - 1)
    ]
)
dist_bin_widths = np.diff(dist_bin_edges)
dist_bin_mapping = dict(zip(dist_bin_centres, dist_bin_widths))

# For some reason the distance bin centres in the
# disaggregation data are incorrect, fix them here.
incorrect_dist_bins = disagg_df.dist.unique()
assert incorrect_dist_bins.size == dist_bin_centres.size
disagg_df["dist"] = disagg_df["dist"].replace(
    dict(zip(incorrect_dist_bins, dist_bin_centres))
)

# Get min/max values for magnitude and distance
min_mag, max_mag = mag_bin_edges[0], mag_bin_edges[-1]
min_dist, max_dist = dist_bin_edges[0], dist_bin_edges[-1]

### Tectonic Region Type color coding
# Get contribution for each magnitude, distance and tectonic region type (TRT) bin
group_disagg_df = (
    disagg_df.groupby(["mag", "dist", "TRT"])["contribution"].sum().reset_index()
)
group_disagg_df["mag_bin_width"] = 0.2
group_disagg_df["dist_bin_width"] = group_disagg_df["dist"].map(dist_bin_mapping)

disagg_plot(
    group_disagg_df,
    (min_dist, max_dist, min_mag, max_mag),
    "TRT",
    {
        "Active Shallow Crust": (None, "blue"),
        "Subduction Interface": (None, "orange"),
        "Subduction Intraslab": (None, "red"),
    },
    Path("/Users/claudy/dev/work/tmp/disagg_plots/test.png"),
)

### Epsilon color coding
# # Get contribution for each magnitude, distance and epsilon bin
# group_disagg_df = (
#     disagg_df.groupby(["mag", "dist", "epsilon"])["contribution"].sum().reset_index()
# )
# group_disagg_df["mag_bin_width"] = 0.2
# group_disagg_df["dist_bin_width"] = group_disagg_df["dist"].map(dist_bin_mapping)

# categories = group_disagg_df.epsilon.unique()
# category_colors = sns.color_palette("rocket", n_colors=len(categories))
# category_specs = {cur_eps: (None, mcolors.to_hex(category_colors[i])) for i, cur_eps in enumerate(categories)}

# disagg_plot(
#     group_disagg_df,
#     (min_dist, max_dist, min_mag, max_mag),
#     "epsilon",
#     category_specs,
#     Path("/Users/claudy/dev/work/tmp/disagg_plots/test.png"),
# )

print("wtf")