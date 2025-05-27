from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from qcore import coordinates as coords
from qcore import nhm
from source_modelling import sources

from .. import site_source


def read_ds_nhm(background_ffp: Path) -> pd.DataFrame:
    """
    Reads a background seismicity file.
    The txt file is formatted for OpenSHA.

    Parameters
    ----------
    background_ffp: Path
        The path to the background seismicity file

    Returns
    -------
    pd.DataFrame
        The background seismicity as a dataframe
    """
    return pd.read_csv(
        background_ffp,
        skiprows=5,
        sep=r"\s+",
        header=None,
        names=[
            "a",
            "b",
            "M_min",
            "M_cutoff",
            "n_mags",
            "totCumRate",
            "source_lat",
            "source_lon",
            "source_depth",
            "rake",
            "dip",
            "tect_type",
        ],
    )


def create_ds_rupture_name(
    lat: float, lon: float, depth: float, mag: float, tect_type: str
):
    """
    Create a unique name for the distributed seismicity source.
    A source represents a single rupture, and a fault is a
    collection of ruptures at a certain point (lat, lon, depth).

    Parameters
    ----------
    lat: float
    lon: float
    depth: float
    mag: float
    tect_type: str

    Returns
    -------
    str
        The unique name of the rupture source
    """
    return f"{create_ds_fault_name(lat, lon, depth)}--{mag}_{tect_type}"


def create_ds_fault_name(lat: float, lon: float, depth: float):
    """
    Create the unique name for the fault.

    A fault is a collection of ruptures at a
    certain point (lat, lon, depth).

    Parameters
    ----------
    lat: float
    lon: float
    depth: float

    Returns
    -------
    str
        The unique name of the fault
    """
    return f"{lat}_{lon}_{depth}"


def get_ds_source_df(background_ffp: Path):
    """
    Convert the background seismicity to a rupture dataframe.
    Magnitudes are sampled for each rupture.

    Todo: This should be re-written and test cases added

    Parameters
    ----------
    background_ffp

    Returns
    -------
    rupture_df
        A dataframe with columns rupture_name, fault_name, mag,
        dip, rake, dbot, dtop, tect_type, lat, lon, depth
    """
    background_df = read_ds_nhm(background_ffp)
    data = np.ndarray(
        sum(background_df.n_mags),
        dtype=[
            ("rupture_name", str, 64),
            ("fault_name", str, 64),
            ("mag", np.float64),
            ("dip", np.float64),
            ("rake", np.float64),
            ("dbot", np.float64),
            ("dtop", np.float64),
            ("tectonic_type", str, 64),
            ("lat", np.float64),
            ("lon", np.float64),
            ("depth", np.float64),
        ],
    )

    indexes = np.cumsum(background_df.n_mags.values)
    indexes = np.insert(indexes, 0, 0)
    index_mask = np.zeros(len(data), dtype=bool)

    for i, line in background_df.iterrows():
        index_mask[indexes[i] : indexes[i + 1]] = True

        # Generate the magnitudes for each rupture
        sample_mags = np.linspace(line.M_min, line.M_cutoff, line.n_mags)

        for ii, iii in enumerate(range(indexes[i], indexes[i + 1])):
            data["rupture_name"][iii] = create_ds_rupture_name(
                line.source_lat,
                line.source_lon,
                line.source_depth,
                sample_mags[ii],
                line.tect_type,
            )

        data["fault_name"][index_mask] = create_ds_fault_name(
            line.source_lat, line.source_lon, line.source_depth
        )
        data["rake"][index_mask] = line.rake
        data["dip"][index_mask] = line.dip
        data["dbot"][index_mask] = line.source_depth
        data["dtop"][index_mask] = line.source_depth
        data["tectonic_type"][index_mask] = line.tect_type
        data["mag"][index_mask] = sample_mags
        data["lat"][index_mask] = line.source_lat
        data["lon"][index_mask] = line.source_lon
        data["depth"][index_mask] = line.source_depth

        index_mask[indexes[i] : indexes[i + 1]] = False  # reset the index mask

    rupture_df = pd.DataFrame(data=data)
    rupture_df["fault_name"] = rupture_df["fault_name"].astype("category")
    rupture_df["rupture_name"] = rupture_df["rupture_name"].astype("category")
    rupture_df["tectonic_type"] = rupture_df["tectonic_type"].astype("category")
    rupture_df = rupture_df.set_index("rupture_name")

    rupture_df[["nztm_y", "nztm_x", "depth"]] = coords.wgs_depth_to_nztm(
        rupture_df[["lat", "lon", "depth"]].values
    )

    return rupture_df


def get_fault_objects(fault_nhm: nhm.NHMFault) -> sources.Fault:
    """
    Converts a NHM fault to a source object

    Parameters
    ----------
    fault_nhm: nhm.NHMFault

    Returns
    -------
    sources.Fault
        Source object representing the fault
    """
    trace_points_nztm = coords.wgs_depth_to_nztm(fault_nhm.trace[:, ::-1])

    n_planes = fault_nhm.trace.shape[0] - 1
    planes = []
    for i in range(n_planes):
        plane = sources.Plane.from_nztm_trace(
            np.array([trace_points_nztm[i], trace_points_nztm[i + 1]]),
            fault_nhm.dtop,
            fault_nhm.dbottom,
            fault_nhm.dip,
            dip_dir=fault_nhm.dip_dir
        )
        planes.append(plane)

    return sources.Fault(planes)


def run_site_to_source_dist(faults: dict[str, sources.Fault], site_nztm_coords: np.ndarray[float]):
    """
    Computes the source to site distances for the given faults and sites

    Parameters
    ----------
    faults: dictionary of Fault objects
        Fault object for each fault id
    site_nztm_coords: array of floats
        The site coordinates in NZTM (X, Y, Depth)
    """
    fault_id_mapping = {cur_name: i for i, cur_name in enumerate(faults.keys())}
    plane_nztm_coords = []
    scenario_ids = []
    scenario_section_ids = []
    segment_section_ids = []
    for cur_name, cur_fault in tqdm(faults.items(), desc="Fault distances"):
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

    # Compute rupture scenario distances
    dist_df = site_source.get_scenario_distances(
        scenario_ids,
        scenario_section_ids,
        plane_nztm_coords,
        segment_section_ids,
        site_nztm_coords,
    )

    return dist_df