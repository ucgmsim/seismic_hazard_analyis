import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import qcore.nhm as nhm
import seismic_hazard_analysis as sha
from empirical.util.classdef import GMM, TectType
from qcore import coordinates as coords

config = yaml.safe_load(
    (Path(__file__).parent / "nshm2010_hazard_bench_config.yaml").read_text()
)

root_dir = Path(__file__).parent.parent 

@pytest.fixture(scope="module")
def erf_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, any]]:
    # Load the ERF files
    background_ffp = (root_dir / "data/NSHM2010"
        / "NZBCK2015_Chch50yearsAftershock_OpenSHA_modType4.txt")
    ds_erf_ffp = root_dir / "data/NSHM2010" / "NZ_DSModel_2015.txt"
    fault_erf_ffp = (
        root_dir / "data/NSHM2010" / "NZ_FLTModel_2010.txt"
    )

    ds_erf_df = pd.read_csv(ds_erf_ffp, index_col="rupture_name")
    ds_rupture_df = sha.nshm_2010.get_ds_source_df(background_ffp)

    flt_erf = nhm.load_nhm(fault_erf_ffp)
    flt_erf_df = nhm.load_nhm_df(str(fault_erf_ffp))

    # Create fault objects
    faults = {
        cur_name: sha.nshm_2010.get_fault_objects(cur_fault)
        for cur_name, cur_fault in flt_erf.items()
    }

    return ds_rupture_df, ds_erf_df, flt_erf_df, faults


@pytest.fixture(scope="module")
def gmm_mapping() -> dict[TectType, GMM]:
    # GMMs to use for each tectonic type
    return {
        TectType.ACTIVE_SHALLOW: GMM.from_str(config["gmm_config"]["ACTIVE_SHALLOW"]),
        TectType.SUBDUCTION_SLAB: GMM.from_str(config["gmm_config"]["SUBDUCTION_SLAB"]),
        TectType.SUBDUCTION_INTERFACE: GMM.from_str(
            config["gmm_config"]["SUBDUCTION_INTERFACE"]
        ),
    }


@pytest.mark.parametrize("site_name", config["sites"])
@pytest.mark.parametrize("im", config["ims"])
def test_site_im_hazard(
    site_name: str,
    im: str,
    erf_data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, any]],
    gmm_mapping: dict[TectType, GMM],
):
    ds_rupture_df, ds_erf_df, flt_erf_df, faults = erf_data
    site_params = config["sites"][site_name]
    site_coords = np.array([[site_params["lat"], site_params["lon"], 0]])
    site_nztm = coords.wgs_depth_to_nztm(site_coords)[0, [1, 0, 2]]
    site_vs30 = site_params["vs30"]
    site_z1p0 = site_params["z1p0"]

    ### DS Hazard
    ds_hazard = sha.nshm_2010.compute_gmm_ds_hazard(
        ds_rupture_df, ds_erf_df, site_nztm, site_vs30, site_z1p0, gmm_mapping, [im]
    )

    ### Fault Hazard
    flt_hazard = sha.nshm_2010.compute_gmm_flt_hazard(
        site_nztm, site_vs30, site_z1p0, flt_erf_df, gmm_mapping, [im], faults=faults
    )

    # Combine into single dataframe
    comb_hazard = pd.concat((ds_hazard[im], flt_hazard[im]), axis=1).rename(
        columns={0: "ds", 1: "flt"}
    )

    # Load benchmark hazard data
    benchmark_dir = root_dir / "tests/nshm2010_bench_data" / "hazard_data"
    with open(benchmark_dir / f"{site_name}_hazard.pkl", "rb") as f:
        benchmark_hazard = pickle.load(f)

    # Compare the generated hazard data with the benchmark data for the specific IM
    pd.testing.assert_frame_equal(comb_hazard, benchmark_hazard[im])
