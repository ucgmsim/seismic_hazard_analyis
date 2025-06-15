import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

import oq_wrapper as oqw
import qcore.nhm as nhm
import seismic_hazard_analysis as sha
from qcore import coordinates as coords
from source_modelling import sources


def process_site(
    site_item: tuple[str, dict],
    ds_source_df: pd.DataFrame,
    ds_erf_df: pd.DataFrame,
    faults: list[sources.Fault],
    flt_erf_df: pd.DataFrame,
    gmm_mapping: dict,
    ims: list[str],
):
    site_name, site_params = site_item
    site_coords = np.array([[site_params["lat"], site_params["lon"], 0]])
    site_nztm = coords.wgs_depth_to_nztm(site_coords)[0, [1, 0, 2]]
    site_properties = {
        "vs30": site_params["vs30"],
        "z1p0": site_params["z1p0"],
        "vs30measured": True,
    }

    ### DS Hazard
    ds_hazard = sha.nshm_2010.compute_gmm_ds_hazard(
        ds_source_df, ds_erf_df, site_nztm, site_properties, gmm_mapping, ims
    )

    ### Fault Hazard
    flt_hazard = sha.nshm_2010.compute_gmm_flt_hazard(
        site_nztm, site_properties, flt_erf_df, gmm_mapping, ims, faults=faults
    )

    # Combine into single dataframe
    comb_hazard = {
        cur_im: pd.concat((ds_hazard[cur_im], flt_hazard[cur_im]), axis=1).rename(
            columns={0: "ds", 1: "flt"}
        )
        for cur_im in ims
    }

    ### Save hazard data
    output_dir = Path(__file__).parent / "hazard_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / f"{site_name}_hazard.pkl", "wb") as f:
        pickle.dump(comb_hazard, f)


def main():
    # Load the hazard bench config
    config_fp = Path(__file__).parent.parent / "nshm2010_hazard_bench_config.yaml"
    with open(config_fp, "r") as f:
        config = yaml.safe_load(f)

    # Periods to compute hazard for
    ims = config["ims"]

    # GMMs to use for each tectonic type
    gmm_mapping = {
        oqw.constants.TectType.ACTIVE_SHALLOW: oqw.get_model_from_str(
            config["gmm_config"]["ACTIVE_SHALLOW"]
        ),
        oqw.constants.TectType.SUBDUCTION_SLAB: oqw.get_model_from_str(
            config["gmm_config"]["SUBDUCTION_SLAB"]
        ),
        oqw.constants.TectType.SUBDUCTION_INTERFACE: oqw.get_model_from_str(
            config["gmm_config"]["SUBDUCTION_INTERFACE"]
        ),
    }

    # Load the ERF files
    background_ffp = (
        Path(__file__).parent.parent.parent
        / "data/NSHM2010"
        / "NZBCK2015_Chch50yearsAftershock_OpenSHA_modType4.txt"
    )
    ds_erf_ffp = (
        Path(__file__).parent.parent.parent / "data/NSHM2010" / "NZ_DSModel_2015.txt"
    )
    fault_erf_ffp = (
        Path(__file__).parent.parent.parent / "data/NSHM2010" / "NZ_FLTModel_2010.txt"
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

    for cur_site in tqdm(config["sites"].items(), desc="Processing sites"):
        process_site(
            cur_site, ds_rupture_df, ds_erf_df, faults, flt_erf_df, gmm_mapping, ims
        )


if __name__ == "__main__":
    main()
