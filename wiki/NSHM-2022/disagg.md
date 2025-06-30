# NSHM 2022 Disaggregation

## OQ Disagg Execution

The disaggregation is run directly via OQ, and takes a significant amount of time (~6 hours on `hypocentre`) and memory. 
Memory usage is generally the main limiting factor, and can easily exceed 100Gb, depending on the disagg bin specification. 
Would not recommend trying to run on a standard desktop, `hypocentre` is generally a good choice.

### Setup
- Create a run directoy
- Add the NSHM GMM logic tree file,  `NZ_NSHM_GMM_LT_final_EE.xml`
- Add the source model file, `source_model.xml`
- Add the `source_definitions` folder, generally via a symbolic link
- Add the `site.csv` 
- Add the `disagg_job.ini`

**Example `site.csv`**
```
lon,lat,vs30,z1pt0,z2pt5,backarc
172.63,-43.53,400.0,355.7170357122867,1.2646109912757877,False
```

**Disagg Configuration**
All configuration is done in the `disagg_job.ini` file, an example is shown below:
```
[general]
description = Disaggregation
random_seed = 25
calculation_mode = disaggregation
ps_grid_spacing = 30

[logic_tree]
number_of_logic_tree_samples = 0

[erf]
rupture_mesh_spacing = 4
width_of_mfd_bin = 0.1
complex_fault_mesh_spacing = 10.0
area_source_discretization = 10.0

[site_params]
site_model_file = ./sites.csv

[calculation]
source_model_logic_tree_file = source_model.xml
gsim_logic_tree_file = ./NZ_NSHM_GMM_LT_final_EE.xml
investigation_time = 1.0
truncation_level = 4
maximum_distance = {'Active Shallow Crust': [(4.0, 0), (5.0, 100.0), (6.0, 200.0), (9.5, 300.0)],
        'Subduction Interface': [(5.0, 0), (6.0, 200.0), (10, 500.0)],
        'Subduction Intraslab': [(5.0, 0), (6.0, 200.0), (10, 500.0)]}

[output]
individual_curves = true

[disagg]
max_sites_disagg = 1
mag_bin_width = 0.2
disagg_outputs = TRT_Mag_Dist_Eps
disagg_bin_edges = {'dist': [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200], 'eps': [-5, -2, -1, -0.5, 0, 0.5, 1.0, 2.0, 5.0]}
iml_disagg = {"SA(0.075)": 0.6486028178906605}
num_rlzs_disagg = 0
```

The `general`, `logic_tree`, `erf`, `site_params`, `calculation`, `output` sections should generally not be modified.
The only exception is `number_of_logic_tree_samples`, which can be used to specify the number of logic tree branches to use via MC sampling. This can result in significant speed-up, however will not give the exact result. 

The `disagg` section allows setting of the disagg bins. 
The `disagg_outputs` field specifies along which dimensions to perform disaggregation, in this example it is set to `TRT_Mag_Dist_Eps`, which corresponds to Tectonic Type, Magnitude, Distance, Epsilon. For supported `disagg_outputs` values see [here](https://docs.openquake.org/oq-engine/master/manual/user-guide/outputs/disaggregation-outputs.html#hazard-disaggregation).
The `mag_bin_width` field, specifies the width of the magnitude bins, 0.2 in this case.
The `disagg_bin_edges` dictionary allows for manual definition of the bins, in this case for distance and epsilon. This could also be used for magnitude instead of the `mag_bin_width` field.
In this example the level at which to perform disagg is set via the IM directly using `iml_disagg`, an alternative option is to use `poes_disagg` which selects it based on the probability of exceedance. 
However, this requires the computation of hazard first, increasing the computational time. See [here](https://docs.openquake.org/oq-engine/master/manual/user-guide/configuration-file/classical-psha-config.html#seismic-hazard-disaggregation) for additional information on the disagg options.

Notes:
- Increasing the number of bins increases the memory usage

### Run
Starting the run is straightforward, navigate to the run directory, and execute 
`oq engine --run disagg_job.ini -p use_rates=true -p max_potential_paths=1000000`
and proceed to wait a few hours. It is also a good idea to check memory usage periodically.