# NSHM 2022 Disaggregation

## OQ Hazard Execution

The disaggregation is run directly via OQ, and can take several hours and memory. 
Memory usage is generally the main limiting factor, and can easily exceed 100Gb, depending on the number of IM levels.
Would not recommend trying to run on a standard desktop, `hypocentre` is generally a good choice.

### Setup
- Create a run directoy
- Add the NSHM GMM logic tree file,  `NZ_NSHM_GMM_LT_final_EE.xml`
- Add the source model file, `source_model.xml`
- Add the `source_definitions` folder, generally via a symbolic link
- Add the `site.csv` 
- Add the `hazard_job.ini`

**OQ Engine Version**
Version 3.23.2 works correctly, there was some issues with earlier version so would not recommend using those.

**Example `site.csv`**
```
lon,lat,vs30,z1pt0,z2pt5,backarc
172.63,-43.53,400.0,355.7170357122867,1.2646109912757877,False
```

**Hazard Configuration**
All configuration is done in the `hazard_job.ini` file, an example is shown below:
```

```

