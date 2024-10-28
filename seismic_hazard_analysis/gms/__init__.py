"""
Functionality for running ground motion selection based on the Generalized Conditional Intensity Measure (GCIM) approach.

Modules:
- distributions: Classes for the common distributions used in the GCIM approach.
- gcim_emp: Functions for computing GCIM from empirical models.
- gcim_sim: Functions for computing GCIM from physics-based GM simulations.
- gms_emp: Functions for running ground motion selection based on the empirical GCIM.
- plots: Functions for plotting GCIM and GMS results.

References:
- Bradley (2010), "A generalized conditional intensity measure approach and holistic ground-motion selection".
- Bradley (2012), "A ground motion selection algorithm based on the generalized conditional intensity measure approach"
- Bradley (2015), "Ground motion selection for simulation-based seismic hazard and structural reliability assessment"
"""

from . import gcim_emp, gcim_sim, gms_emp, plots
from .distributions import (
    CondIMjDist,
    Multi_lnIM_IMj_Rup,
    Uni_lnIMi_IMj,
    Uni_lnIMi_IMj_Rup,
    UniIMiDist,
)
