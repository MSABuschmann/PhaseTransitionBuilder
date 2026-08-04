"""
ptbuilder — Phase Transition Builder

Provides a unified Python interface for setting up, running, and analysing
cosmological phase transition / gravitational wave simulations.

Canonical workflow (see notebooks/):
  1. Define a PhysicsModel (potential + instanton).
  2. Run a 2D scan over (γ, t_cut) to get dE/dt(γ, t, k).
  3. Sample a BubblePopulation and compute collision weights.
  4. Bootstrap the predicted 3D GW spectrum from the 2D scan.
  5. Optionally validate against a full sledgehamr 3D run.
"""

from .config      import Config
from .physics     import (Potential, Phi4Potential, Phi4PiecewisePotential,
                           PolynomialPotential, potential_from_hdf5,
                           InstantonProfile, BubbleKinematics, PhysicsModel)
from .nucleation  import (NucleationRate, UniformNucleation,
                           ExponentialNucleation, FixedNucleation,
                           BubbleEvent, CollisionEvent, BubblePopulation)
from .ic          import (BubbleMasterParams, write_2d_setup, write_3d_bubbles)
from .scan        import (ScanConfig, ScanResult, run_scan, load_scan,
                          load_bm_result)
from .sim2d       import (run_bubblemaster, run_solver_1d, run_weights)
from .analysis    import (bootstrap_spectrum, load_weights, write_weights_input,
                           sledgehamr_normalization, bubblemaster_normalization,
                           build_scan_interpolator, sledgehamr_spectrum_range,
                           apply_keffsq)
from .sledgehamr  import (load_output, get_gw_spectrum, get_final_spectrum,
                           get_spectrum_at_time)

__all__ = [
    # config
    "Config",
    # physics
    "Potential", "Phi4Potential", "Phi4PiecewisePotential", "PolynomialPotential",
    "potential_from_hdf5",
    "InstantonProfile", "BubbleKinematics", "PhysicsModel",
    # nucleation
    "NucleationRate", "UniformNucleation", "ExponentialNucleation",
    "FixedNucleation", "BubbleEvent", "CollisionEvent", "BubblePopulation",
    # ic
    "BubbleMasterParams", "write_2d_setup", "write_3d_bubbles",
    # scan
    "ScanConfig", "ScanResult", "run_scan", "load_scan", "load_bm_result",
    # sim2d
    "run_bubblemaster", "run_solver_1d", "run_weights",
    # analysis
    "bootstrap_spectrum", "load_weights", "write_weights_input",
    "sledgehamr_normalization", "bubblemaster_normalization",
    "build_scan_interpolator", "sledgehamr_spectrum_range",
    "apply_keffsq",
    # sledgehamr
    "load_output", "get_gw_spectrum", "get_final_spectrum", "get_spectrum_at_time",
]
