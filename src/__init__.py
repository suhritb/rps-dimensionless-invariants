"""
rps-dimensionless-invariants/src
================================
Core package for the dynamical-similarity neural operator experiments.

Burgers modules (re-exported here)
----------------------------------
solver   : CNAB2 spectral Burgers solver + Cole-Hopf exact reference
dataset  : IC generation, solution caching, feature encodings
models   : NaiveOperator and SimilarityOperator wrappers
plotting : Burgers sweep figures (used only by experiments/burgers/run_burgers_sweep.py)

Compressible modules (import directly, e.g. src.compressible_dataset)
---------------------------------------------------------------------
compressible_solver, compressible_dataset, compressible_models
"""
from .solver import solve_burgers, cole_hopf_solution, apply_scaling_symmetry
from .dataset import (make_ic_vector, solve_config, precompute_solutions,
                      assemble_dataset, features_naive, features_similarity)
from .models import NaiveOperator, SimilarityOperator
from .plotting import (plot_data_efficiency_curve, plot_solution_comparison,
                       plot_physics_check)
