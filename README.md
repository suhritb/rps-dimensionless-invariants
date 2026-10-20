# The Limits of Exact Symmetry as a Transfer Prior

Code and results for

> **The Limits of Exact Symmetry as a Transfer Prior: Dimensionless Invariants Give No Durable Advantage over Log-Scaled Inputs in 1D Neural Operators**
> RPS Workshop (Representations for the Physical Sciences) at NeurIPS 2026 — poster.

**Summary.** A neural operator conditioned on the exact similarity invariants (Re, Ma) is compared with one conditioned on the log-scaled dimensional parameters (log μ, log L, log c₀), on held-out length scales and (Re, Ma) configurations in 1D compressible Navier–Stokes and viscous Burgers. The invariant encoding wins clearly when training sees a single length scale (MLP 4.17×, FNO 1.35×). With two or more scales, or two or more (Re, Ma) configurations, the advantage disappears. The collapse holds with the training-set size held fixed (sample-matched control). A positive control supports the interpretation that the collapse depends on the invariant being cheap to recover: in log coordinates log Re = log L − log μ is a linear readout, and when that readout is made nonlinear a pooled 2.28–2.74× advantage persists (above 2.0× at every k ≥ 2). The null is therefore specific to smooth 1D flows and log-scaled baseline inputs; raw or partly raw encodings retain a large advantage.

---

## Layout

```
src/                 solvers, dataset assembly, operators
  compressible_solver.py    1D compressible Navier–Stokes, non-dimensional in (Re, Ma)
  compressible_dataset.py   initial conditions, solve cache, dimensional labels, feature encodings
  compressible_models.py    naive / similarity MLP operators (scikit-learn MLPRegressor)
  solver.py, dataset.py, models.py   1D viscous Burgers counterparts
experiments/
  mlp/          scale sweep, configuration sweep, positive control, sample-matched control, absolute-error rerun
  fno/          FNO scale sweep (PyTorch, GPU recommended)
  burgers/      Burgers single-seed and 5-seed sweeps
  diagnostics/  checks behind the methodological notes (pitfall audit trails, dataset and convergence checks)
results/        output files behind every reported number
docs/           development log (superseded by the paper; kept for the pitfall history)
analysis/       confidence intervals, paired tests, power / MDE, gap-statistic checks
figures/        scripts for paper Figures 1–4
tests/          unit tests, solver validation, convergence studies
```

## Setup

```bash
pip install -r requirements.txt     # Python >= 3.10; numpy, scipy, scikit-learn >= 1.7, matplotlib
pip install torch                    # only for experiments/fno
python tests/test_all.py             # 20 unit tests
python tests/validate_compressible.py
```

**scikit-learn ≥ 1.7 is required.** The MLP operators train with an amplitude-weighted squared error (weights 1/‖Y‖², a relative-L2 surrogate) passed through `MLPRegressor.fit(sample_weight=...)`, which earlier versions do not accept. The reported MLP results were produced with scikit-learn 1.9.0 on Windows; the sample-matched control and the diagnostics in `results/diagnostics/dataset_checks.txt` and `results/diagnostics/degenerate_robustness.txt` were run with scikit-learn 1.9.1 on Linux.

Run every script from the repository root.

## Reproducing the results

| Paper result | Script | Output |
|---|---|---|
| MLP scale sweep (k=1: 4.17×; k ≥ 2 ≈ 1) and configuration sweep | `experiments/mlp/run_multiseed_robustness.py` | `results/mlp/multiseed_robustness.txt` |
| Positive control: naive-log / mix / μ / raw | `experiments/mlp/run_positive_control.py` | `results/mlp/positive_control.txt` |
| Sample-matched scale sweep, ~300 training rows at every k (Appendix F) | `experiments/mlp/run_sample_matched_control.py` | `results/mlp/sample_matched_control.{txt,json}` |
| Per-seed absolute errors for all MLP arms | `experiments/mlp/rerun_mlp_abs_errors.py` | `results/mlp/mlp_per_seed_abs_errors.{txt,json}` |
| FNO scale sweep (k=1: 1.35×) | `experiments/fno/run_fno_scale_sweep.py` | `results/fno/fno_compressible_lsweep.{txt,json}` |
| Burgers 5-seed sweep (k=1: 10.15×; pooled k ≥ 2: 1.020×) | `experiments/burgers/run_burgers_multiseed.py` | `results/burgers/burgers_multiseed.txt` |
| Burgers console log, incl. exact reproduction of the original single-seed run (Appendix D) | same script | `results/burgers/burgers_multiseed_console.txt` |
| Burgers original single-seed sweep (superseded) | `experiments/burgers/run_burgers_sweep.py` | `results/burgers/sweep_numbers.txt` |
| Degenerate targets on the diagnosis IC set: 79/300 on the original grid vs 0/300 on Re ≥ 500 (§5) | `experiments/diagnostics/diag_degenerate_count_diagnosis_ics.py` | `results/diagnostics/degenerate_count_diagnosis_ics.txt` |
| Degenerate targets per seed, feature dynamic ranges (§4.3), MLP iterations used | `experiments/diagnostics/diag_dataset_checks.py` | `results/diagnostics/dataset_checks.txt` |
| Every reported MLP gap with near-zero test targets excluded (Appendix C) | `experiments/diagnostics/diag_degenerate_robustness.py` | `results/diagnostics/degenerate_robustness.{txt,json}` |
| Methodological pitfalls (§5) | `experiments/diagnostics/diag_forced_k1_artifact.py`, `experiments/diagnostics/run_lsweep_original_ics.py`, `experiments/diagnostics/diag_norm_distribution.py` | `results/diagnostics/*` (console logs) |
| CIs, p-values, MDEs, pooled gaps | `analysis/*.py` | printed to console |
| Paper Figures 1–4 | `figures/make_fig*.py` (mapping below) | `fig*.pdf` in the working directory |
| Solver validation | `tests/validate_compressible.py`, `tests/convergence_study2.py` | console (recorded in `tests/ACOUSTIC_VALIDATION_AUDIT.txt`) |
| Single-seed exploratory runs (not paper numbers) | `experiments/mlp/run_compressible_lsweep.py`, `experiments/mlp/run_config_sweep.py` | console only |

Experiment scripts write to `results/` (top level). The files under `results/*/` are the outputs reported in the paper.

Most diagnostic scripts print to the console. To save their output, use `python experiments/diagnostics/<script>.py > results/diagnostics/<name>.txt`; on Windows PowerShell use `python experiments\diagnostics\<script>.py | Out-File -Encoding utf8 results\diagnostics\<name>.txt`, because PowerShell's plain `>` writes UTF-16.

Runtime: the MLP experiments run on a laptop CPU in roughly 15–30 minutes each, dominated by the PDE solves. The FNO sweep took about 15 minutes on a T4 GPU; on a CPU expect on the order of a day.

## Notes for interpreting the results

- **Gap statistic.** Every gap in the paper is the geometric mean of per-seed ratios (naive error / similarity error) over 5 seeds. Two results files also print other statistics: the `gap_mean` column of `results/mlp/multiseed_robustness.txt` is an arithmetic mean, and the `Gap` column of `results/fno/fno_compressible_lsweep.txt` is a ratio of means. `analysis/gap_statistic_reconciliation_geomean.py` recomputes the geometric means.
- **Statistics from logged values.** The MLP confidence intervals, p-values and MDEs (paper Table 9) are computed from the per-seed gaps as originally logged, to two decimals (`results/mlp/multiseed_robustness.txt`), so they can be reproduced exactly from the paper's Table 8.
- **Seed scope.** For the MLP and Burgers arms, the seed controls both the initial conditions and the network initialization. For the FNO scale sweep, the initial conditions are fixed and only the network initialization and minibatch order vary, so its spread is narrower and not directly comparable. The FNO trains for a fixed number of epochs, so larger k also means more optimizer steps.
- **Out-of-distribution asymmetry in the scale sweep.** The solver is non-dimensional, so every L on a similarity orbit shares one cached solution. The similarity operator's test inputs and targets therefore coincide with training examples (same ICs and (Re, Ma)), so its scale-sweep test error is in-sample; only the naive operator extrapolates in L. The configuration sweep is where both operators face unseen inputs.
- **Burgers k=1 magnitude.** The Burgers arm has its own grid, number of ICs and 1200-iteration budget, so its 10.15× is not comparable with the compressible 4.17×. The original single-seed sweep (10.18×, 0.87×, 0.84× at k=1, 2, 5) is one draw of the initial conditions; `run_burgers_multiseed.py` reproduces it exactly before running the five reported seeds. Every MLP fit in the compressible scale sweep stops on its tolerance criterion well within its 5000-iteration budget (at most 1108 iterations; `results/diagnostics/dataset_checks.txt`).
- **Exact reproduction.** `rerun_mlp_abs_errors.py` re-ran every MLP experiment and compared 152 values with the saved originals. On Linux with scikit-learn 1.9.1, 144 matched exactly and the other 8 differed by at most 0.004 (four positive-control gaps at k=5, seed 3, and the m=2 configuration-sweep means and standard deviations). The paper reports the originally logged values.
- **Figure and analysis scripts** run without re-running the experiments. `make_fig3_fno.py` reads `results/` directly; the other figure scripts and the analysis scripts contain per-seed values transcribed from `results/`.
- **Figure numbering.** Script file names predate the final paper order: paper Figure 1 = `make_fig0_control.py`, Figure 2 = `make_fig3_fno.py`, Figure 3 = `make_fig1_error_curves.py`, Figure 4 = `make_fig2_gap_collapse.py`. Every gap plotted is the geometric mean of per-seed ratios with a 95% CI (Student t on log-gaps, df=4).

## Citation

```bibtex
@inproceedings{rps2026limits,
  title     = {The Limits of Exact Symmetry as a Transfer Prior: Dimensionless Invariants Give No Durable Advantage over Log-Scaled Inputs in 1D Neural Operators},
  author    = {Bestavemula, Suhrit and Gudepu, Rishith and Raut, Aditya and Raut, Sagar and Movva, Neeraj Sai Venkat and Mburu, Nelson and Loke, Sathvik},
  booktitle = {NeurIPS 2026 Workshop on Representations for the Physical Sciences (RPS)},
  year      = {2026}
}
```

## License

MIT — see `LICENSE`.
