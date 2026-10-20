# Development log (superseded by the paper)

This is the working log kept while the compressible experiments were developed. It is superseded by the paper: gaps here are arithmetic means of per-seed ratios (the paper uses geometric means), and numbers predate the five-seed Burgers runs and the positive control. Kept for transparency about the two methodological pitfalls.

Note: where this log says the naive MLP "discovers" the invariants, read it as "behaves as if it recovers them": the paper treats recovery as an interpretation of the encoding-dependent results, not as something measured (paper §3).

## Compressible Navier-Stokes Similarity Operator: Null Result

## Summary

Exact dynamical-similarity equivariance — encoding Reynolds number Re and Mach
number Ma directly as features instead of the raw dimensional parameters
(μ, L, c₀) — does **not** yield a durable advantage in 1D viscous compressible
Navier-Stokes.  The similarity operator leads by a mean of 1.19× at m = 1 in the config-count
sweep (range 1.04–1.55× across 5 independent seeds) and is matched by the
naive operator from m = 2 onward (mean gap 0.95–0.99×).  This replicates the
collapse seen in the Burgers experiment, now confirmed also against the
two-invariant (Re, Ma) axis and validated across seeds.

---

## System

1D compressible NS in conservation form (ρ, m = ρu, E) on periodic [0,1),
non-dimensionalized with Re = ρ₀U₀L/μ, Ma = U₀/c₀, γ = 1.4, Pr = 0.72.
Solver: pseudo-spectral RK4, N = 64 grid points, CFL = 0.4.
Validated in `tests/ACOUSTIC_VALIDATION_AUDIT.txt`: single-mode acoustic decay
rate matches the cubic-dispersion eigenvalue to relative error 5.6×10⁻⁶
(N=64), with convergence order ~1.8.

Operator architecture (both models identical): 2-layer MLP, 128×128 tanh,
Adam, max 5000 iterations, α = 10⁻⁶.  Feature encodings:

- **Naive**: `[u_IC | log μ | log L | log c₀]` — raw dimensional parameters
- **Similarity**: `[u_IC | log Re | log Ma]` — exact dynamical-similarity invariants

Loss: an amplitude-weighted squared error approximating
`L = mean_i( ‖ŷᵢ − Yᵢ‖² / ‖Yᵢ‖² )`, implemented via sklearn
`sample_weight = 1/‖Yᵢ‖²` (percentile-floored at the 10th-percentile squared
norm).  Target Y is the unmodified true solution field; no decay factor, IC
norm, or output-derived quantity was applied to Y.

---

## Experiment 1: L-scale sweep

**Setting**: Re ∈ {500, 1000, 2000, 5000, 10000}, Ma ∈ {0.3, 0.5, 0.7},
T_final = 1.5.  Random multi-mode ICs, k ∈ {1..5}.  Train on k distinct L
values from pool {0.8, 1.2, 1.6, 2.0, 2.4}, test at out-of-band L ∈ {0.4,
3.0}.  Both operators see all (Re, Ma) combinations at every k.  0 degenerate
samples (‖Y‖ < 10⁻³) on the seed-42 IC set; the five-seed runs have 2/300
(seed 0) and 5/300 (seed 3), see results/diagnostics/dataset_checks.txt and
results/diagnostics/degenerate_robustness.txt.

Mean ± std across 5 independent seeds (seed controls ICs and MLP init):

| k | L_train | naive mean±std | sim mean±std | gap mean±std |
|---|---------|---------------|-------------|-------------|
| 1 | [0.8] | 0.743±0.401 | 0.189±0.120 | 4.2×±0.7 |
| 2 | [0.8, 1.2] | 0.172±0.104 | 0.176±0.101 | 1.00×±0.26 |
| 3 | [0.8, 1.2, 1.6] | 0.141±0.093 | 0.131±0.092 | 1.10×±0.06 |
| 5 | [0.8, 1.2, 1.6, 2.0, 2.4] | 0.117±0.078 | 0.119±0.074 | 1.03×±0.23 |

The similarity operator leads at k = 1 (mean gap 4.2×±0.7, range 3.2–5.1×
across seeds) when trained on a single length scale and tested on unseen
scales.  By k = 2 the gap collapses to ~1.0× (1.00×±0.26): two distinct L
values are sufficient for the naive MLP to discover that L/μ = Re and 1/c₀ =
Ma govern the flow.  At k = 3 similarity retains a slight consistent edge
(1.10×±0.06, similarity ahead in all 5 seeds); at k = 5 both operators are
equivalent (1.03×±0.23).  The earlier single-seed result showing naive leading
at k = 3 (0.6×) was a one-seed artifact not reproduced across seeds.

This is quantitatively identical in character to the Burgers result:

| k | Burgers naive | Burgers sim | gap |
|---|--------------|------------|-----|
| 1 | 0.359 | 0.035 | 10.2× |
| 2 | 0.026 | 0.030 | 0.9× (naive leads) |
| 5 | 0.018 | 0.021 | 0.8× |

The Burgers gap is larger (10× vs ~4× at k = 1), but both systems see the gap
collapse to ~1× already at k = 2 — the naive model needs only two distinct L
values to discover the governing invariants in both systems.

---

## Experiment 2: Configuration-count sweep (the two-invariant test)

**Setting**: same Re, Ma, IC, loss, solver as above.  L_train = {0.8, 1.2,
1.6} fixed (giving the naive model enough L coverage to discover L-scaling).
Sweep m = number of genuinely distinct (Re, Ma) training PAIRS; test on the
remaining (15 − m) held-out (Re, Ma) pairs at L_test = {0.4, 3.0}.
n_train = m × 20 × 3 = 60m; n_test = (15 − m) × 20 × 2 = 40(15 − m).
Each test pair is verified disjoint from the training set at every m.

CONFIG_ORDER (space-filling, so first m entries give good coverage):
(2000, 0.5) → (10000, 0.7) → (500, 0.3) → (500, 0.7) → (10000, 0.3) →
(1000, 0.5) → (5000, 0.5) → (2000, 0.3) → …

Mean ± std across 5 independent seeds:

| m | n_train | n_test | naive mean±std | sim mean±std | gap mean±std |
|---|---------|--------|---------------|-------------|-------------|
| 1 | 60 | 560 | 2.111±1.004 | 1.704±0.518 | 1.19×±0.19 |
| 2 | 120 | 520 | 1.969±1.006 | 2.155±1.325 | 0.95×±0.09 |
| 4 | 240 | 440 | 0.553±0.103 | 0.564±0.114 | 0.99×±0.04 |
| 8 | 480 | 280 | 0.339±0.156 | 0.346±0.155 | 0.98×±0.08 |

The similarity advantage peaks at mean 1.19× (m = 1, range 1.04–1.55× across
seeds) and is indistinguishable from 1.0× at m ≥ 2 (mean gaps 0.95–0.99×,
std ≤ 0.09).  At m = 1–2 both operators score above 1.0 (worse than predicting
zero), so neither can extrapolate from one or two (Re, Ma) anchor points
regardless of feature encoding.  By m = 8, both operators carry mean errors
~0.34 with neither systematically ahead (gap 0.98×±0.08), confirming that the
held-out generalisation task is hard for both and that exact-similarity encoding
gives no durable benefit.

Per-(Re, Ma) at m = 1 — representative single run (seed=42 ICs, MLP random_state=0); aggregate conclusions confirmed by multi-seed check (train on Re=2000, Ma=0.5 only; test: 14 genuinely held-out pairs, 560 samples):

| Re | Ma | naive | similarity | gap |
|----|----|-------|-----------|-----|
| 10000 | 0.70 | 2.282 | 2.201 | 1.0× |
| 500 | 0.30 | 3.168 | 2.613 | 1.2× |
| 500 | 0.70 | 8.224 | 7.921 | 1.0× |
| 10000 | 0.30 | 0.623 | 0.440 | 1.4× |
| 1000 | 0.50 | 0.691 | 0.561 | 1.2× |
| 5000 | 0.50 | 0.438 | 0.302 | 1.5× |
| 2000 | 0.30 | 0.102 | 0.100 | 1.0× |
| 2000 | 0.70 | 2.782 | 2.803 | 1.0× |
| 1000 | 0.30 | 0.745 | 0.584 | 1.3× |
| 5000 | 0.70 | 2.380 | 2.322 | 1.0× |
| 1000 | 0.70 | 3.855 | 3.850 | 1.0× |
| 5000 | 0.30 | 0.423 | 0.317 | 1.3× |
| 10000 | 0.50 | 0.640 | 0.434 | 1.5× |
| 500 | 0.50 | 3.064 | 2.601 | 1.2× |

Per-(Re, Ma) at m = 8 — representative single run (seed=42 ICs, MLP random_state=0); multi-seed mean gap at m=8 is 0.98×±0.08 (7 genuinely held-out pairs, no test pair appears in training set;
280 test samples; all 7 are novel combinations of individually-seen Re and Ma values):

| Re | Ma | naive | similarity | gap |
|----|----|-------|-----------|-----|
| 2000 | 0.70 | 0.460 | 0.609 | 0.8× |
| 1000 | 0.30 | 0.155 | 0.179 | 0.9× |
| 5000 | 0.70 | 0.282 | 0.324 | 0.9× |
| 1000 | 0.70 | 0.479 | 0.691 | 0.7× |
| 5000 | 0.30 | 0.126 | 0.154 | 0.8× |
| 10000 | 0.50 | 0.220 | 0.334 | 0.7× |
| 500 | 0.50 | 1.036 | 0.992 | 1.0× |

---

## Conclusion

**Exact (Re, Ma) similarity equivariance gives a mean advantage of 1.19×
(range 1.04–1.55× across 5 seeds) at m = 1, collapsing to ~1.0× (mean
0.95–0.99×, std ≤ 0.09) from m = 2 onward.  At m = 8 with 7 genuinely
held-out (Re, Ma) pairs, both operators carry mean errors ~0.34 and the gap
is 0.98×±0.08 — statistically indistinguishable.  The gap does not persist
and never clearly favours similarity.  All conclusions hold in every seed.**

Diagnosis: the advantage of the similarity encoding is that the MLP works
directly in the (log Re, log Ma) plane rather than the higher-dimensional
(log μ, log L, log c₀) space.  But the advantage only matters when training
data is so scarce that the naive model cannot discover the invariant from
examples.  At m = 1–2, neither model can interpolate a function it has only
one or two (Re, Ma) points for, regardless of feature encoding (both errors
exceed 1.0 — worse than predicting zero).  From m = 2 onward, the two models
are statistically indistinguishable (mean gap 0.95–0.99×, std ≤ 0.09 across
seeds); any per-seed directional differences reflect sampling noise, not a
systematic effect.

The absolute errors at m = 8 (~0.34 mean across seeds) also reveal a harder
generalisation problem than the flawed experiment suggested (~0.09): with only
8 of 15 (Re, Ma) pairs in training, both operators struggle to extrapolate to
the 7 novel combinations, even though each individual Re and Ma value has
been seen.

**This replicates the Burgers collapse and extends it to the two-invariant
case.  Exact-symmetry equivariance does not yield a durable advantage in this
system: a general-purpose MLP discovers the invariant structure from data at
approximately the same rate it would use built-in knowledge of it.**

---

## Methodological history (why this result is trustworthy)

### Solver validation
The compressible NS solver (`src/compressible_solver.py`, pseudo-spectral RK4)
was validated against the exact acoustic decay rate from the cubic-dispersion
eigenvalue.  Full record in `tests/ACOUSTIC_VALIDATION_AUDIT.txt`.  The key
finding: an early apparent 0.27% error was traced to a measurement artifact
(isentropic IC projects onto both acoustic modes, creating a beat that biases
two-point exponential estimates).  Using the exact finite-Re eigenvector IC and
a 40-point least-squares fit, the solver error is 5.6×10⁻⁶ at N = 64 with
convergence order ~1.8.  The solver is correct.

### Loss fix
The original experiment used the default MSE loss with StandardScaler on Y.
This gave 1e9 relative-L2 errors at Re = 50 because low-Re samples (small
amplitude) were ignored by the MSE gradient.  The fix: per-sample relative-L2
loss via `sample_weight = 1/‖Yᵢ‖²`, with the denominator floored at the 10th
percentile of squared norms to prevent any single near-zero sample from
collapsing the Adam step size.  Target Y remains the unmodified true solution
field throughout.

### Caught artifact: forced-k=1 ICs (development-time observation, reproducible reconstruction produces 3.9×)
An early attempt to fix the optimizer produced a large apparent similarity
advantage at k = 1.  This was identified as an artifact: the fix had modified
`make_ic_velocity` to always include a k = 1 Fourier mode with minimum
amplitude 0.01, which changed the data distribution to eliminate the cases
(ICs with only k ≥ 3 modes) where ‖Y(Re = 50, T = 1.5)‖ ≈ 10⁻¹⁵.

The original modified IC generator was reverted without being saved, so the
exact development-time gap figure (recalled as ~6.1× at k = 1) is not
reproducible from the current codebase.  A faithful reconstruction —
`experiments/diagnostics/diag_forced_k1_artifact.py`, same RNG seed (42), same
`_MIN_K1_AMP = 0.01`, same original Re range [50, 100, 200, 500, 1000] —
produces a **3.9× gap at k = 1** (naive = 0.530, sim = 0.136;
`results/diagnostics/diag_forced_k1_output.txt`).  This confirms the artifact's existence
and direction; an earlier draft cited 3.9×, not the non-reproducible 6.1× figure.

The mechanism is confirmed: forcing k = 1 reduces degenerate test samples from
158/600 to 0/600 total (across all Re values; the degeneracy was concentrated
at Re ≤ 200, mostly Re = 50), shifting the data distribution and inflating the apparent
similarity advantage.  On the same original grid without forcing, the k = 1 gap is 1.2×
(`results/diagnostics/diag_original_ics_output.txt`), compared with 3.9× under
forcing — confirming the forcing inflated the apparent similarity advantage.

When the IC generator was reverted to the original (k ∈ {1..5}, fully random),
both operators are pinned near the predict-zero baseline on the original Re
range [50, 1000]: relative-L2 ranges 0.876–1.123 across all k values, with a
residual 1.2× similarity edge at k = 1 that vanishes by k = 2 — neither
operator learns anything useful (`results/diagnostics/diag_original_ics_output.txt`).
This failure is caused by ~68% of Re = 50 training samples being degenerate
(modes k ≥ 3 completely decay by T = 1.5 at Re = 50), which dominated the
sample-weight loss even after the percentile floor fix.

### Re-range shift and degenerate-sample analysis
A systematic mode-decay diagnostic was run.  At Re = 50, T = 1.5: single-mode
survival is 33% (k = 1), 1.1% (k = 2), 0% (k ≥ 3).  The original Re range
[50, 100, 200, 500, 1000] produces 79/300 degenerate training samples (26%),
concentrated at Re = 50 (68%) and Re = 100 (43%).  The experiment was
redesigned with Re ∈ {500, 1000, 2000, 5000, 10000}, where 0/300 training
samples are degenerate and all Fourier modes k = 1..5 survive above 6% at
T = 1.5.  This is the physically-motivated change: the original Re range
included a regime (Re = 50, T = 1.5) where the solution is in effective
equilibrium for most ICs, not a regime where the operator is learning useful
dynamics.

### Caught bug: Cartesian-product mislabelling in config-count sweep
A review of the original Experiment 2 found a dataset-assembly bug: `assemble_dataset` builds the Cartesian product of the
Re and Ma arrays it receives.  Passing `Re_tr_vals = [c[0] for c in
train_configs]` and `Ma_tr_vals = [c[1] for c in train_configs]` as two
separate arrays caused it to cross every Re value against every Ma value,
producing m² Cartesian pairs rather than m genuinely paired configs:

| m (labelled) | actual training pairs | actual n_train |
|---|---|---|
| 1 | 1 | 60 (correct) |
| 2 | 4 | 240 (should be 120) |
| 4 | 9 | 960 (should be 240) |
| 8 | 15 (full grid!) | 3840 (should be 480) |

At m = 8 the Cartesian product covered all 15 (Re, Ma) combinations, making
the "held-out" test pairs not held out at all.  The flawed n_test counts
(7840/6760/4840/1960) also reflect the same m² × 40 formula on the test side.

Fix: added `assemble_dataset_paired` to `src/compressible_dataset.py`, which
iterates explicit (Re, Ma) tuples instead of a Cartesian product.
`experiments/mlp/run_config_sweep.py` was updated to use it, with an assertion
verifying train/test disjointness at every m.  The corrected n_train/n_test
counts (60/120/240/480 and 560/520/440/280) confirm linear scaling.  The
corrected numbers are what appear in this document.  The null result is
unchanged in its direction; the corrected absolute errors at m = 4–8 are
roughly 4–5× higher, reflecting the reduced training coverage.

### Robustness across seeds

Both sweeps were re-run over five independent seeds (0–4).  Each seed
independently controls IC generation (`np.random.default_rng(seed)`) and MLP
weight initialisation (`MLPRegressor(random_state=seed)`), giving genuinely
independent realizations.  Full output: `results/mlp/multiseed_robustness.txt`;
script: `experiments/mlp/run_multiseed_robustness.py`.

Key findings:

- **L-sweep k=1**: gap ranges 3.2–5.1× (mean 4.2×); the single-seed (seed=42)
  value of 3.0× was the lowest draw.
- **L-sweep k≥2**: gap is 0.64–1.47× across all seeds — no seed shows a large
  persistent advantage for either operator.
- **L-sweep k=3**: similarity is consistently slightly ahead (1.10×±0.06) in
  all 5 seeds.  The earlier single-seed result showing naive leading (0.6×) was
  a one-seed artifact.
- **Config-sweep m=1**: gap ranges 1.04–1.55× (mean 1.19×); the single-seed
  1.1× was near the low end of the distribution.
- **Config-sweep m≥2**: mean gap 0.95–0.99×, std ≤ 0.09 — a tight cluster
  around 1.0× with no systematic directional bias in any seed.
- **Null result**: confirmed in every seed.  No seed shows a large durable
  similarity advantage in either sweep.

The main sweep tables (Experiment 1 and the config-count sweep) report
multi-seed mean ± std; the per-(Re, Ma) breakdown tables are from a
representative single run, labeled as such.

---

## Files

| File | Role |
|------|------|
| `src/compressible_solver.py` | Pseudo-spectral RK4 compressible NS solver |
| `src/compressible_dataset.py` | IC generation, solution caching, feature encodings |
| `src/compressible_models.py` | NaiveCompressibleOperator, SimilarityCompressibleOperator |
| `experiments/mlp/run_compressible_lsweep.py` | L-scale sweep (Experiment 1) |
| `experiments/mlp/run_config_sweep.py` | Config-count sweep (Experiment 2) |
| `experiments/diagnostics/diag_norm_distribution.py` | Mode-decay and degenerate-sample analysis |
| `experiments/diagnostics/diag_forced_k1_artifact.py` | Forced-k=1 artifact reconstruction (see methodological history) |
| `experiments/diagnostics/run_lsweep_original_ics.py` | L-sweep on original Re range [50..1000] with random ICs |
| `tests/ACOUSTIC_VALIDATION_AUDIT.txt` | Solver validation record |
| `results/burgers/sweep_numbers.txt` | Burgers sweep reference numbers |
| `results/diagnostics/diag_forced_k1_output.txt` | Saved output of forced-k=1 diagnostic; gap = 3.9× at k = 1 |
| `results/diagnostics/diag_original_ics_output.txt` | Saved output of original-ICs L-sweep; errors 0.876–1.123 |
| `experiments/mlp/run_multiseed_robustness.py` | 5-seed robustness check for both sweeps |
| `results/mlp/multiseed_robustness.txt` | Full multi-seed output (mean ± std per k and m across seeds 0–4) |
