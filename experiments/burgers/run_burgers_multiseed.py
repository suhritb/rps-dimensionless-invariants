"""
experiments/burgers/run_burgers_multiseed.py
=====================================
Five-seed Burgers scale sweep, reported in the paper (Appendix D).

The Burgers system receives the same geometric-mean-of-per-seed-ratios
treatment and the same paired bounds as the compressible arms.

WHAT IS HELD FIXED
------------------
Everything from experiments/burgers/run_burgers_sweep.py: N=64,
Re in {15,22,30,45,65}, n_ic=60, L_pool, L_test, ks, ref_L=2.8, TAU/NT from
src.dataset, and the _MLP_KWARGS backbone from src.models. The Burgers
operators do NOT use sample_weight (unlike the compressible ones), so their
fit path is reproduced exactly and only random_state varies.

WHAT VARIES PER SEED
--------------------
  - IC generation : np.random.default_rng(seed)
  - MLP init      : random_state=seed
Exactly the scope used by run_multiseed_robustness.py for the compressible
sweeps, so the two systems are treated identically.

SANITY CHECK
------------
Before the multi-seed loop, this reruns the original single-seed
configuration of run_burgers_sweep.py (rng=1234 including its one extra
physics-check IC draw, random_state=0) and compares it with the original
single-seed values (results/burgers/sweep_numbers.txt). It reproduces them.
The original run is one draw of the initial conditions; the five-seed values
are the reported ones. Use --skip-sanity to skip this pass.

Run from the project root:
    python experiments/burgers/run_burgers_multiseed.py
    python experiments/burgers/run_burgers_multiseed.py --skip-sanity
"""

import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from src.dataset import (
    make_ic_vector, precompute_solutions, assemble_dataset,
    features_naive, features_similarity,
)
from src.models import _MLP_KWARGS

# ── Config: identical to run_burgers_sweep.py ───────────────────────────────
N         = 64
Re_values = np.array([15., 22., 30., 45., 65.])
n_ic      = 60
L_pool    = np.array([2.6, 2.9, 3.2, 2.3, 3.5, 2.0, 3.8])
L_test    = np.array([1.2, 1.6, 4.4, 5.2])
ks        = [1, 2, 3, 5, 7]
ref_L     = 2.8
SEEDS     = [0, 1, 2, 3, 4]

# original single-seed values (rng=1234, random_state=0)
PUBLISHED = {1: 10.18, 2: 0.87, 5: 0.84}   # results/burgers/sweep_numbers.txt


def _make_op(featfn, seed):
    """Reproduces the src.models fit path exactly; only random_state varies.
    Note: the Burgers operators use no sample_weight, unlike the compressible
    ones, so it is not added here."""
    mlp = MLPRegressor(**{**_MLP_KWARGS, "random_state": seed})
    xs, ys = StandardScaler(), StandardScaler()

    class _Op:
        def fit(self, feats, Y):
            X = featfn(*feats)
            xs.fit(X); ys.fit(Y)
            mlp.fit(xs.transform(X), ys.transform(Y))
            return self

        def score(self, feats, Y):
            X = featfn(*feats)
            pred = ys.inverse_transform(mlp.predict(xs.transform(X)))
            return float((np.linalg.norm(pred - Y, axis=1)
                          / np.linalg.norm(Y, axis=1)).mean())
    return _Op()


def run_one(rng_seed, mlp_seed, verbose=True, physics_check_draw=False):
    """One full sweep. Returns {k: (e_naive, e_sim)}.

    physics_check_draw=True reproduces run_burgers_sweep.py exactly: that script
    draws one extra IC (its orbit-invariance physics check) from the generator
    before the n_ic training ICs. The five reported seeds do not make this draw.
    """
    rng = np.random.default_rng(rng_seed)
    if physics_check_draw:
        make_ic_vector(rng, N)
    ics = [make_ic_vector(rng, N) for _ in range(n_ic)]
    cache = precompute_solutions(ics, Re_values, ref_L=ref_L)

    Xic_te, nu_te, L_te, Re_te, Y_te = assemble_dataset(
        ics, Re_values, L_test, cache)

    out = {}
    for k in ks:
        Xic_tr, nu_tr, L_tr, Re_tr, Y_tr = assemble_dataset(
            ics, Re_values, L_pool[:k], cache)
        e_n = _make_op(features_naive, mlp_seed).fit(
            (Xic_tr, nu_tr, L_tr), Y_tr).score((Xic_te, nu_te, L_te), Y_te)
        e_s = _make_op(features_similarity, mlp_seed).fit(
            (Xic_tr, Re_tr), Y_tr).score((Xic_te, Re_te), Y_te)
        out[k] = (e_n, e_s)
        if verbose:
            print(f"    k={k}: naive={e_n:.4f}  sim={e_s:.4f}  "
                  f"gap={e_n/max(e_s,1e-9):.2f}x")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-sanity", action="store_true")
    args = ap.parse_args()

    print("=" * 72)
    print("BURGERS MULTI-SEED SCALE SWEEP")
    print(f"Re {Re_values.tolist()}  n_ic={n_ic}  N={N}  ref_L={ref_L}")
    print(f"L_pool {L_pool.tolist()}")
    print(f"L_test {L_test.tolist()}  ks={ks}  seeds={SEEDS}")
    print("=" * 72)

    # ── sanity: reproduce the published single-seed run ──────────────────────
    if not args.skip_sanity:
        print("\n--- SANITY CHECK: original config (rng=1234, random_state=0) ---")
        orig = run_one(1234, 0, verbose=True, physics_check_draw=True)
        print("\n  published vs this run:")
        ok = True
        for k, ref in PUBLISHED.items():
            g = orig[k][0] / max(orig[k][1], 1e-9)
            near = abs(g - ref) / ref < 0.20
            ok &= near
            print(f"    k={k}: {g:.2f}x vs published {ref}x  "
                  f"{'ok' if near else 'DIVERGENT'}")
        if not ok:
            print("\n  Note: the original single-seed values were not reproduced.")
        else:
            print("  Original single-seed values reproduced.")

    # ── multi-seed ───────────────────────────────────────────────────────────
    runs = {}
    for s in SEEDS:
        print(f"\n--- seed {s} ---")
        runs[s] = run_one(s, s, verbose=True)

    geo = lambda v: float(np.exp(np.mean(np.log(np.array(v)))))

    print("\n" + "=" * 72)
    print("SUMMARY (mean +/- std of errors; gap = geometric mean of per-seed ratios)")
    print("=" * 72)
    print(f"  {'k':>2}  {'naive':>18}  {'similarity':>18}  {'gap':>8}")
    print("  " + "-" * 54)
    table = {}
    for k in ks:
        ns = [runs[s][k][0] for s in SEEDS]
        ss = [runs[s][k][1] for s in SEEDS]
        gs = [n / max(x, 1e-9) for n, x in zip(ns, ss)]
        table[k] = (ns, ss, gs)
        print(f"  {k:>2}  {np.mean(ns):>8.4f} +/- {np.std(ns):<7.4f}  "
              f"{np.mean(ss):>8.4f} +/- {np.std(ss):<7.4f}  {geo(gs):>7.2f}x")

    # pooled bound over k>=2, matching the compressible treatment
    lg = np.concatenate([np.log(np.array(table[k][2])) for k in ks if k >= 2])
    n_tot = len(lg)
    se = lg.std(ddof=1) / np.sqrt(len(table[2][2]))
    tcrit, tpow = 2.776, 0.941     # df=4, as used for the compressible arms
    print("  " + "-" * 54)
    print(f"  pooled k>=2 geometric gap = {np.exp(lg.mean()):.3f}x "
          f"(n={n_tot} seed-k pairs)")
    print(f"  MDE at ~80% power        = {np.exp((tcrit+tpow)*se):.2f}x")

    print("\n  per-seed gaps:")
    for k in ks:
        print(f"    k={k}: " + "  ".join(f"s{s}={g:.2f}x"
              for s, g in zip(SEEDS, table[k][2])))

    # ── save ────────────────────────────────────────────────────────────────
    out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "results")
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, "burgers_multiseed.txt")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write("BURGERS MULTI-SEED SCALE SWEEP\n")
        fh.write(f"seeds={SEEDS}  Re={Re_values.tolist()}  n_ic={n_ic}  N={N}\n")
        fh.write(f"L_pool={L_pool.tolist()}  L_test={L_test.tolist()}  ks={ks}\n")
        fh.write("seed scope: IC rng + MLP random_state (same as compressible)\n\n")
        fh.write(f"{'k':>3} {'naive_mean':>11} {'naive_std':>10} "
                 f"{'sim_mean':>10} {'sim_std':>9} {'geo_gap':>8}\n")
        for k in ks:
            ns, ss, gs = table[k]
            fh.write(f"{k:>3} {np.mean(ns):>11.4f} {np.std(ns):>10.4f} "
                     f"{np.mean(ss):>10.4f} {np.std(ss):>9.4f} {geo(gs):>7.3f}x\n")
        fh.write(f"\npooled k>=2 geometric gap = {np.exp(lg.mean()):.4f}x\n")
        fh.write(f"MDE (~80% power)          = {np.exp((tcrit+tpow)*se):.4f}x\n")
        fh.write("\nper-seed gaps:\n")
        for k in ks:
            fh.write(f"  k={k}: {['%.3f' % g for g in table[k][2]]}\n")
    print(f"\nSaved to {os.path.abspath(p)}")


if __name__ == "__main__":
    main()
