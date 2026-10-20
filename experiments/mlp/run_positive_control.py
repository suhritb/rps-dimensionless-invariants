"""
experiments/mlp/run_positive_control.py
=======================================
Positive control for the recoverability mechanism (compressible MLP).

Four naive feature maps are each compared against the same similarity arm
([log Re, log Ma]):

  naive-log : [log mu, log L, log c0]   log Re = log L - log mu is a linear
                                        readout of the inputs.
  naive-mix : [log mu,     L, log c0]   feature ranges matched to within
                                        1.8-3.4x, but L -- the extrapolation
                                        axis (train L in {0.8..2.4}, test
                                        L in {0.4, 3.0}) -- is presented raw.
  naive-mu  : [    mu, log L, log c0]   L presented exactly as in naive-log;
                                        mu presented raw, so recovering Re
                                        requires a nonlinearity.
  naive-raw : [    mu,     L,     c0]   all inputs raw.

naive-mu separates loss of linear Re-recoverability from the difficulty of
extrapolating in raw L. mu itself is mildly extrapolated: 2-3 of the 10 test
(L, Re) combinations fall outside the training mu range at every k (see
analysis/mu_extrapolation_check_and_param_count_deeponet_vs_fno.py).

All arms use the same seeds and the same solver cache, and every gap is
computed against the similarity arm of the same seed, so comparisons are
paired within seed.

Output: results/positive_control.txt (relative to the project root).

Run from the project root:
    python experiments/mlp/run_positive_control.py --seeds 5
"""

import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from src.compressible_dataset import (
    make_ic_velocity, precompute_solutions, assemble_dataset,
    features_naive, features_similarity,
)
from src.compressible_models import _MLP_KWARGS, _amplitude_weights

N          = 64
Re_values  = np.array([500., 1000., 2000., 5000., 10000.])
Ma_values  = np.array([0.3, 0.5, 0.7])
n_ic       = 20
L_pool     = np.array([0.8, 1.2, 1.6, 2.0, 2.4])
L_test     = np.array([0.4, 3.0])
ks         = [1, 2, 3, 5]


def f_log(u_ic, mu, L, c0):     # naive-log: log Re is a linear readout
    return np.hstack([u_ic, np.log(mu)[:, None], np.log(L)[:, None], np.log(c0)[:, None]])

def f_mix(u_ic, mu, L, c0):     # naive-mix: nonlinear readout, L (extrapolation axis) raw
    return np.hstack([u_ic, np.log(mu)[:, None], L[:, None], np.log(c0)[:, None]])

def f_mu(u_ic, mu, L, c0):      # naive-mu: nonlinear readout, L presented as in f_log
    return np.hstack([u_ic, mu[:, None], np.log(L)[:, None], np.log(c0)[:, None]])

def f_raw(u_ic, mu, L, c0):     # naive-raw: everything raw
    return np.hstack([u_ic, mu[:, None], L[:, None], c0[:, None]])

ARMS = [("log", f_log), ("mix", f_mix), ("mu", f_mu), ("raw", f_raw)]


def make_op(featfn, seed):
    mlp = MLPRegressor(**{**_MLP_KWARGS, "random_state": seed})
    xs, ys = StandardScaler(), StandardScaler()
    class _Op:
        def fit(self, feats, Y):
            X = featfn(*feats); sw = _amplitude_weights(Y)
            xs.fit(X); ys.fit(Y)
            mlp.fit(xs.transform(X), ys.transform(Y), sample_weight=sw); return self
        def score(self, feats, Y):
            X = featfn(*feats)
            pred = ys.inverse_transform(mlp.predict(xs.transform(X)))
            return float((np.linalg.norm(pred-Y,axis=1)/np.linalg.norm(Y,axis=1)).mean())
    return _Op()


def run_seed(seed, verbose=True):
    rng     = np.random.default_rng(seed)
    u_perts = [make_ic_velocity(rng, N) for _ in range(n_ic)]
    cache   = precompute_solutions(u_perts, Re_values, Ma_values, verbose=verbose)
    (u_te, mu_te, L_te, c0_te, Re_te, Ma_te, Y_te) = assemble_dataset(
        u_perts, Re_values, Ma_values, L_test, cache)

    out = {}
    for k in ks:
        (u_tr, mu_tr, L_tr, c0_tr, Re_tr, Ma_tr, Y_tr) = assemble_dataset(
            u_perts, Re_values, Ma_values, L_pool[:k], cache)
        e_sim = make_op(features_similarity, seed).fit(
            (u_tr, Re_tr, Ma_tr), Y_tr).score((u_te, Re_te, Ma_te), Y_te)
        row = {}
        for name, fn in ARMS:
            e = make_op(fn, seed).fit(
                (u_tr, mu_tr, L_tr, c0_tr), Y_tr).score((u_te, mu_te, L_te, c0_te), Y_te)
            row[name] = e / max(e_sim, 1e-9)
        row["_sim"] = e_sim
        out[k] = row
        if verbose:
            print("    k={}: ".format(k) +
                  "  ".join(f"gap({n})={row[n]:5.2f}x" for n, _ in ARMS))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=5)
    seeds = list(range(ap.parse_args().seeds))

    print("=" * 76)
    print("POSITIVE CONTROL — isolating recoverability from L-extrapolation")
    print(f"Seeds: {seeds}   arms: log | mix | mu | raw   (vs similarity)")
    print("  log = [log mu, log L, log c0]   linear readout, L logged")
    print("  mix = [log mu,     L, log c0]   nonlinear, L raw (extrapolation axis)")
    print("  mu  = [    mu, log L, log c0]   nonlinear, L logged")
    print("  raw = [    mu,     L,     c0]   nonlinear, all raw")
    print("=" * 76)

    runs = {}
    for s in seeds:
        print(f"\n--- seed {s} ---")
        runs[s] = run_seed(s)

    geo = lambda v: float(np.exp(np.mean(np.log(np.array(v)))))
    per = {name: {k: [runs[s][k][name] for s in seeds] for k in ks} for name, _ in ARMS}

    print("\n" + "=" * 76)
    print("SUMMARY (geometric mean of per-seed gaps)")
    print("=" * 76)
    print(f"  {'k':>2}  " + "  ".join(f"{'gap('+n+')':>9}" for n, _ in ARMS))
    print("  " + "-" * 52)
    for k in ks:
        print(f"  {k:>2}  " + "  ".join(f"{geo(per[n][k]):>8.2f}x" for n, _ in ARMS))
    pooled = {n: geo(np.concatenate([per[n][k] for k in ks if k >= 2])) for n, _ in ARMS}
    print("  " + "-" * 52)
    print(f"  {'>=2':>2}  " + "  ".join(f"{pooled[n]:>8.2f}x" for n, _ in ARMS))

    print("\n  Sanity: gap(log) must track the published sweep (4.17, 0.97, 1.10, 1.00)")
    pub = {1: 4.17, 2: 0.97, 3: 1.10, 5: 1.00}
    for k in ks:
        g = geo(per["log"][k])
        print(f"    k={k}: {g:.2f}x vs {pub[k]:.2f}x  "
              f"{'ok' if abs(g-pub[k])/pub[k] < 0.25 else 'DIVERGENT'}")

    print("\n" + "=" * 76)
    print("READ-OUT — recoverability vs L-extrapolation")
    print("=" * 76)
    print(f"  pooled k>=2:  log {pooled['log']:.2f}x | mix {pooled['mix']:.2f}x | "
          f"mu {pooled['mu']:.2f}x | raw {pooled['raw']:.2f}x")
    if pooled["mu"] > 1.6:
        print("  -> naive-mu elevated with the extrapolation axis presented as in")
        print("     naive-log: consistent with loss of linear Re-recoverability.")
    elif pooled["mu"] < 1.2:
        print("  -> naive-mu near parity: the naive-mix gap is largely attributable")
        print("     to extrapolation in raw L rather than to recoverability.")
    else:
        print("  -> naive-mu intermediate: both mechanisms contribute (partial")
        print("     dissociation).")

    out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "results")
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, "positive_control.txt")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write("POSITIVE CONTROL — recoverability vs L-extrapolation\n")
        fh.write(f"seeds={seeds} N={N}\n")
        fh.write("log=[log mu,log L,log c0] mix=[log mu,L,log c0] "
                 "mu=[mu,log L,log c0] raw=[mu,L,c0]; vs similarity=[log Re,log Ma]\n\n")
        fh.write(f"{'k':>3} " + " ".join(f"{'geo '+n:>11}" for n, _ in ARMS) + "\n")
        for k in ks:
            fh.write(f"{k:>3} " + " ".join(f"{geo(per[n][k]):>10.3f}x" for n, _ in ARMS) + "\n")
        fh.write(f"{'>=2':>3} " + " ".join(f"{pooled[n]:>10.3f}x" for n, _ in ARMS) + "\n")
        fh.write("\nper-seed gaps:\n")
        for n, _ in ARMS:
            for k in ks:
                fh.write(f"  {n} k={k}: {['%.3f' % v for v in per[n][k]]}\n")
    print(f"\nSaved to {os.path.abspath(p)}")


if __name__ == "__main__":
    main()
