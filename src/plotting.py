"""
plotting.py
===========
Plotting utilities for the similarity-operator experiment.
All functions save to disk and return the matplotlib Figure object.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_data_efficiency_curve(ks, naive_errors, sim_errors, save_path):
    """
    Plot test error vs number of distinct training scales.

    Parameters
    ----------
    ks           : list[int]   number of training scales at each sweep point
    naive_errors : list[float] mean rel-L2 for naive operator
    sim_errors   : list[float] mean rel-L2 for similarity operator
    save_path    : str         output file path (.png or .pdf)
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ks, naive_errors, "o-", color="#e05a3a", lw=2, ms=7,
            label="Naive  [log ν, log L]")
    ax.plot(ks, sim_errors, "s-", color="#3a7ec0", lw=2, ms=7,
            label="Similarity-invariant  [log Re]")
    ax.axhline(0.05, ls=":", color="gray", lw=1, label="5% threshold")
    ax.set_xlabel("Number of distinct scales sampled in training", fontsize=12)
    ax.set_ylabel("Mean relative L2 error  (out-of-band test scales)", fontsize=12)
    ax.set_title("Exact similarity invariance buys scale-data efficiency", fontsize=13)
    ax.legend(fontsize=10); ax.grid(alpha=0.3)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    return fig


def plot_solution_comparison(x, u0, u_naive, u_sim, u_true, nu, L, Re,
                             save_path):
    """
    Plot IC, ground truth, and model predictions for one test configuration.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(x, u0, "k--", lw=1.5, label="IC  u(x,0)")
    axes[0].plot(x, u_true, "k-", lw=2, label="Ground truth  u(x,T)")
    axes[0].plot(x, u_naive, color="#e05a3a", lw=1.5, ls="-.",
                 label="Naive prediction")
    axes[0].plot(x, u_sim, color="#3a7ec0", lw=1.5, ls="--",
                 label="Similarity prediction")
    axes[0].set_xlabel("x"); axes[0].set_ylabel("u")
    axes[0].set_title(f"Test config: ν={nu:.4f}, L={L:.2f}, Re={Re:.1f}")
    axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)

    err_naive = np.abs(u_naive - u_true)
    err_sim = np.abs(u_sim - u_true)
    axes[1].plot(x, err_naive, color="#e05a3a", lw=1.5, label="Naive |error|")
    axes[1].plot(x, err_sim, color="#3a7ec0", lw=1.5, label="Similarity |error|")
    axes[1].set_xlabel("x"); axes[1].set_ylabel("|prediction − truth|")
    axes[1].set_title("Pointwise absolute error")
    axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    return fig


def plot_physics_check(Re_vals, errors, save_path):
    """Bar chart confirming similarity-orbit solution invariance."""
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(range(len(Re_vals)), errors, color="#3a7ec0", alpha=0.8)
    ax.set_xticks(range(len(Re_vals)))
    ax.set_xticklabels([f"Re={r:.0f}" for r in Re_vals])
    ax.set_ylabel("Max relative L2 across orbit members")
    ax.set_title("Physics check: orbit members share identical solution vector")
    ax.set_yscale("log"); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    return fig
