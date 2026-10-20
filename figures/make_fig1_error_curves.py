# Paper Figure 3 (fig1_error_curves.pdf): out-of-band test error of the naive and similarity
# operators for the compressible scale sweep (a) and configuration sweep (b), mean +/- std
# over 5 seeds. Values are taken from results/mlp/multiseed_robustness.txt.
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9.5,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.4,
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "font.family": "serif",
    "pdf.fonttype": 42,   # embed TrueType, not Type 3
    "ps.fonttype": 42,
})

NAIVE_C = "#c0392b"   # red
SIM_C   = "#2471a3"   # blue
GREY    = "#7f8c8d"

# ----------------------------------------------------------------------
# Data (transcribed from results/mlp/multiseed_robustness.txt)
# ----------------------------------------------------------------------
# Compressible L-scale sweep, mean +/- std over 5 seeds
Lk        = np.array([1, 2, 3, 5])
L_naive_m = np.array([0.743, 0.172, 0.141, 0.117])
L_naive_s = np.array([0.401, 0.104, 0.093, 0.078])   # k=1: population std of the per-seed errors (0.40148)
L_sim_m   = np.array([0.189, 0.176, 0.131, 0.119])
L_sim_s   = np.array([0.120, 0.101, 0.092, 0.074])

# Compressible config-count sweep, mean +/- std over 5 seeds
Cm        = np.array([1, 2, 4, 8])
C_naive_m = np.array([2.111, 1.969, 0.553, 0.339])
C_naive_s = np.array([1.004, 1.006, 0.103, 0.156])
C_sim_m   = np.array([1.704, 2.155, 0.564, 0.346])
C_sim_s   = np.array([0.518, 1.325, 0.114, 0.155])

# ======================================================================
# Paper Figure 3: error curves (both operators) for the two sweeps
# ======================================================================
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.9, 2.7))

ax1.errorbar(Lk, L_naive_m, yerr=L_naive_s, marker="o", ms=4, color=NAIVE_C,
             capsize=2.5, label="Naive  [$\\log\\mu,\\log L,\\log c_0$]")
ax1.errorbar(Lk, L_sim_m, yerr=L_sim_s, marker="s", ms=4, color=SIM_C,
             capsize=2.5, label="Similarity  [$\\log Re,\\log Ma$]")
ax1.set_xlabel("Number of training length scales $k$")
ax1.set_ylabel("Test relative $L_2$ error")
ax1.set_title("(a) Scale sweep (compressible)")
ax1.set_xticks(Lk)
ax1.grid(alpha=0.25, lw=0.5)
ax1.legend(frameon=False, loc="upper right")
ax1.set_ylim(bottom=0)

ax2.errorbar(Cm, C_naive_m, yerr=C_naive_s, marker="o", ms=4, color=NAIVE_C,
             capsize=2.5, label="Naive")
ax2.errorbar(Cm, C_sim_m, yerr=C_sim_s, marker="s", ms=4, color=SIM_C,
             capsize=2.5, label="Similarity")
ax2.set_xlabel("Number of training $(Re,Ma)$ pairs $m$")
ax2.set_ylabel("Test relative $L_2$ error")
ax2.set_title("(b) Configuration sweep (compressible)")
ax2.set_xticks(Cm)
ax2.grid(alpha=0.25, lw=0.5)
ax2.legend(frameon=False, loc="upper right")
ax2.set_ylim(bottom=0)

fig1.tight_layout()
fig1.savefig("fig1_error_curves.pdf", bbox_inches="tight")
print("wrote fig1_error_curves")
