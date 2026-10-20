# Paper Figure 2 (fig3_fno.pdf): FNO replication of the compressible scale sweep.
# (a) FNO out-of-band test error, mean +/- std (population, ddof=0) over 5 seeds.
# (b) Gap = naive / similarity error for the MLP and FNO backbones: geometric mean of the
#     per-seed ratios with a 95% CI (Student t on log-gaps, df=4), the same statistic as the
#     tables and the other figures.
# Reads results/mlp/multiseed_robustness.txt and results/fno/fno_compressible_lsweep.json.
# Run from the repository root or from anywhere; writes fig3_fno.pdf to the current directory.
import os, re, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy import stats

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.5, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.linewidth": 0.8,
    "lines.linewidth": 1.4, "figure.dpi": 200, "savefig.dpi": 300, "font.family": "serif",
    "pdf.fonttype": 42, "ps.fonttype": 42,   # embed TrueType, not Type 3
})
NAIVE_C, SIM_C, GREY = "#c0392b", "#2471a3", "#7f8c8d"
K = [1, 2, 3, 5]
TC = float(stats.t.ppf(0.975, 4))


def geo_ci(gaps):
    lg = np.log(np.asarray(gaps, dtype=float))
    m, se = lg.mean(), lg.std(ddof=1) / np.sqrt(len(lg))
    return np.exp(m), np.exp(m - TC * se), np.exp(m + TC * se)


# MLP per-seed gaps, as logged by experiments/mlp/run_multiseed_robustness.py
with open(os.path.join(ROOT, "results", "mlp", "multiseed_robustness.txt"), encoding="utf-8") as fh:
    ms = fh.read()
lsweep = ms.split("L-sweep per-seed gaps:")[1]
mlp = {k: [float(g) for g in re.findall(r"s\d=([\d.]+)x",
                                         re.search(rf"^\s*k={k}: (.*)$", lsweep, re.M).group(1))] for k in K}
# FNO per-seed errors, as logged by experiments/fno/run_fno_scale_sweep.py
with open(os.path.join(ROOT, "results", "fno", "fno_compressible_lsweep.json"), encoding="utf-8") as fh:
    fno = json.load(fh)
fno_gaps = {k: np.array(fno[str(k)]["naive_all"]) / np.array(fno[str(k)]["sim_all"]) for k in K}

fig, (axl, axr) = plt.subplots(1, 2, figsize=(6.9, 2.7))

for key, color, marker, label in [("naive", NAIVE_C, "o", r"Naive  [$\log\mu,\log L,\log c_0$]"),
                                  ("sim", SIM_C, "s", r"Similarity  [$\log Re,\log Ma$]")]:
    m = [np.mean(fno[str(k)][f"{key}_all"]) for k in K]
    s = [np.std(fno[str(k)][f"{key}_all"]) for k in K]
    axl.errorbar(K, m, yerr=s, marker=marker, ms=4, color=color, capsize=2.5, label=label)
axl.set_xlabel("Number of training length scales $k$")
axl.set_ylabel("Test relative $L_2$ error")
axl.set_title("(a) FNO scale sweep (compressible)")
axl.set_xticks(K); axl.grid(alpha=0.25, lw=0.5)
axl.legend(frameon=False); axl.set_ylim(bottom=0)

axr.axhline(1.0, color=GREY, ls="--", lw=1.0, label="no advantage")
for gaps, color, marker, label in [(mlp, "#884ea0", "o", "MLP backbone"),
                                   (fno_gaps, "#117a65", "^", "FNO backbone")]:
    g = np.array([geo_ci(gaps[k]) for k in K])
    axr.errorbar(K, g[:, 0], yerr=[g[:, 0] - g[:, 1], g[:, 2] - g[:, 0]],
                 marker=marker, ms=4, color=color, capsize=2.5, label=label)
axr.set_xlabel("Number of training length scales $k$")
axr.set_ylabel("Gap = naive err / similarity err")
axr.set_title("(b) Same collapse, two backbones")
axr.set_xticks(K); axr.grid(alpha=0.25, lw=0.5)
axr.legend(frameon=False)

fig.tight_layout()
fig.savefig("fig3_fno.pdf", bbox_inches="tight")
for name, gaps in [("MLP", mlp), ("FNO", fno_gaps)]:
    print(name, " ".join(f"k={k}: {geo_ci(gaps[k])[0]:.2f}x" for k in K))
print("wrote fig3_fno.pdf")
