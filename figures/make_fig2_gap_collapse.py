# Paper Figure 4 (fig2_gap_collapse.pdf): (a) compressible MLP scale and configuration sweeps,
# (b) Burgers 5-seed sweep (k = 1,2,3,5,7) vs compressible. Error bars are 95% CIs on the
# geometric mean of per-seed ratios (Student t, df=4). Per-seed values are taken from
# results/mlp/multiseed_robustness.txt and results/burgers/burgers_multiseed.txt.
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy import stats
rcParams.update({"font.size":9,"axes.labelsize":9,"axes.titlesize":9.5,
 "legend.fontsize":8,"xtick.labelsize":8,"ytick.labelsize":8,
 "axes.linewidth":0.8,"lines.linewidth":1.4,"savefig.dpi":300,"font.family":"serif",
 "pdf.fonttype":42,"ps.fonttype":42})   # embed TrueType, not Type 3
NAIVE_C="#c0392b"; SIM_C="#2471a3"; GREY="#7f8c8d"
tc=float(stats.t.ppf(0.975,4))
def gc(d, ks):
    g,lo,hi=[],[],[]
    for k in ks:
        lg=np.log(np.array(d[k])); m=lg.mean(); se=lg.std(ddof=1)/np.sqrt(5)
        g.append(np.exp(m)); lo.append(np.exp(m)-np.exp(m-tc*se)); hi.append(np.exp(m+tc*se)-np.exp(m))
    return np.array(g), np.array([lo,hi])

# compressible MLP scale + config sweeps: per-seed gaps as logged (results/mlp/multiseed_robustness.txt)
mlp={1:[3.21,4.44,5.06,3.75,4.68],2:[0.81,1.47,0.72,1.05,0.95],
     3:[1.13,1.03,1.15,1.03,1.17],5:[0.64,0.97,1.27,1.27,0.99]}
cfg={1:[1.16,1.10,1.09,1.55,1.04],2:[1.03,1.04,0.91,0.81,0.98],
     4:[0.95,0.94,1.02,0.97,1.03],8:[1.07,0.84,1.03,0.98,0.96]}
# Burgers five-seed per-seed gaps (results/burgers/burgers_multiseed.txt)
burg={1:[9.815,9.904,9.027,11.358,10.791],2:[1.328,0.783,0.875,1.331,0.821],
      3:[1.251,1.114,1.092,0.918,0.899],5:[1.015,0.892,1.096,1.230,1.237],
      7:[0.938,0.984,0.846,1.036,0.980]}

fig,(axa,axb)=plt.subplots(1,2,figsize=(6.9,2.7))
Lk=[1,2,3,5]; Cm=[1,2,4,8]; Bk=[1,2,3,5,7]
g1,e1=gc(mlp,Lk); g2,e2=gc(cfg,Cm)
axa.axhline(1.0,color=GREY,ls="--",lw=1.0,label="no advantage")
axa.errorbar(Lk,g1,yerr=e1,marker="o",ms=4,color=SIM_C,capsize=2.5,label="Scale sweep ($k$)")
axa.errorbar(Cm,g2,yerr=e2,marker="^",ms=4,color="#117a65",capsize=2.5,label="Config sweep ($m$)")
axa.set_xlabel("Training coverage  ($k$ scales or $m$ pairs)")
axa.set_ylabel("Gap = naive err / similarity err")
axa.set_title("(a) Compressible: advantage collapses")
axa.grid(alpha=0.25,lw=0.5); axa.legend(frameon=False); axa.set_ylim(0,5.6)

g3,e3=gc(burg,Bk)
axb.axhline(1.0,color=GREY,ls="--",lw=1.0,label="no advantage")
axb.errorbar(Bk,g3,yerr=e3,marker="D",ms=4,color="#884ea0",capsize=2.5,
             label="Burgers (1 invariant)")
axb.errorbar(Lk,g1,yerr=e1,marker="o",ms=4,color=SIM_C,capsize=2.5,
             label="Compressible (2 invariants)")
axb.set_xlabel("Number of training length scales $k$")
axb.set_ylabel("Gap = naive err / similarity err")
axb.set_title("(b) Same collapse across two systems")
axb.set_xticks([1,2,3,5,7]); axb.set_yscale("log")
axb.set_yticks([1,2,5,10]); axb.set_yticklabels(["1","2","5","10"])
axb.grid(alpha=0.25,lw=0.5); axb.legend(frameon=False)
fig.tight_layout()
fig.savefig("fig2_gap_collapse.pdf",bbox_inches="tight")
print("wrote fig2_gap_collapse.pdf")
