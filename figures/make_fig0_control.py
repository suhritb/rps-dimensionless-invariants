# Paper Figure 1 (fig0_control.pdf): positive control. Gap (naive / similarity) vs number of
# training length scales k for the four naive feature maps (naive-log, naive-mix, naive-mu,
# naive-raw). Geometric mean of per-seed gaps +/- 95% CI (Student t, df=4) on log-gaps.
# Per-seed gaps are taken from results/mlp/positive_control.txt.
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy import stats
rcParams.update({"font.size":7,"axes.labelsize":7,"legend.fontsize":6,
 "xtick.labelsize":6.5,"ytick.labelsize":6.5,"axes.linewidth":0.7,
 "lines.linewidth":1.2,"savefig.dpi":300,"font.family":"serif",
 "pdf.fonttype":42,"ps.fonttype":42})   # embed TrueType, not Type 3
arms={
'log':{1:[3.21,4.44,5.06,3.75,4.68],2:[0.81,1.47,0.72,1.05,0.95],3:[1.13,1.03,1.15,1.03,1.17],5:[0.64,0.97,1.27,1.27,0.99]},
'mix':{1:[3.63,4.80,5.47,4.59,4.98],2:[1.62,3.07,2.29,1.43,3.08],3:[2.24,2.78,3.04,1.22,3.53],5:[1.07,2.82,3.12,2.14,2.85]},
'mu' :{1:[5.20,5.59,6.18,4.15,6.18],2:[2.52,3.56,2.64,2.15,3.78],3:[2.96,3.54,2.86,1.85,3.67],5:[1.29,2.60,2.47,1.71,2.53]},
'raw':{1:[5.59,6.47,6.70,4.73,6.30],2:[2.53,3.97,2.81,2.39,4.21],3:[3.08,3.84,3.46,2.02,3.97],5:[1.14,2.52,2.83,1.95,2.56]}}
k=np.array([1,2,3,5]); tc=float(stats.t.ppf(0.975,4))
def gc(d):
    g,lo,hi=[],[],[]
    for kk in [1,2,3,5]:
        lg=np.log(np.array(d[kk])); m=lg.mean(); se=lg.std(ddof=1)/np.sqrt(5)
        g.append(np.exp(m)); lo.append(np.exp(m)-np.exp(m-tc*se)); hi.append(np.exp(m+tc*se)-np.exp(m))
    return np.array(g),np.array([lo,hi])
style=[('raw',"#c0392b","s","naive-raw"),('mu',"#b9770e","D",r"naive-$\mu$"),
       ('mix',"#117a65","^","naive-mix"),('log',"#2471a3","o","naive-log")]
fig,ax=plt.subplots(figsize=(4.2,1.7))
ax.axhline(1.0,color="#7f8c8d",ls="--",lw=1.0,zorder=1,label="no advantage")
for nm,c,mk,lab in style:
    g,e=gc(arms[nm]); ax.errorbar(k,g,yerr=e,marker=mk,ms=3.5,color=c,capsize=2,label=lab)
ax.set_xlabel("Training length scales $k$")
ax.set_ylabel("Gap (naive / sim.)")
ax.set_xticks([1,2,3,5]); ax.set_yscale("log")
ax.set_yticks([1,2,3,5,7]); ax.set_yticklabels(["1","2","3","5","7"])
ax.set_xlim(0.7,5.9)
ax.grid(alpha=0.22,lw=0.5)
h,l=ax.get_legend_handles_labels()
order=[1,2,3,4,0]
ax.legend([h[i] for i in order],[l[i] for i in order],frameon=False,loc="upper right",
          ncol=2,fontsize=6,handlelength=1.4,columnspacing=0.9)
from matplotlib.ticker import NullFormatter
ax.yaxis.set_minor_formatter(NullFormatter())  # suppress minor-tick labels (newer matplotlib adds them on log axes)
fig.tight_layout(pad=0.25); fig.savefig("fig0_control.pdf",bbox_inches="tight")
print("ok")
