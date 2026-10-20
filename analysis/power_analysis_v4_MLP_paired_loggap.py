# Paired analysis of the per-seed MLP gaps from results/mlp/multiseed_robustness.txt.
# Part B (log-gap t-test) is the version reported in the paper.
# Expected: scale k=1 4.17x, k=2 0.97x, k=3 1.10x (p=0.026; Bonferroni x3 p=0.076), k=5 1.00x;
# config m=1 1.18x, m=2 0.95x, m=4 0.98x, m=8 0.97x. Pooled MDE: scale k>=2 1.46x, config m>=2 1.15x.
# Note: MDEs describe design sensitivity, not bounds on the observed effect; see
# stats_checks_paired_pvalues_and_CI_upper_limits.py for CI upper limits.

# ================= Part A (raw gaps, H0 gap=1) =================
import numpy as np
from math import sqrt, log
from scipy import stats

# Per-seed MLP gaps from results/mlp/multiseed_robustness.txt
mlp_gaps = {
 1:[3.21,4.44,5.06,3.75,4.68],
 2:[0.81,1.47,0.72,1.05,0.95],
 3:[1.13,1.03,1.15,1.03,1.17],
 5:[0.64,0.97,1.27,1.27,0.99],
}
cfg_gaps = {
 1:[1.16,1.10,1.09,1.55,1.04],
 2:[1.03,1.04,0.91,0.81,0.98],
 4:[0.95,0.94,1.02,0.97,1.03],
 8:[1.07,0.84,1.03,0.98,0.96],
}
n=5; df=n-1
tc=float(stats.t.ppf(0.975,df)); tp=float(stats.t.ppf(0.80,df))

def analyse(gaps,label):
    print(f"\n=== {label} : paired analysis on per-seed GAPS (H0: gap=1) ===")
    print(f"{'k':>3} {'mean gap':>9} {'95% CI':>20} {'p':>7} {'MDE(gap)':>9}")
    for k,g in gaps.items():
        g=np.array(g)
        m,sd=g.mean(),g.std(ddof=1); se=sd/sqrt(n)
        lo,hi=m-tc*se,m+tc*se
        t,p=stats.ttest_1samp(g,1.0)
        mde=1+(tc+tp)*se
        print(f"{k:>3} {m:>9.3f} [{lo:>8.3f},{hi:>8.3f}] {p:>7.3f} {mde:>8.2f}x")
    # pooled k>=2
    keys=[k for k in gaps if k!=list(gaps)[0]]
    allg=np.concatenate([np.array(gaps[k]) for k in keys])
    sds=[np.array(gaps[k]).std(ddof=1) for k in keys]
    pooled=sqrt(np.mean(np.array(sds)**2)); se=pooled/sqrt(n)
    print(f"  pooled(k>=2): SD={pooled:.4f} SE={se:.4f} -> MDE gap = {1+(tc+tp)*se:.2f}x  (mean gap {allg.mean():.3f})")

analyse(mlp_gaps,"MLP scale sweep")
analyse(cfg_gaps,"MLP config sweep")

# ================= Part B (log-gaps; reported in the paper) =================
import numpy as np
from math import sqrt, exp
from scipy import stats
mlp={1:[3.21,4.44,5.06,3.75,4.68],2:[0.81,1.47,0.72,1.05,0.95],
     3:[1.13,1.03,1.15,1.03,1.17],5:[0.64,0.97,1.27,1.27,0.99]}
cfg={1:[1.16,1.10,1.09,1.55,1.04],2:[1.03,1.04,0.91,0.81,0.98],
     4:[0.95,0.94,1.02,0.97,1.03],8:[1.07,0.84,1.03,0.98,0.96]}
n=5; df=4; tc=float(stats.t.ppf(0.975,df)); tp=float(stats.t.ppf(0.80,df))

def logan(g,label):
    print(f"\n{label} — log-gap analysis (H0: log gap = 0)")
    print(f"{'k':>3} {'geo-mean':>9} {'95% CI (gap)':>22} {'p':>7} {'MDE':>7}")
    for k,v in g.items():
        lg=np.log(np.array(v)); m,sd=lg.mean(),lg.std(ddof=1); se=sd/sqrt(n)
        t,p=stats.ttest_1samp(lg,0.0)
        lo,hi=exp(m-tc*se),exp(m+tc*se)
        mde=exp((tc+tp)*se)
        star=" *" if p<0.05 else ""
        print(f"{k:>3} {exp(m):>9.3f} [{lo:>8.3f},{hi:>8.3f}] {p:>7.3f} {mde:>6.2f}x{star}")
    keys=list(g)[1:]
    sds=[np.log(np.array(g[k])).std(ddof=1) for k in keys]
    pooled=sqrt(np.mean(np.array(sds)**2)); se=pooled/sqrt(n)
    print(f"  pooled(k>=2): MDE = {exp((tc+tp)*se):.2f}x")
logan(mlp,"MLP SCALE"); logan(cfg,"MLP CONFIG")

# Bonferroni note for the three k>=2 comparisons
lg=np.log(np.array(mlp[3])); t,p=stats.ttest_1samp(lg,0.0)
print(f"\nMLP k=3 log-gap p={p:.4f}; Bonferroni x3 = {min(1,p*3):.3f}")

# ================= Part C (table values) =================
import numpy as np
from math import sqrt, exp
from scipy import stats
mlp={1:[3.21,4.44,5.06,3.75,4.68],2:[0.81,1.47,0.72,1.05,0.95],
     3:[1.13,1.03,1.15,1.03,1.17],5:[0.64,0.97,1.27,1.27,0.99]}
cfg={1:[1.16,1.10,1.09,1.55,1.04],2:[1.03,1.04,0.91,0.81,0.98],
     4:[0.95,0.94,1.02,0.97,1.03],8:[1.07,0.84,1.03,0.98,0.96]}
n=5;df=4;tc=float(stats.t.ppf(0.975,df));tp=float(stats.t.ppf(0.80,df))
print("Table values (geometric-mean gap, 95% CI, p, MDE):")
for name,g in [("Scale",mlp),("Config",cfg)]:
    for k,v in g.items():
        lg=np.log(np.array(v)); m,sd=lg.mean(),lg.std(ddof=1); se=sd/sqrt(n)
        t,p=stats.ttest_1samp(lg,0.0)
        print(f"  {name} {k}: geo={exp(m):.2f} CI[{exp(m-tc*se):.2f},{exp(m+tc*se):.2f}] p={p:.3f} MDE={exp((tc+tp)*se):.2f}x")
for name,g in [("Scale",mlp),("Config",cfg)]:
    keys=list(g)[1:]
    sds=[np.log(np.array(g[k])).std(ddof=1) for k in keys]
    se=sqrt(np.mean(np.array(sds)**2))/sqrt(n)
    print(f"  {name} pooled MDE = {exp((tc+tp)*se):.2f}x")
