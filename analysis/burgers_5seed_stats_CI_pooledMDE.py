# Burgers 5-seed sweep: per-k one-sample t-test on log-gaps (df=4), 95% CI, and pooled-SD
# minimum detectable effect (MDE).
# Per-seed gaps: results/burgers/burgers_multiseed.txt (three decimals), as in paper Table 3.
import numpy as np
from math import exp, sqrt
from scipy import stats
g={1:[9.815,9.904,9.027,11.358,10.791],2:[1.328,0.783,0.875,1.331,0.821],
   3:[1.251,1.114,1.092,0.918,0.899],5:[1.015,0.892,1.096,1.230,1.237],
   7:[0.938,0.984,0.846,1.036,0.980]}
n=5;df=4;tc=float(stats.t.ppf(0.975,df));tp=float(stats.t.ppf(0.80,df))
print("BURGERS 5-seed, log-gap test vs H0: gap=1")
print(f"{'k':>3} {'geo':>7} {'95% CI':>18} {'p':>8}")
for k,v in g.items():
    lg=np.log(np.array(v)); m,se=lg.mean(),lg.std(ddof=1)/sqrt(n)
    t,p=stats.ttest_1samp(lg,0.0)
    print(f"{k:>3} {exp(m):>7.2f} [{exp(m-tc*se):>6.2f},{exp(m+tc*se):>6.2f}] {p:>8.4f}")
allg=np.concatenate([np.log(np.array(g[k])) for k in [2,3,5,7]])
sds=[np.log(np.array(g[k])).std(ddof=1) for k in [2,3,5,7]]
pooled=sqrt(np.mean(np.array(sds)**2)); se=pooled/sqrt(n)
print(f"\npooled k>=2: geo={exp(allg.mean()):.3f}x  pooled SD={pooled:.4f}  "
      f"MDE={exp((tc+tp)*se):.2f}x")
print(f"k=1 across seeds: {min(g[1]):.2f}-{max(g[1]):.2f}x  (tight, no seed near 1)")
