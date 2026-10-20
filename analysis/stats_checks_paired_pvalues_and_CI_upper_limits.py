# Part 1: paired log-ratio t-tests of naive-mix and naive-mu against naive-log at k = 2, 3, 5
# (naive-mix p = 0.0073, 0.0097, 0.0028, i.e. p < 0.01; naive-mu p < 0.005), and the MLP
# naive-raw / naive-log paired ratio at k=2 (3.19x).
# Part 2: largest 95% CI upper limit at k>=2: MLP scale 1.41x, FNO scale 1.21x, MLP config 1.09x.
#
# Run from the project root:
#     python analysis/stats_checks_paired_pvalues_and_CI_upper_limits.py
# ---------------- Part 1 ----------------
import numpy as np
from math import exp, sqrt
from scipy import stats
print("=== Paired log-ratio tests: naive-mix / naive-mu vs naive-log ===")
log={2:[0.81,1.47,0.72,1.05,0.95],3:[1.13,1.03,1.15,1.03,1.17],5:[0.64,0.97,1.27,1.27,0.99]}
mix={2:[1.62,3.07,2.29,1.43,3.08],3:[2.24,2.78,3.04,1.22,3.53],5:[1.07,2.82,3.12,2.14,2.85]}
mu ={2:[2.52,3.56,2.64,2.15,3.78],3:[2.96,3.54,2.86,1.85,3.67],5:[1.29,2.60,2.47,1.71,2.53]}
for nm,arm in [("mix",mix),("mu",mu)]:
    for k in [2,3,5]:
        d=np.log(np.array(arm[k]))-np.log(np.array(log[k]))
        t,p=stats.ttest_1samp(d,0.0)
        print(f"  {nm} vs log k={k}: p={p:.4f}")
print()
print("=== MLP naive-raw / naive-log paired ratio at k=2 ===")
raw2=[2.53,3.97,2.81,2.39,4.21]
d2=np.log(np.array(raw2))-np.log(np.array(log[2]))
print(f"     MLP raw/log k=2 = {exp(d2.mean()):.2f}x  (geometric mean of per-seed ratios)")

# ---------------- Part 2 ----------------
print()
print("=== Largest 95% CI upper limit at k>=2 ===")
tc=float(stats.t.ppf(0.975,4))
mlp={2:[0.81,1.47,0.72,1.05,0.95],3:[1.13,1.03,1.15,1.03,1.17],5:[0.64,0.97,1.27,1.27,0.99]}
cfg={2:[1.03,1.04,0.91,0.81,0.98],4:[0.95,0.94,1.02,0.97,1.03],8:[1.07,0.84,1.03,0.98,0.96]}
fn={2:[0.3124,0.3584,0.3285,0.4039,0.2972],3:[0.4848,0.3425,0.3670,0.3258,0.2687],5:[0.2178,0.1959,0.1854,0.2711,0.1975]}
fs={2:[0.3685,0.3600,0.2990,0.3298,0.3124],3:[0.5701,0.3100,0.3280,0.4210,0.3112],5:[0.1966,0.1947,0.1570,0.2715,0.1890]}
fno={k:list(np.array(fn[k])/np.array(fs[k])) for k in [2,3,5]}
for nm,d in [("MLP scale",mlp),("FNO scale",fno),("MLP config",cfg)]:
    ups=[]
    for k,v in d.items():
        lg=np.log(np.array(v)); m,se=lg.mean(),lg.std(ddof=1)/sqrt(5)
        ups.append(exp(m+tc*se))
    print(f"  {nm:12} max CI upper at k>=2: {max(ups):.2f}x")
