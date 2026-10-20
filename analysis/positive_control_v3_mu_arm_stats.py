# Statistics for the positive control's naive-mu arm (5 seeds).
# Expected: pooled k>=2 2.57x, 95% CI [2.17, 3.04].
import numpy as np
from math import exp, sqrt
from scipy import stats
arms={
'log':{1:[3.21,4.44,5.06,3.75,4.68],2:[0.81,1.47,0.72,1.05,0.95],3:[1.13,1.03,1.15,1.03,1.17],5:[0.64,0.97,1.27,1.27,0.99]},
'mix':{1:[3.63,4.80,5.47,4.59,4.98],2:[1.62,3.07,2.29,1.43,3.08],3:[2.24,2.78,3.04,1.22,3.53],5:[1.07,2.82,3.12,2.14,2.85]},
'mu' :{1:[5.20,5.59,6.18,4.15,6.18],2:[2.52,3.56,2.64,2.15,3.78],3:[2.96,3.54,2.86,1.85,3.67],5:[1.29,2.60,2.47,1.71,2.53]},
'raw':{1:[5.59,6.47,6.70,4.73,6.30],2:[2.53,3.97,2.81,2.39,4.21],3:[3.08,3.84,3.46,2.02,3.97],5:[1.14,2.52,2.83,1.95,2.56]}}
n=5;df=4;tc=float(stats.t.ppf(0.975,df))
print("naive-mu vs H0: gap=1 (log-gap t-test)")
for k,v in arms['mu'].items():
    lg=np.log(np.array(v)); m,se=lg.mean(),lg.std(ddof=1)/sqrt(n)
    t,p=stats.ttest_1samp(lg,0.0)
    print(f"  k={k}: geo={exp(m):.2f}x CI[{exp(m-tc*se):.2f},{exp(m+tc*se):.2f}] p={p:.4f}")
print("\npaired mu vs log (same seed):")
for k in arms['log']:
    d=np.log(np.array(arms['mu'][k]))-np.log(np.array(arms['log'][k]))
    t,p=stats.ttest_1samp(d,0.0); m,se=d.mean(),d.std(ddof=1)/sqrt(n)
    print(f"  k={k}: mu/log={exp(m):.2f}x CI[{exp(m-tc*se):.2f},{exp(m+tc*se):.2f}] p={p:.4f}")
print("\npooled k>=2 (geometric):")
for nm in ['log','mix','mu','raw']:
    a=np.concatenate([np.log(np.array(arms[nm][k])) for k in [2,3,5]])
    se=a.std(ddof=1)/sqrt(15)
    print(f"  {nm}: {exp(a.mean()):.3f}x  CI[{exp(a.mean()-2.145*se):.2f},{exp(a.mean()+2.145*se):.2f}]")
print("\nmu vs mix (do they differ?):")
d=np.concatenate([np.log(np.array(arms['mu'][k]))-np.log(np.array(arms['mix'][k])) for k in [2,3,5]])
t,p=stats.ttest_1samp(d,0.0)
print(f"  mu/mix = {exp(d.mean()):.2f}x  p={p:.3f}")
