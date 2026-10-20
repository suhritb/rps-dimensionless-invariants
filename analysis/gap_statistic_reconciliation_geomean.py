# Verifies the gap statistic used throughout: the geometric mean of per-seed ratios
# (vs ratio-of-means or arithmetic mean of ratios).
# Expected: MLP scale 4.17/0.97/1.10/1.00; FNO scale 1.35/1.02/0.93/1.07; MLP config 1.18/0.95/0.98/0.97.

# ================= Part A: three candidate statistics =================
import numpy as np
from math import exp
# MLP per-seed gaps (from results/mlp/multiseed_robustness.txt)
mlp={1:[3.21,4.44,5.06,3.75,4.68],2:[0.81,1.47,0.72,1.05,0.95],
     3:[1.13,1.03,1.15,1.03,1.17],5:[0.64,0.97,1.27,1.27,0.99]}
cfg={1:[1.16,1.10,1.09,1.55,1.04],2:[1.03,1.04,0.91,0.81,0.98],
     4:[0.95,0.94,1.02,0.97,1.03],8:[1.07,0.84,1.03,0.98,0.96]}
mlp_means={1:(0.743,0.189),2:(0.172,0.176),3:(0.141,0.131),5:(0.117,0.119)}
cfg_means={1:(2.111,1.704),2:(1.969,2.155),4:(0.553,0.564),8:(0.339,0.346)}

print("MLP scale sweep")
print(f"{'k':>2} {'ratio-of-means':>15} {'arith mean ratios':>18} {'geo mean ratios':>16}")
for k in mlp:
    n,s=mlp_means[k]; g=np.array(mlp[k])
    print(f"{k:>2} {n/s:>15.3f} {g.mean():>18.3f} {exp(np.log(g).mean()):>16.3f}")
print("\nConfig sweep")
print(f"{'m':>2} {'ratio-of-means':>15} {'arith mean ratios':>18} {'geo mean ratios':>16}")
for m in cfg:
    n,s=cfg_means[m]; g=np.array(cfg[m])
    print(f"{m:>2} {n/s:>15.3f} {g.mean():>18.3f} {exp(np.log(g).mean()):>16.3f}")

# FNO
fno={1:dict(n=[0.7458,0.7466,0.7155,0.7229,0.7942],s=[0.5291,0.5805,0.5475,0.5503,0.5536]),
     2:dict(n=[0.3124,0.3584,0.3285,0.4039,0.2972],s=[0.3685,0.3600,0.2990,0.3298,0.3124]),
     3:dict(n=[0.4848,0.3425,0.3670,0.3258,0.2687],s=[0.5701,0.3100,0.3280,0.4210,0.3112]),
     5:dict(n=[0.2178,0.1959,0.1854,0.2711,0.1975],s=[0.1966,0.1947,0.1570,0.2715,0.1890])}
print("\nFNO scale sweep")
print(f"{'k':>2} {'ratio-of-means':>15} {'geo mean ratios':>16}")
for k,v in fno.items():
    a=np.array(v['n']); b=np.array(v['s']); g=a/b
    print(f"{k:>2} {a.mean()/b.mean():>15.3f} {exp(np.log(g).mean()):>16.3f}")

# ================= Part B: geometric means (the reported gap statistic) =================
import numpy as np
from math import exp
mlp={1:[3.21,4.44,5.06,3.75,4.68],2:[0.81,1.47,0.72,1.05,0.95],
     3:[1.13,1.03,1.15,1.03,1.17],5:[0.64,0.97,1.27,1.27,0.99]}
cfg={1:[1.16,1.10,1.09,1.55,1.04],2:[1.03,1.04,0.91,0.81,0.98],
     4:[0.95,0.94,1.02,0.97,1.03],8:[1.07,0.84,1.03,0.98,0.96]}
fno={1:([0.7458,0.7466,0.7155,0.7229,0.7942],[0.5291,0.5805,0.5475,0.5503,0.5536]),
     2:([0.3124,0.3584,0.3285,0.4039,0.2972],[0.3685,0.3600,0.2990,0.3298,0.3124]),
     3:([0.4848,0.3425,0.3670,0.3258,0.2687],[0.5701,0.3100,0.3280,0.4210,0.3112]),
     5:([0.2178,0.1959,0.1854,0.2711,0.1975],[0.1966,0.1947,0.1570,0.2715,0.1890])}
gm=lambda g: exp(np.log(np.array(g)).mean())
print("MLP scale sweep:", {k: round(gm(v),2) for k,v in mlp.items()})
print("FNO scale sweep:", {k: round(gm(np.array(a)/np.array(b)),2) for k,(a,b) in fno.items()})
print("Config sweep   :", {m: round(gm(v),2) for m,v in cfg.items()})
print("\nThese geometric means should match the per-seed tables exactly.")
