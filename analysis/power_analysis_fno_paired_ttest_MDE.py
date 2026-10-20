# FNO scale sweep: paired per-seed t-test, 95% CI and minimum detectable effect (MDE at
# ~80% power) on the per-seed errors in results/fno/fno_compressible_lsweep.json.
# Note: the pooled k>=2 bound is superseded by power_analysis_v2_pooledSD_fix_and_MLP_unpaired_bound.py
# (pooled SD = sqrt(mean variance), giving ~1.25x).
import numpy as np
from math import sqrt
from scipy import stats

fno = {
 1: dict(naive=[0.7458,0.7466,0.7155,0.7229,0.7942], sim=[0.5291,0.5805,0.5475,0.5503,0.5536]),
 2: dict(naive=[0.3124,0.3584,0.3285,0.4039,0.2972], sim=[0.3685,0.3600,0.2990,0.3298,0.3124]),
 3: dict(naive=[0.4848,0.3425,0.3670,0.3258,0.2687], sim=[0.5701,0.3100,0.3280,0.4210,0.3112]),
 5: dict(naive=[0.2178,0.1959,0.1854,0.2711,0.1975], sim=[0.1966,0.1947,0.1570,0.2715,0.1890]),
}
n=5; df=n-1
tcrit=float(stats.t.ppf(0.975,df)); tpow=float(stats.t.ppf(0.80,df))
print(f"paired t-test, n={n}, df={df}; MDE=(t.975+t.80)*SE = ({tcrit:.3f}+{tpow:.3f})*SE\n")
print(f"{'k':>2} {'diff':>8} {'95% CI':>20} {'p':>7} {'MDE':>7} {'as gap':>8}")
rows=[]
for k,v in fno.items():
    a=np.array(v['naive']); b=np.array(v['sim']); d=a-b
    md=d.mean(); sd=d.std(ddof=1); se=sd/sqrt(n)
    lo,hi=md-tcrit*se, md+tcrit*se
    t,p=stats.ttest_rel(a,b)
    mde=(tcrit+tpow)*se
    gapmde=(b.mean()+mde)/b.mean()
    rows.append((k,md,lo,hi,p,mde,gapmde))
    print(f"{k:>2} {md:>8.4f} [{lo:>7.4f},{hi:>7.4f}] {p:>7.3f} {mde:>7.4f} {gapmde:>7.2f}x")

print("\npooled k>=2:")
ds=[np.array(fno[k]['naive'])-np.array(fno[k]['sim']) for k in [2,3,5]]
sdp=np.sqrt(np.mean([d.var(ddof=1) for d in ds])); sep=sdp/sqrt(n)   # pooled SD (root mean square)
mde=(tcrit+tpow)*sep
msim=np.mean([np.mean(fno[k]['sim']) for k in [2,3,5]])
print(f"  pooled SD(diff)={sdp:.4f}  SE={sep:.4f}")
print(f"  MDE(abs)={mde:.4f}   mean sim err={msim:.4f}")
print(f"  smallest reliably detectable gap at k>=2 = {(msim+mde)/msim:.2f}x")
