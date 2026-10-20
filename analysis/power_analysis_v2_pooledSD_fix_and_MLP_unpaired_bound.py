# Pooled k>=2 MDE for the FNO scale sweep with pooled SD = sqrt(mean within-k variance)
# (bound ~1.25x), and an unpaired bound for the MLP from per-arm means and SDs.

# ================= Part A: pooled SD (FNO, k>=2) =================
import numpy as np
from math import sqrt
from scipy import stats

fno = {
 2: dict(naive=[0.3124,0.3584,0.3285,0.4039,0.2972], sim=[0.3685,0.3600,0.2990,0.3298,0.3124]),
 3: dict(naive=[0.4848,0.3425,0.3670,0.3258,0.2687], sim=[0.5701,0.3100,0.3280,0.4210,0.3112]),
 5: dict(naive=[0.2178,0.1959,0.1854,0.2711,0.1975], sim=[0.1966,0.1947,0.1570,0.2715,0.1890]),
}
ds = {k: np.array(v['naive'])-np.array(v['sim']) for k,v in fno.items()}

print("within-k SD of paired difference (ddof=1):")
for k,d in ds.items(): print(f"  k={k}: sd={d.std(ddof=1):.5f}  var={d.var(ddof=1):.6f}")

sds = np.array([d.std(ddof=1) for d in ds.values()])
vars_ = np.array([d.var(ddof=1) for d in ds.values()])

mean_of_sds = sds.mean()                       # mean of SDs (not a pooled SD)
pooled_sd   = sqrt(vars_.mean())               # correct pooled (equal n)
raw_all     = np.concatenate(list(ds.values())).std(ddof=1)  # ignoring k grouping

print(f"\n  mean of SDs (not a pooled SD): {mean_of_sds:.5f}")
print(f"  pooled SD sqrt(mean var):      {pooled_sd:.5f}")
print(f"  raw SD over all 15 diffs:      {raw_all:.5f}")

df=4
tcrit=float(stats.t.ppf(0.975,df)); tpow=float(stats.t.ppf(0.80,df))
msim=np.mean([np.mean(v['sim']) for v in fno.values()])
print(f"\n  mean similarity error (k>=2) = {msim:.4f}")
for label, sd in [("mean of SDs",mean_of_sds),("pooled SD",pooled_sd),("raw all-15",raw_all)]:
    se=sd/sqrt(5); mde=(tcrit+tpow)*se
    print(f"  {label:>22}: SE={se:.5f}  MDE={mde:.5f}  bound={(msim+mde)/msim:.3f}x")

# ================= Part B: MLP unpaired bound from reported mean/std =================
import numpy as np
from math import sqrt
from scipy import stats
# MLP: only mean/std available (no per-seed), so bound is unpaired & approximate
mlp = {1:dict(n=(0.743,0.402), s=(0.189,0.120)),
       2:dict(n=(0.172,0.104), s=(0.176,0.101)),
       3:dict(n=(0.141,0.093), s=(0.131,0.092)),
       5:dict(n=(0.117,0.078), s=(0.119,0.074))}
df=4; tcrit=float(stats.t.ppf(0.975,df)); tpow=float(stats.t.ppf(0.80,df))
print("MLP (unpaired, from reported mean/std, n=5):")
vs=[]; sims=[]
for k in [2,3,5]:
    sn,ss = mlp[k]['n'][1], mlp[k]['s'][1]
    se_diff = sqrt(sn**2/5 + ss**2/5)     # unpaired SE of difference
    vs.append(se_diff); sims.append(mlp[k]['s'][0])
    mde=(tcrit+tpow)*se_diff
    print(f"  k={k}: SE(diff)={se_diff:.4f}  MDE={mde:.4f}  bound={(mlp[k]['s'][0]+mde)/mlp[k]['s'][0]:.2f}x")
se_p=np.sqrt(np.mean(np.array(vs)**2)); mde=(tcrit+tpow)*se_p; msim=np.mean(sims)
print(f"  pooled k>=2: SE={se_p:.4f}  MDE={mde:.4f}  mean sim={msim:.4f}  bound={(msim+mde)/msim:.2f}x")

# ================= Part C: re-derivation of the FNO paired-test table =================
import numpy as np
from math import sqrt
from scipy import stats
fno = {
 1: dict(naive=[0.7458,0.7466,0.7155,0.7229,0.7942], sim=[0.5291,0.5805,0.5475,0.5503,0.5536]),
 2: dict(naive=[0.3124,0.3584,0.3285,0.4039,0.2972], sim=[0.3685,0.3600,0.2990,0.3298,0.3124]),
 3: dict(naive=[0.4848,0.3425,0.3670,0.3258,0.2687], sim=[0.5701,0.3100,0.3280,0.4210,0.3112]),
 5: dict(naive=[0.2178,0.1959,0.1854,0.2711,0.1975], sim=[0.1966,0.1947,0.1570,0.2715,0.1890]),
}
df=4; tc=float(stats.t.ppf(0.975,df)); tp=float(stats.t.ppf(0.80,df)); n=5
print("FNO paired-test table re-derivation:")
for k,v in fno.items():
    a=np.array(v['naive']); b=np.array(v['sim']); d=a-b
    md,sd=d.mean(),d.std(ddof=1); se=sd/sqrt(n)
    t,p=stats.ttest_rel(a,b); mde=(tc+tp)*se
    print(f"  k={k}: diff={md:+.4f} CI[{md-tc*se:+.4f},{md+tc*se:+.4f}] p={p:.3f} MDE={mde:.4f} gap={(b.mean()+mde)/b.mean():.2f}x")
ds=[np.array(fno[k]['naive'])-np.array(fno[k]['sim']) for k in [2,3,5]]
sds=[d.std(ddof=1) for d in ds]
pooled=sqrt(np.mean([d.var(ddof=1) for d in ds])); se=pooled/sqrt(n); mde=(tc+tp)*se
msim=np.mean([np.mean(fno[k]['sim']) for k in [2,3,5]])
print(f"\n  within-k SDs: {[round(float(s),3) for s in sds]}")
print(f"  pooled SD={pooled:.3f}  SE={se:.3f}  MDE={mde:.3f}  mean sim={msim:.3f}  bound={(msim+mde)/msim:.2f}x")
