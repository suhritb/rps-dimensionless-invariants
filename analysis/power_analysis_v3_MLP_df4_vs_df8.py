# Unpaired MLP bound: compares df=4 and df=8 (df=8 used; MDE 0.184, bound ~2.3x).
import numpy as np
from math import sqrt
from scipy import stats
mlp = {2:dict(n=(0.172,0.104), s=(0.176,0.101)),
       3:dict(n=(0.141,0.093), s=(0.131,0.092)),
       5:dict(n=(0.117,0.078), s=(0.119,0.074))}
for df,label in [(4,"df=4 (paired df; not appropriate here)"),(8,"df=8 (unpaired, n1+n2-2)")]:
    tc=float(stats.t.ppf(0.975,df)); tp=float(stats.t.ppf(0.80,df))
    ses=[]; sims=[]
    for k,v in mlp.items():
        se=sqrt(v['n'][1]**2/5 + v['s'][1]**2/5); ses.append(se); sims.append(v['s'][0])
    sep=sqrt(np.mean(np.array(ses)**2)); mde=(tc+tp)*sep; msim=np.mean(sims)
    print(f"{label}")
    print(f"   t.975={tc:.3f} t.80={tp:.3f}  SE={sep:.4f}  MDE={mde:.4f}  bound={(msim+mde)/msim:.2f}x\n")
