# Checks that the MLP per-seed gap table (tab:mlpseeds) reproduces the log-gap summary
# statistics in tab:mlppower (geometric-mean gap and 95% CI), compared at 2-decimal rounding.
import numpy as np
from math import exp, sqrt
from scipy import stats
# per-seed gaps as printed in tab:mlpseeds
tbl={("Scale",1):[3.21,4.44,5.06,3.75,4.68],("Scale",2):[0.81,1.47,0.72,1.05,0.95],
     ("Scale",3):[1.13,1.03,1.15,1.03,1.17],("Scale",5):[0.64,0.97,1.27,1.27,0.99],
     ("Config",1):[1.16,1.10,1.09,1.55,1.04],("Config",2):[1.03,1.04,0.91,0.81,0.98],
     ("Config",4):[0.95,0.94,1.02,0.97,1.03],("Config",8):[1.07,0.84,1.03,0.98,0.96]}
# values printed in tab:mlppower
claim={("Scale",1):(4.17,3.32,5.24),("Scale",2):(0.97,0.69,1.36),
       ("Scale",3):(1.10,1.02,1.19),("Scale",5):(1.00,0.70,1.41),
       ("Config",1):(1.18,0.96,1.43),("Config",2):(0.95,0.84,1.08),
       ("Config",4):(0.98,0.93,1.03),("Config",8):(0.97,0.87,1.09)}
tc=float(stats.t.ppf(0.975,4))
print("Does tab:mlpseeds reproduce tab:mlppower?")
ok=True
for key,g in tbl.items():
    lg=np.log(np.array(g)); m,se=lg.mean(),lg.std(ddof=1)/sqrt(5)
    geo,lo,hi=exp(m),exp(m-tc*se),exp(m+tc*se)
    cg,cl,ch=claim[key]
    good = round(geo,2)==cg and round(lo,2)==cl and round(hi,2)==ch
    ok &= good
    print(f"  {key[0]:>6} {key[1]}: geo {geo:.3f} vs {cg:.2f}  CI[{lo:.2f},{hi:.2f}] vs [{cl:.2f},{ch:.2f}]  {'OK' if good else 'MISMATCH'}")
print("\nALL CONSISTENT" if ok else "\n*** INCONSISTENCY ***")
