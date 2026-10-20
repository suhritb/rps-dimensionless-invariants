# (a) Counts how many test (L, Re) combinations place mu = L/Re outside the training mu range
#     at each k (result: 2-3 of 10 at every k, so the naive-mix and naive-mu confounds are
#     largely, not fully, disjoint).
# (b) Analytic parameter counts for an FNO and a DeepONet configuration (reference only).
import numpy as np
Re=np.array([500.,1000.,2000.,5000.,10000.])
Lpool=np.array([0.8,1.2,1.6,2.0,2.4]); Ltest=np.array([0.4,3.0])
print("=== Is mu extrapolated? ===")
for k in [1,2,3,5]:
    tr=np.array([L/r for L in Lpool[:k] for r in Re])
    te=np.array([L/r for L in Ltest for r in Re])
    out=((te<tr.min())|(te>tr.max())).sum()
    print(f"  k={k}: train mu [{tr.min():.2e},{tr.max():.2e}]  test [{te.min():.2e},{te.max():.2e}]  outside: {out}/{len(te)}")
print()
print("=== DeepONet vs FNO parameter counts ===")
def fno(w=32,m=16,d=4,nc=3):
    return d*(w*w*m*2 + w*w+w) + (2+nc)*w+w + w*64+64+64+1
def don(p=128,w=256,d=4,nc=3,N=64):
    br=(N+nc)*w+w + (d-2)*(w*w+w) + w*p+p
    tr=1*w+w + (d-2)*(w*w+w) + w*p+p
    return br+tr+N
print(f"  FNO      ~{fno():,}")
print(f"  DeepONet ~{don():,}   ratio {don()/fno():.1f}x")


# ---------------------------------------------------------------------------------------------
# (c) Extrapolation factors for mu vs L are identical (mu = L/Re on a shared Re grid);
#     the difference is coverage. Result: k=5 low 2.00x, high 1.25x for both; k=1 high 3.75x.
import numpy as np
Re=np.array([500.,1000.,2000.,5000.,10000.])
Lpool=np.array([0.8,1.2,1.6,2.0,2.4]); Ltest=np.array([0.4,3.0])
print("=== extrapolation FACTORS: mu vs L ===")
for k in [1,5]:
    trL, teL = Lpool[:k], Ltest
    trm=np.array([L/r for L in trL for r in Re]); tem=np.array([L/r for L in teL for r in Re])
    print(f"  k={k}")
    print(f"    L : train [{trL.min()},{trL.max()}]  test [{teL.min()},{teL.max()}]"
          f"   low {trL.min()/teL.min():.2f}x  high {teL.max()/trL.max():.2f}x")
    print(f"    mu: train [{trm.min():.2e},{trm.max():.2e}]  test [{tem.min():.2e},{tem.max():.2e}]"
          f"   low {trm.min()/tem.min():.2f}x  high {tem.max()/trm.max():.2f}x")
    out=((tem<trm.min())|(tem>trm.max())).sum()
    outL=((teL<trL.min())|(teL>trL.max())).sum()
    print(f"    coverage: L {outL}/{len(teL)} test values outside;  mu {out}/{len(tem)} outside")
print()
print("  -> factors are IDENTICAL (mu = L/Re on a shared Re grid preserves the ratio).")
print("  -> Extrapolation factors are identical; the difference is coverage.")
