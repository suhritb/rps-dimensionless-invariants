"""
FNO compressible scale sweep (5 seeds): produces results/fno_compressible_lsweep.{txt,json}
(the reported copy is in results/fno/). Same solver, L pool / test L and (Re, Ma) grid as
the MLP scale sweep; ICs from np.random.default_rng(42) (as in
experiments/mlp/run_compressible_lsweep.py), fixed across seeds. Only the backbone changes
(FNO1d, width 32, modes 16, depth 4; 200 epochs, Adam lr 3e-4, batch 64). Originally run on
a single NVIDIA T4 GPU (~15 min).
Seed scope: initial conditions are fixed across seeds; only network initialization and minibatch order vary.
Requires: pip install torch.

Run from the repository root:
    python experiments/fno/run_fno_scale_sweep.py
"""

import sys, os
sys.path.insert(0, os.environ.get("REPO_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))))

import numpy as np, torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# validated compressible solver and dataset assembly (src/)
from src.compressible_dataset import (
    make_ic_velocity, precompute_solutions, assemble_dataset, T_FINAL, CFL,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Torch:", torch.__version__)
print("Device:", device)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")

# ============================================================
# FNO compressible scale sweep — mirrors run_compressible_lsweep.py
# Same solver, same ICs (rng 42), same L_pool/L_test, same (Re,Ma) grid.
# Loss is per-sample relative L2 (the MLP uses an amplitude-weighted
# squared error that approximates it). Only the backbone changes: MLP -> FNO.
# ============================================================

# ---- FNO architecture ----
class SpectralConv1d(nn.Module):
    def __init__(self, in_c, out_c, modes):
        super().__init__()
        self.modes = modes
        scale = 1 / (in_c * out_c)
        self.w = nn.Parameter(scale * torch.rand(in_c, out_c, modes, dtype=torch.cfloat))
    def forward(self, x):
        B, C, N = x.shape
        xf = torch.fft.rfft(x)
        out = torch.zeros(B, self.w.shape[1], N//2+1, dtype=torch.cfloat, device=x.device)
        out[:, :, :self.modes] = torch.einsum("bci,coi->boi", xf[:,:,:self.modes], self.w)
        return torch.fft.irfft(out, n=N)

class FNOBlock(nn.Module):
    def __init__(self, width, modes):
        super().__init__()
        self.spec = SpectralConv1d(width, width, modes)
        self.lin = nn.Conv1d(width, width, 1)
        self.act = nn.GELU()
    def forward(self, x):
        return self.act(self.spec(x) + self.lin(x))

class FNO1d(nn.Module):
    """FNO conditioned on scalar params, broadcast across the grid as channels."""
    def __init__(self, n_scalar_cond, N=64, width=32, modes=16, depth=4):
        super().__init__()
        self.lift = nn.Linear(1 + 1 + n_scalar_cond, width)  # pos + ic + cond
        self.blocks = nn.Sequential(*[FNOBlock(width, modes) for _ in range(depth)])
        self.proj = nn.Sequential(nn.Linear(width, 64), nn.GELU(), nn.Linear(64, 1))
    def forward(self, ic, cond):
        B, N = ic.shape
        pos = torch.linspace(0, 1, N, device=ic.device).unsqueeze(0).expand(B, -1)
        cond_exp = cond.unsqueeze(1).expand(-1, N, -1)
        x = torch.cat([pos.unsqueeze(-1), ic.unsqueeze(-1), cond_exp], dim=-1)
        x = self.lift(x).permute(0, 2, 1)
        x = self.blocks(x)
        x = self.proj(x.permute(0, 2, 1)).squeeze(-1)
        return x + ic   # residual: predict correction to IC

# ---- training / eval (per-sample relative-L2 loss) ----
def train_fno(model, Xic, cond, Y, epochs=200, lr=3e-4, batch=64, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ds = TensorDataset(torch.tensor(Xic, dtype=torch.float32),
                       torch.tensor(cond, dtype=torch.float32),
                       torch.tensor(Y, dtype=torch.float32))
    dl = DataLoader(ds, batch_size=batch, shuffle=True)
    model.to(device).train()
    for ep in range(epochs):
        for xb, cb, yb in dl:
            xb, cb, yb = xb.to(device), cb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb, cb)
            loss = torch.mean(torch.norm(pred-yb, dim=1) / (torch.norm(yb, dim=1) + 1e-8))
            loss.backward(); opt.step()
    return model

def eval_fno(model, Xic, cond, Y):
    model.eval()
    with torch.no_grad():
        xb = torch.tensor(Xic, dtype=torch.float32).to(device)
        cb = torch.tensor(cond, dtype=torch.float32).to(device)
        pred = model(xb, cb).cpu().numpy()
    rel = np.linalg.norm(pred-Y, axis=1) / np.linalg.norm(Y, axis=1)
    return float(rel.mean())

# ---- config: identical to run_compressible_lsweep.py ----
N          = 64
Re_values  = np.array([500., 1000., 2000., 5000., 10000.])
Ma_values  = np.array([0.3, 0.5, 0.7])
n_ic       = 20
L_pool     = np.array([0.8, 1.2, 1.6, 2.0, 2.4])
L_test     = np.array([0.4, 3.0])
ks         = [1, 2, 3, 5]

RNG = np.random.default_rng(42)
print("Generating ICs and precomputing solutions (real compressible solver)...")
u_perts   = [make_ic_velocity(RNG, N) for _ in range(n_ic)]
sol_cache = precompute_solutions(u_perts, Re_values, Ma_values, verbose=True)

# fixed test set (same assemble_dataset as the MLP experiments)
u_ic_te, mu_te, L_te, c0_te, Re_te, Ma_te, Y_te = assemble_dataset(
    u_perts, Re_values, Ma_values, L_test, sol_cache)

# conditioning: naive = [log mu, log L, log c0]; similarity = [log Re, log Ma]
def cond_naive(mu, L, c0): return np.stack([np.log(mu), np.log(L), np.log(c0)], axis=1)
def cond_sim(Re, Ma):      return np.stack([np.log(Re), np.log(Ma)], axis=1)

cN_te = cond_naive(mu_te, L_te, c0_te)
cS_te = cond_sim(Re_te, Ma_te)

# This single-seed pass is kept: it advances the torch RNG state that the 5-seed loop's
# initializations depend on (each model is built before train_fno seeds the RNG).
# GPU runs are not bitwise reproducible.
print("\nFNO compressible scale sweep (1 seed):")
print(f"{'k':>3}  {'naive':>8}  {'similarity':>11}  {'gap':>7}")
print("-"*36)
for k in ks:
    L_train_k = L_pool[:k]
    u_ic_tr, mu_tr, L_tr, c0_tr, Re_tr, Ma_tr, Y_tr = assemble_dataset(
        u_perts, Re_values, Ma_values, L_train_k, sol_cache)

    mN = FNO1d(n_scalar_cond=3, N=N)
    train_fno(mN, u_ic_tr, cond_naive(mu_tr, L_tr, c0_tr), Y_tr, epochs=200, seed=0)
    eN = eval_fno(mN, u_ic_te, cN_te, Y_te)

    mS = FNO1d(n_scalar_cond=2, N=N)
    train_fno(mS, u_ic_tr, cond_sim(Re_tr, Ma_tr), Y_tr, epochs=200, seed=0)
    eS = eval_fno(mS, u_ic_te, cS_te, Y_te)

    print(f"{k:>3}  {eN:>8.4f}  {eS:>11.4f}  {eN/max(eS,1e-9):>6.2f}x")

# ============================================================
# FNO compressible scale sweep — 5 SEEDS (reported numbers)
# Uses the solver cache and helpers defined above.
# Reports mean +/- std of the relative-L2 error across 5 seeds.
# ============================================================
import time

SEEDS = [0, 1, 2, 3, 4]
ks    = [1, 2, 3, 5]

print("FNO compressible scale sweep — 5 seeds")
print(f"{'k':>3}  {'naive (mean+/-std)':>22}  {'similarity (mean+/-std)':>24}  {'gap':>8}")
print("-"*66)

results = {}
t0 = time.time()

for k in ks:
    L_train_k = L_pool[:k]
    u_ic_tr, mu_tr, L_tr, c0_tr, Re_tr, Ma_tr, Y_tr = assemble_dataset(
        u_perts, Re_values, Ma_values, L_train_k, sol_cache)
    cN_tr = cond_naive(mu_tr, L_tr, c0_tr)
    cS_tr = cond_sim(Re_tr, Ma_tr)

    naive_errs, sim_errs = [], []
    for s in SEEDS:
        mN = FNO1d(n_scalar_cond=3, N=N)
        train_fno(mN, u_ic_tr, cN_tr, Y_tr, epochs=200, seed=s)
        naive_errs.append(eval_fno(mN, u_ic_te, cN_te, Y_te))

        mS = FNO1d(n_scalar_cond=2, N=N)
        train_fno(mS, u_ic_tr, cS_tr, Y_tr, epochs=200, seed=s)
        sim_errs.append(eval_fno(mS, u_ic_te, cS_te, Y_te))

    nm, ns = np.mean(naive_errs), np.std(naive_errs)
    sm, ss = np.mean(sim_errs),   np.std(sim_errs)
    gap    = nm / max(sm, 1e-9)
    results[k] = dict(naive_mean=nm, naive_std=ns, sim_mean=sm, sim_std=ss,
                      gap=gap, naive_all=naive_errs, sim_all=sim_errs)
    print(f"{k:>3}  {nm:>8.4f} +/- {ns:<8.4f}  {sm:>8.4f} +/- {ss:<9.4f}  {gap:>6.2f}x")

print("-"*66)
print(f"Total time: {(time.time()-t0)/60:.1f} min")

# ---- summary block ----
print("\n\n=== SUMMARY (FNO, compressible scale sweep, 5 seeds) ===")
print(f"{'k':>3}  {'Naive':>18}  {'Similarity':>18}  {'Gap':>7}")
for k in ks:
    r = results[k]
    print(f"{k:>3}  {r['naive_mean']:>7.3f} +/- {r['naive_std']:<7.3f}  "
          f"{r['sim_mean']:>7.3f} +/- {r['sim_std']:<7.3f}  {r['gap']:>5.2f}x")


# Save FNO results (results/ at the repo root)
import os, json, datetime

out_dir = os.path.join(os.environ.get("REPO_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))), "results")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "fno_compressible_lsweep.txt")

lines = []
lines.append("FNO COMPRESSIBLE SCALE SWEEP — 5 SEEDS")
lines.append("Generated: " + datetime.datetime.now().isoformat(timespec="seconds"))
lines.append(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}. Backbone: FNO1d (width=32, modes=16, depth=4).")
lines.append("Same solver / ICs (rng 42) / L_pool / L_test / (Re,Ma) grid as run_compressible_lsweep.py.")
lines.append("Naive cond = [log mu, log L, log c0]; Similarity cond = [log Re, log Ma].")
lines.append("Loss = per-sample relative L2. Epochs=200, Adam lr=3e-4, batch=64.")
lines.append("")
lines.append(f"{'k':>3}  {'Naive(mean)':>11}  {'Naive(std)':>10}  {'Sim(mean)':>10}  {'Sim(std)':>9}  {'Gap':>6}")
lines.append("-"*62)
for k in ks:
    r = results[k]
    lines.append(f"{k:>3}  {r['naive_mean']:>11.4f}  {r['naive_std']:>10.4f}  "
                 f"{r['sim_mean']:>10.4f}  {r['sim_std']:>9.4f}  {r['gap']:>5.2f}x")
lines.append("")
lines.append("Per-seed raw values (for full reproducibility):")
for k in ks:
    r = results[k]
    lines.append(f"  k={k}  naive={['%.4f'%v for v in r['naive_all']]}")
    lines.append(f"  k={k}  sim  ={['%.4f'%v for v in r['sim_all']]}")

text = "\n".join(lines)
with open(out_path, "w", encoding="utf-8") as f:
    f.write(text)

# also drop a machine-readable JSON copy alongside it
json_path = os.path.join(out_dir, "fno_compressible_lsweep.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump({str(k): {kk: (vv if not isinstance(vv, list) else vv)
                        for kk, vv in results[k].items()} for k in ks}, f, indent=2)

print("Saved:")
print(" ", out_path)
print(" ", json_path)
print("\n--- file contents ---\n")
print(text)

