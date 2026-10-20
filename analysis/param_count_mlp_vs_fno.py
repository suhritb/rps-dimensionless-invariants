# Parameter counts for the MLP (64 + n_cond -> 128 -> 128 -> 64) and FNO1d
# (width 32, modes 16, depth 4) backbones.
# Parameter counts for the two backbones as actually configured
# MLP: input 64 + n_cond -> 128 -> 128 -> 64
def mlp(n_cond):
    i = 64 + n_cond
    return i*128+128 + 128*128+128 + 128*64+64

# FNO: width=32, modes=16, depth=4
def fno(n_cond, width=32, modes=16, depth=4):
    # SpectralConv1d weight: (in,out,modes) complex -> 2 real params each
    spec = width*width*modes*2
    lin  = width*width + width          # Conv1d(w,w,1)
    block = spec + lin
    lift = (1 + 1 + n_cond)*width + width   # pos + ic + cond -> width
    proj = width*64 + 64 + 64*1 + 1
    return depth*block + lift + proj

for name, nc in [("naive",3), ("similarity",2)]:
    print(f"{name:>11}:  MLP {mlp(nc):>8,}   FNO {fno(nc):>9,}   ratio {fno(nc)/mlp(nc):.1f}x")
