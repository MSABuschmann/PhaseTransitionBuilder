import h5py
import numpy as np

rng = np.random.default_rng(42)

# ---- synthetic scan_cache.h5 (matches ptbuilder.scan.ScanResult.to_hdf5) ----
Ng, Nt, n_w, n_k = 5, 6, 4, 7
gammas = np.linspace(1.0, 8.0, Ng)
times  = np.linspace(1.0, 20.0, Nt)
k      = np.linspace(0.0, 1.0, n_k)

# w grid is gamma-dependent, matching real BubbleMaster scans: wmin shrinks
# with gamma (via the spatial grid, see ptbuilder/ic.py's write_2d_setup),
# wmax stays fixed. amp_re/amp_im are generated as smooth functions of w so
# that resampling onto a different gamma's w grid is well-defined/testable
# (not just noise, so a resampling bug shows up as a real numeric mismatch).
wmax = 14.0
w_by_gamma = [np.geomspace(0.5 / g, wmax, n_w) for g in gammas]

def amp_fn(w, t, k, g):
    # arbitrary smooth function of (w, t, k, g) for testing
    return (np.sin(w[:, None] * t + g) * np.cos(3 * k[None, :])
            + 0.1 * w[:, None] * k[None, :])

amp_re_all = []
amp_im_all = []
for ig, g in enumerate(gammas):
    w_g = w_by_gamma[ig]
    re_g = np.zeros((Nt, n_w, n_k))
    im_g = np.zeros((Nt, n_w, n_k))
    for it, t in enumerate(times):
        re_g[it] = amp_fn(w_g, t, k, g)
        im_g[it] = amp_fn(w_g, t, k, g) * 0.5 + 0.05 * np.outer(w_g, k)
    amp_re_all.append(re_g)
    amp_im_all.append(im_g)

with h5py.File('scan_cache.h5', 'w') as f:
    for ig, g in enumerate(gammas):
        grp = f.create_group(str(g))
        grp.attrs['gamma'] = g
        grp.create_dataset('times', data=times)
        grp.create_dataset('w', data=w_by_gamma[ig])
        grp.create_dataset('spectrum', data=np.zeros((Nt, n_w)))
        grp.create_dataset('dEdt', data=np.zeros((Nt, n_w)))
        grp.create_dataset('k', data=k)
        grp.create_dataset('amp_re', data=amp_re_all[ig])
        grp.create_dataset('amp_im', data=amp_im_all[ig])

# ---- synthetic weights_in.h5 / weights_out.h5 (matches cpp/weights format) ----
n_b = 5
L = 200.0
positions = rng.uniform(0, L, size=(n_b, 3))
N_WEIGHTS = 40
t_fine = np.linspace(1.0, 20.0, N_WEIGHTS)

with h5py.File('weights_in.h5', 'w') as f:
    f.attrs['L'] = L
    f.attrs['n_b'] = n_b
    f.attrs['n_t'] = N_WEIGHTS
    f.attrs['rin_0'] = 1.0
    f.attrs['rout_0'] = 2.0
    f.create_dataset('t', data=t_fine)
    f.create_dataset('R', data=np.zeros(N_WEIGHTS))
    f.create_dataset('xlocs', data=positions[:, 0])
    f.create_dataset('ylocs', data=positions[:, 1])
    f.create_dataset('zlocs', data=positions[:, 2])

# a handful of "colliding" pairs with synthetic gammas + weight curves
pairs = [(0, 1), (0, 2), (1, 3), (2, 4)]
n_pairs = len(pairs)
pair_gammas = rng.uniform(1.5, 7.5, n_pairs)
# weight curves: smooth bump, some pairs active only briefly
weights = np.zeros((n_pairs, N_WEIGHTS))
for p in range(n_pairs):
    center = rng.uniform(3.0, 15.0)
    width = rng.uniform(1.0, 4.0)
    weights[p] = np.clip(1.0 - ((t_fine - center) / width) ** 2, 0.0, 1.0)

with h5py.File('weights_out.h5', 'w') as f:
    f.attrs['n_pairs'] = n_pairs
    f.attrs['n_t'] = N_WEIGHTS
    f.create_dataset('weights', data=weights.flatten())
    f.create_dataset('pair_i', data=np.array([p[0] for p in pairs], dtype=float))
    f.create_dataset('pair_j', data=np.array([p[1] for p in pairs], dtype=float))
    f.create_dataset('gamma', data=pair_gammas)

print('Wrote scan_cache.h5, weights_in.h5, weights_out.h5')
print('positions:\n', positions)
print('pairs:', pairs)
print('pair_gammas:', pair_gammas)
