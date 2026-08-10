import h5py
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from numpy.polynomial.legendre import leggauss
import sys

t_max = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
threshold = 0.01
N_THETA, N_PHI = 64, 128  # production resolution

# ---- load scan_cache.h5 ----
gammas_grid, times_by_g, w_by_g, k_by_g, amp_re, amp_im = [], [], [], [], [], []
with h5py.File('scan_cache.h5') as f:
    keys = sorted(f.keys(), key=lambda s: float(s))
    for key in keys:
        grp = f[key]
        gammas_grid.append(float(grp.attrs['gamma']))
        times_by_g.append(grp['times'][:])
        w_by_g.append(grp['w'][:])
        k_by_g.append(grp['k'][:])
        amp_re.append(grp['amp_re'][:])
        amp_im.append(grp['amp_im'][:])
gammas_grid = np.array(gammas_grid)
scan_times = times_by_g[0]
w_amp = w_by_g[0]   # resampling target: smallest gamma's own w grid
k_amp = k_by_g[0]
Ng = len(gammas_grid)
Nt, n_w_native, n_k = amp_re[0].shape
n_w = len(w_amp)

# w is gamma-dependent (wmin shrinks with gamma via BubbleMaster's spatial
# grid) -- resample every gamma's amp_re/amp_im onto w_amp before stacking,
# same fix as amplitude_table.cpp / notebook 06's build_amp_interpolator.
amp_re_resampled = np.zeros((Ng, Nt, n_w, n_k))
amp_im_resampled = np.zeros((Ng, Nt, n_w, n_k))
for ig in range(Ng):
    if np.array_equal(w_by_g[ig], w_amp):
        amp_re_resampled[ig] = amp_re[ig]
        amp_im_resampled[ig] = amp_im[ig]
    else:
        for it in range(Nt):
            for ik in range(n_k):
                amp_re_resampled[ig, it, :, ik] = np.interp(
                    w_amp, w_by_g[ig], amp_re[ig][it, :, ik], left=0., right=0.)
                amp_im_resampled[ig, it, :, ik] = np.interp(
                    w_amp, w_by_g[ig], amp_im[ig][it, :, ik], left=0., right=0.)
amp_re = amp_re_resampled
amp_im = amp_im_resampled

flat_shape = (Ng, Nt, n_w * n_k)
_raw_re = RegularGridInterpolator((gammas_grid, scan_times), amp_re.reshape(flat_shape),
                                   bounds_error=False, fill_value=0.)
_raw_im = RegularGridInterpolator((gammas_grid, scan_times), amp_im.reshape(flat_shape),
                                   bounds_error=False, fill_value=0.)

def amp_interp_fn(pts):
    re = _raw_re(pts).reshape(-1, n_w, n_k)
    im = _raw_im(pts).reshape(-1, n_w, n_k)
    return re, im

# ---- load weights_in.h5 / weights_out.h5 ----
with h5py.File('weights_in.h5') as f:
    L = float(f.attrs['L'])
    n_b = int(f.attrs['n_b'])
    weights_times = f['t'][:]
    xlocs, ylocs, zlocs = f['xlocs'][:], f['ylocs'][:], f['zlocs'][:]
positions = np.stack([xlocs, ylocs, zlocs], axis=1)

with h5py.File('weights_out.h5') as f:
    n_pairs = int(f.attrs['n_pairs'])
    n_t = int(f.attrs['n_t'])
    weights = f['weights'][:].reshape(n_pairs, n_t)
    pair_i = f['pair_i'][:].astype(int)
    pair_j = f['pair_j'][:].astype(int)
    gammas = f['gamma'][:]

# ---- active pairs ----
mask_t = weights_times <= t_max
active = [p for p in range(n_pairs) if np.any(weights[p][mask_t] > threshold)]
n_active = len(active)
print(f'n_active={n_active}  t_max={t_max}')

# ---- pair geometry ----
axis = {}
center = {}
for p in active:
    i, j = pair_i[p], pair_j[p]
    dp = positions[i] - positions[j]
    dp -= L * np.round(dp / L)
    axis[p] = dp / np.linalg.norm(dp)
    center[p] = (0.5 * (positions[i] + positions[j])) % L

# ---- sphere quadrature ----
cos_nodes, gl_w = leggauss(N_THETA)
phi_vals = np.linspace(0., 2.*np.pi, N_PHI, endpoint=False)
d_phi = 2.*np.pi / N_PHI
CT, PH = np.meshgrid(cos_nodes, phi_vals, indexing='ij')
ST = np.sqrt(np.maximum(1. - CT**2, 0.))
hat_k = np.stack([ST*np.cos(PH), ST*np.sin(PH), CT], axis=-1).reshape(-1, 3)
d_omega = np.outer(gl_w, np.full(N_PHI, d_phi)).ravel()
N_sph = len(hat_k)

# ---- fine time grid ----
t_hi = min(t_max, scan_times[-1])
mask = (weights_times >= scan_times[0]) & (weights_times <= t_hi)
t_grid = weights_times[mask]
if len(t_grid) == 0 or t_grid[-1] < t_hi:
    t_grid = np.append(t_grid, t_hi)
n_t_fine = len(t_grid)

# ---- naive, unbatched, unoptimized reference: loop everything explicitly ----
P_coh = np.zeros(n_w)
for it in range(n_t_fine - 1):
    t0, t1 = t_grid[it], t_grid[it + 1]

    dA = {}
    for p in active:
        g = np.clip(gammas[p], gammas_grid.min(), gammas_grid.max())
        re0, im0 = amp_interp_fn(np.array([[g, t0]]))
        re1, im1 = amp_interp_fn(np.array([[g, t1]]))
        wgt = max(0.0, np.interp(t0, weights_times, weights[p]))
        dA[p] = (np.sqrt(wgt) * (re1[0] - re0[0]), np.sqrt(wgt) * (im1[0] - im0[0]))

    for s in range(N_sph):
        A_re = np.zeros(n_w)
        A_im = np.zeros(n_w)
        for p in active:
            cth = np.clip(abs(np.dot(hat_k[s], axis[p])), 0., 1.)
            dar_p, dai_p = dA[p]  # (n_w, n_k)
            # interpolate along k for every w
            ar_w = np.array([np.interp(cth, k_amp, dar_p[iw, :]) for iw in range(n_w)])
            ai_w = np.array([np.interp(cth, k_amp, dai_p[iw, :]) for iw in range(n_w)])
            proj = np.dot(hat_k[s], center[p])
            ang = w_amp * proj
            cp, sp = np.cos(ang), np.sin(ang)
            A_re += ar_w * cp - ai_w * sp
            A_im += ar_w * sp + ai_w * cp
        P_coh += d_omega[s] * (A_re**2 + A_im**2)

P_coh *= w_amp**3 * 2. * np.pi

np.savez('python_reference_out.npz', w=w_amp, P_coh=P_coh, t_max=t_max,
         n_active=n_active, N_THETA=N_THETA, N_PHI=N_PHI)
print('P_coh:', P_coh)
print('Saved python_reference_out.npz')
