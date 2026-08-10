"""
Runs the ACTUAL notebook 06 cells (verbatim, not a paraphrase) against the
real local data, to get a trustworthy Python reference for comparing
against the C++ port -- avoids the risk that an independently-written
Python reference happens to share the same misunderstanding as the C++
port (since both were written by reading the same notebook).
"""
import numpy as np
import h5py
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator
from numpy.polynomial.legendre import leggauss

import ptbuilder as pt
from ptbuilder.analysis import load_weights
from ptbuilder.scan import load_scan
from ptbuilder.ic import _cutoff_times

REPO_ROOT = Path(pt.__file__).parent.parent
DATA_DIR = REPO_ROOT / 'data'

# ---- cell-1 (bubbles) ----
bubbles = np.array([[  0.        ,   0.        ,   0.        ],
       [104.21080818, 134.00331875, 115.32260336],
       [ 82.91633086,  40.22040859, 161.10457444],
       [162.20889478, 164.79584079,   2.8384039 ],
       [154.47733984, 187.65793396,  20.16866321],
       [199.13096101,  27.01104231,  45.30013382],
       [168.5003722 ,  38.64441063,  52.81041635],
       [133.97360158,  91.59693268, 138.34643773],
       [ 50.28297449,  51.95537184,  23.62294384],
       [177.13769118, 134.66889799,   8.71668886],
       [ 30.68289591,   9.45788771,  50.50424856],
       [ 64.12769861, 195.12306707,  81.57164046],
       [ 56.68936652, 130.84497227, 108.2354709 ],
       [ 82.99174886, 121.03920058, 172.37415331],
       [127.09687502, 120.51398912, 134.53558472],
       [ 33.36926752,  13.63720659,   0.60996262],
       [ 18.39375397, 191.40492629, 122.47329764],
       [ 19.63641252,  63.09685344, 174.17147189],
       [ 90.74169878, 154.13735512, 104.39306946],
       [ 86.557716  ,  72.91515918, 182.95789915],
       [119.33310673,  37.76924713, 149.28485556],
       [ 17.13480987, 132.62322674, 189.63031283],
       [109.46636148, 118.08537517,  76.31000499],
       [170.09845239,   3.4854559 ,  23.50245978],
       [ 18.70476325, 147.66938079,  58.80734523],
       [111.64277024, 157.053698  , 132.24005744],
       [ 42.14101276,  50.93855746, 176.75407925],
       [ 14.51198083,  16.93867057,  60.2360491 ],
       [ 52.77856703,  29.30614115,  36.71599   ],
       [ 38.07656718,  13.10864189,  72.07363165],
       [138.51107657, 177.90126384, 114.77664924],
       [ 25.84809123, 112.82413992,  33.27661835],
       [  6.6733563 , 160.69190438, 139.70034561],
       [197.70475434,  13.8230847 ,  60.37034021],
       [ 12.25179549, 100.55314862,  78.92943814],
       [140.93499401, 164.76389065, 124.56483354],
       [100.40316917,  68.76075668,  17.56098047],
       [144.15542294,  50.57241845,  83.47993277],
       [ 43.42498771,  75.67311377, 122.44988237],
       [ 73.56937364,  38.65136447,   0.68298585],
       [160.40914341,   1.50148649,   2.225444  ],
       [127.8706212 ,  29.43082072,  23.094503  ],
       [ 10.24110325, 111.83791255, 156.45308871],
       [161.49404006, 163.38664412, 161.98093237],
       [ 27.04229405,  45.36589381, 117.35604037],
       [126.85079112, 162.26356355,  70.60492779],
       [128.66689612,  68.71766065, 150.35375572],
       [166.89624473, 124.99140447,  49.97804878],
       [ 22.58882999, 172.17683864,  41.74842148],
       [173.1491633 , 103.35193697,  79.45686904],
       [ 63.47467018, 157.89290094,  54.88476635],
       [ 83.264726  , 141.90140818, 151.48688241],
       [ 71.99367711,  16.40398225, 197.67281455],
       [ 55.41434491, 148.76063021,   3.30900276],
       [136.79240841,  60.8792625 , 193.43780416],
       [ 21.14423948,  15.16246438, 137.40606156],
       [ 37.08759453,  55.1292117 ,  75.06349727],
       [  6.48071698, 123.73465632,   4.00908165],
       [ 49.63963798,  63.59991319, 136.27840358],
       [  6.47400652, 161.05000329, 163.66077338],
       [142.56695796, 110.77450728, 194.29867779],
       [ 85.21456401,  72.43223496, 118.71280211],
       [103.95645911,   4.65524665, 154.29805374],
       [151.90410329, 147.74780743,  72.22549058]])

# ---- cell-2 ----
lb         = 0.84
L          = 200.
N_WEIGHTS  = 1024
T_SCAN_MAX = 70.
N_SCAN     = 32

WEIGHTS_IN  = DATA_DIR / 'weights_in_N64.h5'
WEIGHTS_OUT = DATA_DIR / 'weights_out_N64.h5'

scan_times    = np.linspace(1., T_SCAN_MAX, N_SCAN)
weights_times = np.linspace(1., T_SCAN_MAX, N_WEIGHTS)

# ---- cell-6 (load scan) ----
cache = DATA_DIR / f'phi4_lambda_bar{lb}' / 'scan_cache.h5'
scan_results = load_scan(cache)

# ---- cell-8 (load weights) ----
weights, pair_i, pair_j, gammas = load_weights(WEIGHTS_OUT, N_WEIGHTS)
n_pairs = len(pair_i)
print(f'{n_pairs} colliding pairs')

# ---- 878215d2 (verbatim) ----
first_gamma_check = next(iter(scan_results.values()))
if first_gamma_check.amp_re is None:
    raise RuntimeError('no amplitude data')

gammas_grid_amp = np.array(sorted(scan_results.keys()))
first_res = scan_results[gammas_grid_amp[0]]

k_amp = first_res.k
n_w_amp, n_k_amp = first_res.amp_re.shape[1:]
w_amp = first_res.w
print(f'Amplitude grid: n_w={n_w_amp}  n_k={n_k_amp}  '
      f'cos_theta=[{k_amp[0]:.3f},{k_amp[-1]:.3f}]')


def build_amp_interpolator(scan_results, w_out):
    grid_gammas = np.array(sorted(scan_results.keys()))
    grid_times  = next(iter(scan_results.values())).times
    Ng, Nt      = len(grid_gammas), len(grid_times)
    n_w, n_k    = len(w_out), scan_results[grid_gammas[0]].amp_re.shape[2]

    re_grid = np.zeros((Ng, Nt, n_w, n_k))
    im_grid = np.zeros((Ng, Nt, n_w, n_k))
    for ig, g in enumerate(grid_gammas):
        res = scan_results[g]
        if res.amp_re is None:
            raise RuntimeError(f'gamma={g} has no amplitude data in scan_cache.h5')
        if np.array_equal(res.w, w_out):
            re_grid[ig] = res.amp_re
            im_grid[ig] = res.amp_im
        else:
            for it in range(Nt):
                for ik in range(n_k):
                    re_grid[ig, it, :, ik] = np.interp(w_out, res.w, res.amp_re[it, :, ik],
                                                        left=0., right=0.)
                    im_grid[ig, it, :, ik] = np.interp(w_out, res.w, res.amp_im[it, :, ik],
                                                        left=0., right=0.)

    flat_shape = (Ng, Nt, n_w * n_k)
    _raw_re = RegularGridInterpolator((grid_gammas, grid_times), re_grid.reshape(flat_shape),
                                      bounds_error=False, fill_value=0.)
    _raw_im = RegularGridInterpolator((grid_gammas, grid_times), im_grid.reshape(flat_shape),
                                      bounds_error=False, fill_value=0.)

    def interp_fn(pts):
        re = _raw_re(pts).reshape(-1, n_w, n_k)
        im = _raw_im(pts).reshape(-1, n_w, n_k)
        return re, im

    return interp_fn, grid_gammas, grid_times


amp_interp_fn, gammas_grid_amp, scan_t_amp = build_amp_interpolator(scan_results, w_amp)
print(f'Amplitude interpolator built: {len(gammas_grid_amp)} gammas x {len(scan_t_amp)} times')

# ---- 1c41d030 (verbatim) ----
pair_axes    = {}
pair_centers = {}
for p in range(n_pairs):
    i, j = int(pair_i[p]), int(pair_j[p])
    dp = bubbles[i] - bubbles[j]
    dp -= L * np.round(dp / L)
    pair_axes[p]    = dp / np.linalg.norm(dp)
    pair_centers[p] = (0.5 * (bubbles[i] + bubbles[j])) % L

N_THETA, N_PHI = 64, 128
cos_nodes, gl_w = leggauss(N_THETA)
phi_vals = np.linspace(0., 2.*np.pi, N_PHI, endpoint=False)
d_phi    = 2.*np.pi / N_PHI

CT, PH = np.meshgrid(cos_nodes, phi_vals, indexing='ij')
ST     = np.sqrt(np.maximum(1. - CT**2, 0.))
hat_k  = np.stack([ST*np.cos(PH), ST*np.sin(PH), CT], axis=-1).reshape(-1, 3)
d_omega = np.outer(gl_w, np.full(N_PHI, d_phi)).ravel()

N_sph = len(hat_k)
print(f'Sphere quadrature: {N_sph} points')

cos_th = {p: np.clip(np.abs(hat_k @ pair_axes[p]), 0., 1.) for p in range(n_pairs)}
ph_ang = {p: np.outer(w_amp, hat_k @ pair_centers[p]) for p in range(n_pairs)}
cp_ang = {p: np.cos(ph_ang[p]) for p in range(n_pairs)}
sp_ang = {p: np.sin(ph_ang[p]) for p in range(n_pairs)}
print(f'Precomputed direction factors for {n_pairs} pairs')

# ---- 8ffe67f2 (verbatim) ----
def active_pairs_for(t_max, threshold=0.01):
    mask_t = weights_times <= t_max
    return [p for p in range(n_pairs) if np.any(weights[p][mask_t] > threshold)]


def _amp_at_times(g_arr, t_pts):
    n_p, n_t = len(g_arr), len(t_pts)
    pts = np.column_stack([np.repeat(g_arr, n_t), np.tile(t_pts, n_p)])
    re, im = amp_interp_fn(pts)
    return re.reshape(n_p, n_t, n_w_amp, n_k_amp), im.reshape(n_p, n_t, n_w_amp, n_k_amp)


def coherent_sum_n64(t_max, pair_batch=32, t_chunk=32):
    import time as _time
    active   = active_pairs_for(t_max)
    n_active = len(active)
    g_arr    = np.clip(gammas[active], gammas_grid_amp.min(), gammas_grid_amp.max())

    t_hi   = min(t_max, scan_t_amp[-1])
    mask   = (weights_times >= scan_t_amp[0]) & (weights_times <= t_hi)
    t_grid = weights_times[mask]
    if len(t_grid) == 0 or t_grid[-1] < t_hi:
        t_grid = np.append(t_grid, t_hi)
    n_t = len(t_grid)

    cos_th_stack = np.stack([cos_th[p] for p in active])
    cp_stack     = np.stack([cp_ang[p] for p in active])
    sp_stack     = np.stack([sp_ang[p] for p in active])

    idx  = np.clip(np.searchsorted(k_amp, cos_th_stack), 1, n_k_amp - 1)
    frac = (cos_th_stack - k_amp[idx-1]) / (k_amp[idx] - k_amp[idx-1])
    idx_lo3 = np.broadcast_to((idx-1)[:, None, :], (n_active, n_w_amp, N_sph))
    idx_hi3 = np.broadcast_to(idx[:, None, :],     (n_active, n_w_amp, N_sph))
    frac3   = frac[:, None, :]

    t_start = _time.time()
    print(f'  t_max={t_max:.1f}: {n_active}/{n_pairs} pairs active, {n_t} fine time steps')

    P_coh = np.zeros(n_w_amp)
    for c0 in range(0, n_t - 1, t_chunk):
        c1    = min(c0 + t_chunk, n_t - 1)
        t_pts = t_grid[c0:c1 + 1]

        re_c, im_c = _amp_at_times(g_arr, t_pts)
        dar_c = np.diff(re_c, axis=1)
        dai_c = np.diff(im_c, axis=1)

        wp_c = np.sqrt(np.clip(
            np.stack([np.interp(t_pts[:-1], weights_times, weights[p]) for p in active]),
            0., None))
        dar_c = dar_c * wp_c[:, :, None, None]
        dai_c = dai_c * wp_c[:, :, None, None]

        for it in range(dar_c.shape[1]):
            A_tot_re = np.zeros((n_w_amp, N_sph))
            A_tot_im = np.zeros((n_w_amp, N_sph))
            for lo in range(0, n_active, pair_batch):
                hi = min(lo + pair_batch, n_active)

                re_lo = np.take_along_axis(dar_c[lo:hi, it], idx_lo3[lo:hi], axis=2)
                re_hi = np.take_along_axis(dar_c[lo:hi, it], idx_hi3[lo:hi], axis=2)
                im_lo = np.take_along_axis(dai_c[lo:hi, it], idx_lo3[lo:hi], axis=2)
                im_hi = np.take_along_axis(dai_c[lo:hi, it], idx_hi3[lo:hi], axis=2)
                f = frac3[lo:hi]

                Ap_re = re_lo * (1 - f) + re_hi * f
                Ap_im = im_lo * (1 - f) + im_hi * f

                cp_b = cp_stack[lo:hi]
                sp_b = sp_stack[lo:hi]
                A_tot_re += (Ap_re * cp_b - Ap_im * sp_b).sum(axis=0)
                A_tot_im += (Ap_re * sp_b + Ap_im * cp_b).sum(axis=0)

            P_coh += ((A_tot_re**2 + A_tot_im**2) * d_omega[None, :]).sum(axis=-1)

        print(f'    fine steps {c0+1}-{c1}/{n_t-1}: {_time.time()-t_start:.1f} s elapsed', flush=True)

    print(f'  t_max={t_max:.1f} done: {_time.time()-t_start:.1f} s total')
    return P_coh * w_amp**3 * 2. * np.pi


# ---- 7d195bd0 (verbatim) ----
t_max_test = 5.0
active_test = active_pairs_for(t_max_test)
P_coh_test = coherent_sum_n64(t_max_test)
print(f'validation reference: t_max={t_max_test}  n_active={len(active_test)}')
for p in active_test:
    print(f'  active pair idx={p}  gamma={gammas[p]:.17g}  pair_i={int(pair_i[p])}  pair_j={int(pair_j[p])}')

np.savez('/tmp/real_python_reference.npz', w=w_amp, P_coh=P_coh_test, t_max=t_max_test,
         n_active=len(active_test), active=np.array(active_test))
print('Saved /tmp/real_python_reference.npz')
