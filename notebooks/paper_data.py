"""Data for the paper figures (notebook 02_paper_figures.ipynb).

Every figure's numbers are built once by a `build_*` function and stored in
data/paper_cache/<name>.pkl; `cached(name, builder)` only rebuilds when the
file is missing (or rebuild=True). The notebook itself only loads and plots.

Conventions (agreed for the paper):
  * lattice spectra: (A+B)/2 of sledgehamr's gw_spectra and gw_spec_u_times_k.
  * surrogate: pair spectra at wall_points=100 (notebook 01_paper_runs): N_b=2,3 from dedicated runs, one per lattice pair at its
    exact gamma_ij (lambda_bar = 0.84 / 0.069); N_b>3 from gamma_*=1-6 scans with time_factor 2.5 (lambda_bar = 0.069,
    and 0.844777 for the "0.845" runs),
    our BubbleMaster cutoff window, D^3.6 damping (weight_decay_handover.pdf,
    Sec. 3 'D', calibrated on our 1+1D runs, data/damping_D36_kernel.json).
  * lattice vs surrogate spectra are compared through the lattice's own shell
    estimator (surrogate evaluated at the occupied lattice-mode radii, shell
    summed, complete shells only).
  * integrated power I_K = sum_n P_n dk/k_n over shells (the exact sum over lattice modes), lattice and
    surrogate alike: kR_* <= 100 for N_b=2 (fig 2), kR_* <= 30 for N_b>3 (fig 5, excludes the true-vacuum oscillations).
  * normalization [H_* R_* Omega_vac]^-2 uses each run's exact lambda_bar.
"""
from pathlib import Path
import json, os, pickle, subprocess, tempfile, warnings

import h5py
import numpy as np
from scipy.interpolate import PchipInterpolator

import ptbuilder as pt
from ptbuilder.analysis import (load_scan, load_weights, to_chw, build_runtime_scan_interpolator,
                                reconstruct_pair_spectrum, reconstruct_pair_spectrum_series,
                                write_weights_input)
from ptbuilder.sledgehamr import load_output, get_gw_spectrum
from ptbuilder.patch_fraction import damped_weights
from ptbuilder.ic import _cutoff_times
from ptbuilder.weight_scan_diagnostics import load_kinematics

ROOT = Path(pt.__file__).parent.parent
DATA = ROOT / "data"
SH = ROOT / "sledgehamr_runs"
CACHE = DATA / "paper_cache"
CACHE.mkdir(exist_ok=True)
LB845 = 0.8447772491349482          # exact lambda_bar of the "0.845" runs
KR_LO, KR_HI = 1.0, 100.0
KR_IPR = 30.0          # upper edge of the N_b>3 integrated power (fig 5)

SCANS = {0.069: "runtime_scan_lb0.069_gs1-6_kR1-100_nw32",          # N_b > 3: wall_points 100, time_factor 2.5 (notebook 01_paper_runs)
         LB845: "runtime_scan_lb0.844777_gs1-6_kR1-100_nw32"}
N23_RUNS = {0.84: "n2_dedicated_lb0.84_w100",                      # N_b = 2, 3: one row per lattice pair at its exact gamma_ij,
            0.069: "n2_dedicated_lb0.069_w100"}                     # with t_m as an output time (notebook 01_paper_runs)
LB_TAG = {0.84: "0.84", 0.069: "0.069", LB845: "0.844777"}   # wall-radius table / kernel keys


# ----------------------------------------------------------------------------- cache
def cached(name, builder, rebuild=False):
    path = CACHE / f"{name}.pkl"
    if path.exists() and not rebuild:
        return pickle.load(open(path, "rb"))
    val = builder()
    pickle.dump(val, open(path, "wb"))
    return val


# ----------------------------------------------------------------------------- lattice I/O
class AveragedOutput:
    """(A+B)/2 of two sledgehamr outputs with identical snapshot times."""
    def __init__(self, a, b):
        self._a, self._b = a, b

    def GetTimesOfGravitationalWaveSpectra(self):
        return self._a.GetTimesOfGravitationalWaveSpectra()

    def GetGravitationalWaveSpectrum(self, idx):
        da, db = self._a.GetGravitationalWaveSpectrum(idx), self._b.GetGravitationalWaveSpectrum(idx)
        assert abs(da["t"] - db["t"]) < 1e-6
        return {"t": da["t"], "k_sq": da["k_sq"], "spectrum": 0.5 * (da["spectrum"] + db["spectrum"])}


def load_averaged(run):
    path = SH / run
    a = load_output(str(path))
    tmp = Path(tempfile.mkdtemp()); os.symlink((path / "gw_spec_u_times_k").resolve(), tmp / "gw_spectra")
    return AveragedOutput(a, load_output(str(tmp)))


def lattice_snapshot(out, idx, L):
    """(k, raw spectrum) of one snapshot, zero mode dropped."""
    k, y, _ = get_gw_spectrum(out, int(idx), L, 1.)
    return k[1:], y[1:]


# ----------------------------------------------------------------------------- shell estimator
_MULT = {}
def sq_multiplicities(sq_max):
    """Number of integer vectors n with |n|^2 = s, s = 0..sq_max (memoized: the convolution is expensive)."""
    for m, v in _MULT.items():
        if m >= sq_max: return v[:sq_max + 1]
    _MULT[sq_max] = _sq_multiplicities(sq_max); return _MULT[sq_max]


def _sq_multiplicities(sq_max):
    c1 = np.zeros(sq_max + 1)
    x = np.arange(-int(np.sqrt(sq_max)) - 1, int(np.sqrt(sq_max)) + 2)
    np.add.at(c1, x[x * x <= sq_max] ** 2, 1)
    from scipy.signal import fftconvolve
    c2 = np.rint(fftconvolve(c1, c1)[:sq_max + 1])
    return np.rint(fftconvolve(c2, c1)[:sq_max + 1])


def through_shells(Y_of_k, L, keff, k_lo, k_hi, n_max):
    """Smooth Y(k) = dOmega/dlnk pushed through the lattice shell estimator:
    shell n = round(|n_vec|), Y_n = keff_n * sum_{modes in shell} Y(2 pi |n|/L) / (4 pi |n|^3).
    Returns (n, Y_n, complete) for n = 0..n_max; complete = every mode of the shell in [k_lo, k_hi]."""
    mult = sq_multiplicities(int((n_max + 0.5) ** 2))
    s = np.nonzero(mult)[0]; s = s[s > 0]
    r = np.sqrt(s); shell = np.floor(r + 0.5).astype(int); k = 2 * np.pi * r / L
    keep = shell <= n_max; s, r, shell, k = s[keep], r[keep], shell[keep], k[keep]
    ok = (k >= k_lo) & (k <= k_hi)
    contrib = np.zeros(s.size); contrib[ok] = mult[s[ok]] * Y_of_k(k[ok]) / (4 * np.pi * r[ok] ** 3)
    Yn = keff[:n_max + 1] * np.bincount(shell, weights=contrib, minlength=n_max + 1)
    incomplete = np.bincount(shell, weights=(~ok).astype(float), minlength=n_max + 1) > 0
    n = np.arange(n_max + 1)
    return n, Yn, (~incomplete) & (n > 0)


def shell_mode_counts(n_max):
    mult = sq_multiplicities(int((n_max + 0.5) ** 2)); sq = np.arange(mult.size)
    return np.bincount(np.floor(np.sqrt(sq) + 0.5).astype(int), weights=mult, minlength=n_max + 1)[:n_max + 1]


def loglog_fn(k_grid, y):
    """Y(k) by log-log interpolation of a surrogate spectrum given on k_grid."""
    lk, ly = np.log(k_grid), np.log(np.maximum(y, 1e-300))
    return lambda k: np.exp(np.interp(np.log(k), lk, ly))


def I_K_shells(y_n, keff, n_hi):
    """Integrated power: sum over shells 1..n_hi of y_n dk/k_n (dk/k_n = 1/keff_n)."""
    return float(np.sum(y_n[1:n_hi + 1] / keff[1:n_hi + 1]))


# ----------------------------------------------------------------------------- damping kernel
class D36Kernel:
    """K = Dhat^3.6 (handover Sec. 3 'D') from the stored tables: monotone PCHIP in age,
    power-law tail beyond each row, linear in ln Dhat between boosts, boosts clamped."""
    per_gamma = True

    def __init__(self, lb, path=DATA / "damping_D36_kernel.json"):
        tab = json.load(open(path)); self.c = tab["c"]
        rows = tab["lambda_bar"][LB_TAG[lb]]
        self.gammas = np.array([r["gamma"] for r in rows])
        self.rows = [(np.array(r["age_d"]), np.array(r["ln_Dhat"]), r["q"]) for r in rows]
        self.pchip = [PchipInterpolator(x, f) for x, f, q in self.rows]

    def _lnD(self, i, age):
        x, f, q = self.rows[i]; age = np.asarray(age, float); out = np.empty_like(age)
        inside = age <= x[-1]; out[inside] = self.pchip[i](np.maximum(age[inside], 0.))
        out[~inside] = f[-1] - q * np.log(age[~inside] / x[-1])
        return out

    def __call__(self, gamma):
        gs = self.gammas; g = float(np.clip(gamma, gs[0], gs[-1]))
        j = int(np.clip(np.searchsorted(gs, g) - 1, 0, len(gs) - 2)); w = (g - gs[j]) / (gs[j + 1] - gs[j])
        return lambda age: np.exp(self.c * ((1 - w) * self._lnD(j, age) + w * self._lnD(j + 1, age)))


_KERN, _PROF = {}, {}
def kernel(lb):
    if lb not in _KERN: _KERN[lb] = D36Kernel(lb)
    return _KERN[lb]


def profile(lb):
    if lb not in _PROF: _PROF[lb] = pt.PhysicsModel(pt.Phi4Potential(lb), pt.Config()).instanton
    return _PROF[lb]


def damp(w, t, gam, lb, t_end=None, spacing=None):
    """D^3.6-damped weights; optional extension of the time grid (new-style runs)."""
    return damped_weights(w, t, gam, profile(lb), kernel(lb), t_end=t_end, extension_spacing=spacing)


# ----------------------------------------------------------------------------- scans
_SCAN = {}
def scan(lb, n23=False):
    """(interp_fn, gammas, k_out, time bounds) of the scan for lambda_bar lb: the dedicated N_b=2 runs if n23, else the N_b>3 scan."""
    key = (lb, n23)
    if key not in _SCAN:
        sc = load_scan(DATA / (N23_RUNS if n23 else SCANS)[lb])
        k_out = np.geomspace(min(r.wlist[0] for r in sc.rows), max(r.wlist[-1] for r in sc.rows), 128)
        f, g = build_runtime_scan_interpolator(sc, k_out)
        _SCAN[key] = (f, g, k_out, np.array([min(r.times[0] for r in sc.rows), sc.t_max]))
    return _SCAN[key]


def surrogate_at(w, t, gam, lb, T, n23=False):
    f_i, g_s, k_out, tb = scan(lb, n23)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return k_out, sum(reconstruct_pair_spectrum(float(g), w[p], t, T, f_i, tb, g_s) for p, g in enumerate(gam))


def surrogate_series(w, t, gam, lb, tq, n23=False):
    f_i, g_s, k_out, tb = scan(lb, n23)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return k_out, sum(reconstruct_pair_spectrum_series(float(g), w[p], t, tq, f_i, tb, g_s) for p, g in enumerate(gam))


def hold_extend(w, t, t_to):
    """Hold the final weight past the end of the weights grid (constant isolated-pair weights)."""
    if t_to <= t[-1] or not np.allclose(w[:, -1], w[:, -2]):
        return w, t
    dt = t[1] - t[0]; ext = np.arange(t[-1] + dt, t_to + dt, dt)
    return np.hstack([w, np.repeat(w[:, -1:], len(ext), axis=1)]), np.r_[t, ext]


def compare_shells(kR_rec, y_rec, R, L, keff):
    """Surrogate (kR_rec, y_rec) pushed through the lattice shells: returns (shell mask, Y_n on complete shells)."""
    n_max = int(np.floor(min(KR_HI, kR_rec[-1]) / R * L / (2 * np.pi)))
    nn, Yn, comp = through_shells(loglog_fn(kR_rec / R, y_rec), L, keff, max(KR_LO, kR_rec[0]) / R, min(KR_HI, kR_rec[-1]) / R, n_max)
    mask = np.zeros(len(keff) - 1, bool); mask[nn[comp] - 1] = True
    return mask, Yn[comp]


def I_K_series_surrogate(k_out, comb, R, lb, L, keff, n_hi):
    """Surrogate integrated power at each query time, through the lattice shells."""
    out = []
    for q in range(comb.shape[0]):
        y = np.maximum(to_chw(comb[q], R, lb, L**3), 1e-300)
        _, Yq, _ = through_shells(loglog_fn(k_out, y), L, keff, 0., np.inf, n_hi)
        out.append(I_K_shells(Yq, keff, n_hi))
    return np.array(out)


# ----------------------------------------------------------------------------- N_b = 2 and 3 (figs 1-3)
N23 = {0.84: dict(L=200., t_w_end=70., pos=[[72.25, 89.92, 100.], [127.8, 89.92, 100.], [108.7, 110.1, 100.]],
                  pairs=[((0, 1), "bubble_N3_234_0_1__nocutoff", "N3_084_pair_0_1"), ((0, 2), "bubble_N3_234_0_2__nocutoff", "N3_084_pair_0_2"),
                         ((1, 2), "bubble_N3_234_1_2__nocutoff", "N3_084_pair_1_2")],
                  trio=("bubble_N3_234_0_1_2_nocutoff", "N3_084_3b")),
       0.069: dict(L=50., t_w_end=30., pos=[[16.23, 19.06, 22.5], [28.77, 19.06, 22.5], [24.08, 25.94, 22.5]],
                   pairs=[((0, 1), "bubble_N3_456_0_1__nocutoff", "N3_069_pair_0_1"), ((0, 2), "bubble_N3_456_0_2__nocutoff", "N3_069_pair_0_2"),
                          ((1, 2), "bubble_N3_456_1_2__nocutoff", "N3_069_pair_1_2")],
                   trio=("bubble_N3_456_0_1_2_nocutoff", "N3_069_3b"))}


def _pair_sep(c, i, j):
    pos = np.array(c["pos"]); dp = pos[i] - pos[j]; dp -= c["L"] * np.round(dp / c["L"]); return float(np.linalg.norm(dp))


def _image_contact(lb, c, i, j):
    """First periodic-image contact of a pair, from the measured wall midpoint radius."""
    pos = np.array(c["pos"]); L = c["L"]
    dists = sorted(np.linalg.norm(pos[j] + L * (np.array(n) - 1) - pos[i]) for n in np.ndindex(3, 3, 3))[1:] + [L]
    with h5py.File(DATA / f"wall_radius/lb{LB_TAG[lb]}.h5") as tab:
        return float(np.interp(min(dists) / 2, tab["R_mid"][:], tab["s"][:]))


def build_n2():
    """Fig 1 (spectra at the lattice snapshot nearest t_m) and Fig 2 (I_K(t) until image contact) for each N_b=2 pair."""
    out = {}
    for lb, c in N23.items():
        L = c["L"]
        for (i, j), shd, wb in c["pairs"]:
            d = _pair_sep(c, i, j); t_m = float(_cutoff_times(d, 0, 1.)[2]); t_img = _image_contact(lb, c, i, j)
            w, _, _, gam = load_weights(DATA / f"weights_out_{wb}_rnum.h5", 500); t = np.linspace(1., c["t_w_end"], 500)
            w, t = hold_extend(w, t, scan(lb, True)[3][1])
            w, t = damp(w, t, gam, lb)              # no effect before image contact (the pair's weight is constant; image pairs collide later)
            sh = load_averaged(shd); times = np.array(sh.GetTimesOfGravitationalWaveSpectra())
            # Fig 1: lattice snapshot nearest t_m, surrogate at exactly that time
            idx = int(np.argmin(abs(times - t_m))); t_s = float(times[idx])
            k, y = lattice_snapshot(sh, idx, L); keff = np.r_[0., k * L / (2 * np.pi)]
            y_sh = to_chw(y, d, lb, L**3); n_modes = shell_mode_counts(len(k))[1:]
            k_out, s = surrogate_at(w, t, gam, lb, t_s, True); y_rec = to_chw(s, d, lb, L**3)
            mask, Yn = compare_shells(k_out * d, y_rec, d, L, keff)
            # Fig 2: I_K at every lattice snapshot before image contact
            tl = times[times < t_img]; n_hi = int(np.floor(KR_HI / d * L / (2 * np.pi)))
            I_lat = np.array([I_K_shells(np.r_[0., to_chw(lattice_snapshot(sh, q, L)[1], d, lb, L**3)], keff, n_hi) for q in range(len(tl))])
            k_out, comb = surrogate_series(w, t, gam, lb, tl, True)
            I_red = I_K_series_surrogate(k_out, comb, d, lb, L, keff, n_hi)
            out[(lb, (i, j))] = dict(gamma=float(gam.min()), d=d, t_m=t_m, t_snap=t_s, t_img=t_img,
                                     kR_sh=k * d, y_sh=y_sh, yerr=y_sh * np.sqrt(2. / n_modes),
                                     kR_rec=k_out * d, y_rec=y_rec, shell_mask=mask, rec_shells=Yn,
                                     t=tl, I_lat=I_lat, I_red=I_red)
    return out


def build_n3():
    """Fig 3: three-bubble runs at the lattice snapshot nearest the latest pair t_m; damped surrogate."""
    out = {}
    for lb, c in N23.items():
        L = c["L"]; seps = [_pair_sep(c, i, j) for (i, j), _, _ in c["pairs"]]
        R = float(np.mean(seps)); t_m3 = max(float(_cutoff_times(dd, 0, 1.)[2]) for dd in seps)
        shd, wb = c["trio"]
        with h5py.File(DATA / f"weights_in_{wb}.h5") as f: t = f["t"][:]
        w, _, _, gam = load_weights(DATA / f"weights_out_{wb}_rnum.h5", len(t))
        w_raw, t_raw = w, t
        w, t = damp(w, t, gam, lb)
        sh = load_averaged(shd); times = np.array(sh.GetTimesOfGravitationalWaveSpectra())
        idx = int(np.argmin(abs(times - t_m3))); t_s = float(times[idx])
        k, y = lattice_snapshot(sh, idx, L); keff = np.r_[0., k * L / (2 * np.pi)]
        y_sh = to_chw(y, R, lb, L**3); n_modes = shell_mode_counts(len(k))[1:]
        k_out, s = surrogate_at(w, t, gam, lb, t_s, True); y_rec = to_chw(s, R, lb, L**3)
        mask, Yn = compare_shells(k_out * R, y_rec, R, L, keff)
        y_rec_raw = to_chw(surrogate_at(w_raw, t_raw, gam, lb, t_s, True)[1], R, lb, L**3)   # undamped
        _, Yn_raw = compare_shells(k_out * R, y_rec_raw, R, L, keff)
        out[lb] = dict(R=R, t_m3=t_m3, t_snap=t_s, gammas=sorted(float(g) for g in gam[:3]),
                       kR_sh=k * R, y_sh=y_sh, yerr=y_sh * np.sqrt(2. / n_modes),
                       kR_rec=k_out * R, y_rec=y_rec, shell_mask=mask, rec_shells=Yn,
                       y_rec_raw=y_rec_raw, rec_shells_raw=Yn_raw)
    return out


# ----------------------------------------------------------------------------- N_b > 3 (figs 4-5)
# panel = gamma_* column; every 0.845 run is rebuilt from its initial state with lambda_bar = 0.844777
# kinematics and wall-radius table (new=True); the 0.069 runs keep their tracked weights inputs.
NMANY = [
    dict(lb=LB845, ic="Cutting_lb0.845_N29_L175.53", sh="Cutting_lb0.845_N29_L175.53", panel=4, col="C0"),   # same dx as the gamma_*=5 run
    dict(lb=LB845, ic="Cutting_lb0.845_N512_g4.00", sh="Cutting_lb0.84_N512_g4.00", panel=4, col="tab:brown"),
    dict(lb=LB845, ic="Cutting_lb0.845_N15_L176.15", sh="Cutting_lb0.845_N15_L176.15", panel=5, col="C0"),
    dict(lb=LB845, ic="Cutting_lb0.845_N15_L211.4", sh="Cutting_lb0.845_N15_L211.4", panel=6, col="C0"),
    dict(lb=LB845, ic="Cutting_lb0.845_N120_L422.8", sh="Cutting_lb0.845_N120_L422.8", panel=6, col="tab:brown"),
    dict(lb=0.069, ic="Cutting_lb0.069_N64_g4.00_v4", sh="Cutting_lb0.069_N64_g4.00_leapfrog_2048_long_v2", win="N64_lb069_v4", panel=4, col="C0"),
    dict(lb=0.069, ic="Cutting_lb0.069_N34_L33.8", sh="Cutting_lb0.069_N34_L33.8", panel=5, col="C0"),
    dict(lb=0.069, ic="Cutting_lb0.069_N48_g6.00", sh="Cutting_lb0.069_N48_g6.00", win="N48_lb069_g6", panel=6, col="C0"),
]


def _covering_radius(P, L):
    from scipy.spatial import Voronoi, cKDTree
    # all 26 neighbouring periodic copies: with only a few bubbles per box, copies further than a fraction of L
    # away still bound the Voronoi cells inside the box
    imgs = [P + L * np.array([dx, dy, dz]) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]
    V = Voronoi(np.vstack(imgs)).vertices; V = V[np.all((V >= 0) & (V < L), axis=1)]
    return cKDTree(P, boxsize=L).query(V)[0].max()


def run_meta(r):
    with h5py.File(SH / f"{r['ic']}.h5") as f:
        hdr = f["Header"][:]; N = len(f["xlocs"])
        P = np.column_stack([f["xlocs"][:], f["ylocs"][:], f["zlocs"][:]]) % float(hdr[1])
    L = float(hdr[1]); return dict(N_b=N, L=L, gamma_star=float(hdr[3]), R=(L**3 / N) ** (1 / 3), P=P)


def nmany_weights(r):
    """Raw weights (w, t, gammas) and the damped grid end of a run. 0.845 runs: built here from the initial state
    (lambda_bar=0.844777 kinematics and wall-radius table, grid to 5% past the box-filling time, steps R_*/450)."""
    m = run_meta(r)
    if "win" in r:
        with h5py.File(DATA / f"weights_in_{r['win']}.h5") as f: t = f["t"][:]
        w, _, _, g = load_weights(DATA / f"weights_out_{r['win']}_rnum.h5", len(t))
        return w, t, g, None
    tag = r["ic"] + ("_lb0.844777" if r["lb"] == LB845 else "")
    win, wout = CACHE / f"weights_in_{tag}.h5", CACHE / f"weights_out_{tag}_rnum.h5"
    wall = DATA / f"wall_radius/lb{LB_TAG[r['lb']]}.h5"
    if not wout.exists():
        with h5py.File(wall) as tab:
            t_fill = np.interp(_covering_radius(m["P"], m["L"]), tab["R_mid"][:], tab["s"][:])
        t_end = 1.05 * t_fill; n = int(np.ceil((t_end - 1.) / (m["R"] / 450.))) + 1
        kin = load_kinematics(DATA / f"phi4_lambda_bar{LB_TAG[r['lb']]}" / "instanton.h5")
        write_weights_input(path=win, positions=m["P"], t=np.linspace(1., t_end, n), kinematics=kin, L=m["L"], collision_radius="mid")
        subprocess.run([str(ROOT / "cpp/weights/weights"), str(win), str(wout), "--wall-radius", str(wall)], check=True, capture_output=True)
    with h5py.File(win) as f: t = f["t"][:]
    w, _, _, g = load_weights(wout, len(t))
    return w, t, g, t[-1] + m["R"]


def late_time_band(out, times, t_from, L):
    """Median / min / max / last over snapshots t >= t_from (raw units, interpolated onto the last snapshot's k)."""
    late = np.where(times >= t_from)[0]
    k_ref, _ = lattice_snapshot(out, late[-1], L)
    S = np.array([np.interp(k_ref, *lattice_snapshot(out, q, L), left=np.nan, right=np.nan) for q in late])
    return k_ref, np.nanmedian(S, axis=0), np.nanmin(S, axis=0), np.nanmax(S, axis=0), S[-1], times[late]


def build_nmany_one(r, central="median"):
    """central: lattice central value, "median" over the late-time window or the "last" snapshot (band = window min/max either way)."""
    m = run_meta(r); R, L, lb = m["R"], m["L"], r["lb"]
    w, t, g, t_ext = nmany_weights(r)
    t_coll_end = float(t[np.where(w.sum(axis=0) > 0.01)[0][-1]])
    wd, td = damp(w, t, g, lb, t_end=t_ext, spacing=(R / 150. if t_ext else None))
    wr = np.array([np.interp(td, t, row, left=0., right=0.) for row in w])     # undamped, on the same grid
    sh = load_averaged(r["sh"]); times = np.array(sh.GetTimesOfGravitationalWaveSpectra())
    # Fig 4: late-time lattice band vs surrogate at min(weights end, last snapshot)
    T = min(float(td[-1]), float(times[-1]))
    k_ref, med, lo, hi, last, tband = late_time_band(sh, times, t_coll_end, L)
    if central == "last":
        med = last                                # stored as y_med below
    keff = np.r_[0., k_ref * L / (2 * np.pi)]; n_modes = shell_mode_counts(len(k_ref))[1:]
    sig = med * np.sqrt(2. / n_modes)
    lo = med - np.sqrt((med - lo) ** 2 + sig ** 2); hi = med + np.sqrt((hi - med) ** 2 + sig ** 2)
    norm = lambda y: to_chw(y, R, lb, L**3)
    k_out, s = surrogate_at(wd, td, g, lb, T); y_rec = norm(s)
    mask, Yn = compare_shells(k_out * R, y_rec, R, L, keff)
    y_rec_raw = norm(surrogate_at(wr, td, g, lb, T)[1])
    _, Yn_raw = compare_shells(k_out * R, y_rec_raw, R, L, keff)
    # Fig 5: I_K at every snapshot, shells with kR_* <= KR_IPR (excludes the lattice-only true-vacuum oscillation bump)
    n_hi = int(np.floor(KR_IPR / R * L / (2 * np.pi)))
    I_lat = np.array([I_K_shells(np.r_[0., np.interp(k_ref, *lattice_snapshot(sh, q, L))[:]], keff, n_hi) for q in range(len(times))])
    I_lat = norm(I_lat)
    k_out, comb = surrogate_series(wd, td, g, lb, times)
    I_red = I_K_series_surrogate(k_out, comb, R, lb, L, keff, n_hi)
    I_red_raw = I_K_series_surrogate(k_out, surrogate_series(wr, td, g, lb, times)[1], R, lb, L, keff, n_hi)
    return dict(run=r, central=central, N_b=m["N_b"], L=L, R=R, gamma_star=m["gamma_star"], t_coll_end=t_coll_end, T_surr=T, t_band=(tband[0], tband[-1]),
                kR_sh=k_ref * R, y_med=norm(med), y_lo=norm(lo), y_hi=norm(hi), kR_rec=k_out * R, y_rec=y_rec,
                shell_mask=mask, rec_shells=Yn, t=times, I_lat=I_lat, I_red=I_red,
                y_rec_raw=y_rec_raw, rec_shells_raw=Yn_raw, I_red_raw=I_red_raw)


def build_nmany(central="median"):
    return {(r["ic"]): build_nmany_one(r, central) for r in NMANY}


# ----------------------------------------------------------------------------- gamma_* and N_b families (figs 6-8)
# gamma_* family: N_b = FAM_NB, gamma_* in FAM_GS; N_b family: gamma_* = NB_GS, N_b in NB_LIST; N_REAL placements each.
# Weights: cpp/weights with the measured wall-radius table (as every other run here), from t = 0 to 5% past the
# box-filling time. Surrogate: the gamma_* = 1-16 scans (wall_points 50, time_factor 2.5); damped weights extended to
# the scan's T_MAX. One cache per lambda_bar (fam_lb<tag>.pkl), built only once that lambda_bar's scan is complete.
FAM_SCANS = {LB845: "runtime_scan_lb0.844777_w50_tf2.5_gs1-16_kR1-100_nw32",
             0.069: "runtime_scan_lb0.069_w50_tf2.5_gs1-16_kR1-100_nw32"}
FAM_LBS = [LB845, 0.069]
FAM_GS = list(range(2, 17))
FAM_NB = 128
NB_GS = 16
NB_LIST = [4, 8, 16, 32, 64, 128, 256, 512]
DEC_NB = 512                          # fig 8: spectral decomposition of the (NB_GS, DEC_NB) case
N_REAL = 16
FAM_STEPS_PER_RSTAR = 450             # weights time resolution (as the N_b > 3 runs)
FAM_ROOT = DATA / "family_weights"
FAM_SEED = 160016
N_PLOT, X_PLOT = 200, np.geomspace(1., 100., 200)
GAMMA_STARS = FAM_GS


def fam_rstar(lb, gs):
    from ptbuilder.weight_scan_diagnostics import characteristic_scale
    return characteristic_scale(load_kinematics(DATA / f"phi4_lambda_bar{LB_TAG[lb]}" / "instanton.h5"), float(gs))[1]


def fam_cases(lb):
    """(gamma_*, N_b, realization) of both families; the (NB_GS, FAM_NB) case is shared."""
    pairs = sorted({(gs, FAM_NB) for gs in FAM_GS} | {(NB_GS, nb) for nb in NB_LIST})
    return [(gs, nb, rz) for gs, nb in pairs for rz in range(N_REAL)]


def fam_dir(lb, gs, nb, rz):
    return FAM_ROOT / f"lb{LB_TAG[lb]}" / f"gs{gs:02d}_nb{nb:04d}" / f"r{rz:02d}"


def _fam_weights_task(args):
    from ptbuilder.analysis import write_weights_input
    from ptbuilder.weight_scan_diagnostics import place_simultaneous_bubbles
    lb, gs, nb, rz = args
    d = fam_dir(lb, gs, nb, rz); win, wout = d / "weights_input.h5", d / "weights_output.h5"
    if wout.exists():
        return
    d.mkdir(parents=True, exist_ok=True)
    kin = load_kinematics(DATA / f"phi4_lambda_bar{LB_TAG[lb]}" / "instanton.h5")
    R = fam_rstar(lb, gs); L = R * nb ** (1 / 3)
    seed = int(FAM_SEED + gs * 1_000_003 + nb * 10_007 + rz * 100_000_003)
    P = place_simultaneous_bubbles(nb, L, seed, min_separation=2.0 * kin.profile.rmid_0)
    wall = DATA / f"wall_radius/lb{LB_TAG[lb]}.h5"
    with h5py.File(wall) as tab:
        s, Rm = tab["s"][:], tab["R_mid"][:]
    t_end = 1.05 * float(np.interp(_covering_radius(P, L), Rm, s))
    if t_end > s[-1]:
        raise RuntimeError(f"lb={lb} gamma*={gs} N_b={nb} r{rz}: box fills at t={t_end/1.05:.1f}, past the wall-radius table (s<={s[-1]:.1f})")
    t = np.linspace(0., t_end, int(np.ceil(t_end / (R / FAM_STEPS_PER_RSTAR))) + 1)
    write_weights_input(win, P, t, kin, L, collision_radius="mid")
    with h5py.File(win, "a") as f:
        f.attrs.update(dict(gamma_star=float(gs), R_star=R, N_b=nb, placement_seed=seed, realization=rz))
    subprocess.run([str(ROOT / "cpp/weights/weights"), str(win), str(wout), "--wall-radius", str(wall)],
                   check=True, capture_output=True, env=os.environ | {"OMP_NUM_THREADS": "1"})


def build_family_weights(lb, workers=8):
    """Write and run every family weights input for lambda_bar lb (skips cases already done)."""
    from concurrent.futures import ThreadPoolExecutor
    cases = [(lb, gs, nb, rz) for gs, nb, rz in sorted(fam_cases(lb), key=lambda c: -c[1] * c[0] ** 3)]
    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(_fam_weights_task, cases))


_FI = {}
def _fam_interp(lb, gs):
    """Scan interpolator on k = X_PLOT / R_*(gamma_*) (one per worker and (lb, gamma_*))."""
    if (lb, gs) not in _FI:
        sc = _FAM_SCAN.get(lb) or _FAM_SCAN.setdefault(lb, load_scan(DATA / FAM_SCANS[lb]))
        f, g = build_runtime_scan_interpolator(sc, X_PLOT / fam_rstar(lb, gs))
        _FI[(lb, gs)] = (f, g, np.array([min(r.times[0] for r in sc.rows), sc.t_max]), float(np.diff(g).mean()))
    return _FI[(lb, gs)]
_FAM_SCAN = {}


def _accumulate(f, tb, g_ij, w_row, t, h_t=None, start=None):
    """One pair: sum_i w(t_i) [S(g, t_{i+1}) - S(g, t_i)] over t in the scan's range (reconstruct_pair_spectrum's
    sum, restricted to the pair's nonzero window); optionally adds each step's contribution into h_t[step]."""
    nz = np.where(w_row > 0)[0]
    if len(nz) == 0:
        return np.zeros(N_PLOT)
    lo, hi = nz[0], min(nz[-1] + 2, len(t)); t_hi = min(t[-1], tb[1])
    tw, ww = t[lo:hi], w_row[lo:hi]; m = (tw >= tb[0]) & (tw <= t_hi)
    tc, wc = tw[m], ww[m]; i0 = lo + (int(np.argmax(m)) if m.any() else 0)
    app = len(tc) == 0 or tc[-1] < t_hi
    if app:
        tc = np.append(tc, t_hi); wc = np.append(wc, np.interp(t_hi, t, w_row))
    if len(tc) < 2:
        return np.zeros(N_PLOT)
    sp = f(np.column_stack([np.full(len(tc), float(g_ij)), tc])); wt = wc[:-1, None] * np.diff(sp, axis=0)
    if h_t is not None:
        n = len(wt) - 1 if app else len(wt); np.add.at(h_t, np.arange(i0, i0 + n), wt[:n])
    return wt.sum(axis=0)


def _fam_task(args):
    lb, gs, nb, rz = args
    f, sg, tb, dg = _fam_interp(lb, gs)
    d = fam_dir(lb, gs, nb, rz)
    with h5py.File(d / "weights_input.h5") as fh:
        t = fh["t"][:]; R = float(fh.attrs["R_star"]); L = float(fh.attrs["L"])
    w, _, _, g = load_weights(d / "weights_output.h5", len(t))
    wd, td = damp(w, t, g, lb, t_end=tb[1], spacing=R / 150.)
    wr = np.array([np.interp(td, t, row, left=0., right=0.) for row in w])
    out = dict(R=R, L=L, undamped=sum(_accumulate(f, tb, gg, row, td) for gg, row in zip(g, wr)))
    if gs == NB_GS and nb == DEC_NB:              # fig 8: per time step and per gamma_ij bin
        edges_g = np.arange(sg.min() - dg / 2, 3.5 * gs, dg)
        h_t = np.zeros((len(td) - 1, N_PLOT)); h_g = np.zeros((len(edges_g) - 1, N_PLOT)); spec = np.zeros(N_PLOT)
        for gg, row in zip(g, wd):
            c = _accumulate(f, tb, gg, row, td, h_t); spec += c
            b = int(np.digitize([gg], edges_g)[0]) - 1
            if 0 <= b < len(h_g): h_g[b] += c
        # common time bins (width R_*/150 up to the scan's T_MAX) so the realizations can be summed
        edges_t = np.linspace(0., tb[1], int(np.ceil(tb[1] / (R / 150.))) + 1)
        H_t = np.zeros((len(edges_t) - 1, N_PLOT))
        np.add.at(H_t, np.clip(np.digitize(0.5 * (td[:-1] + td[1:]), edges_t) - 1, 0, len(edges_t) - 2), h_t)
        out.update(damped=spec, dec=dict(h_t=H_t, h_g=h_g, t=edges_t, edges_g=edges_g))
    else:
        out["damped"] = sum(_accumulate(f, tb, gg, row, td) for gg, row in zip(g, wd))
    if nb == FAM_NB:                               # fig 7: pair gamma_ij, integrated weight, last collision time
        h = dict(gamma=g)
        for name, ww, tt, thr in (("raw", w, t, 0.), ("damp", wd, td, 0.01)):
            on = ww > thr * ww.max(axis=1, keepdims=True) if thr > 0 else ww > 0
            h[f"W_{name}"] = np.trapz(ww, tt, axis=1)
            h[f"tmax_{name}"] = np.where(on.any(axis=1), tt[len(tt) - 1 - np.argmax(on[:, ::-1], axis=1)], np.nan) / R
        out["hist"] = h
    return (gs, nb, rz), out


def fam_scan_complete(lb):
    """True once every row of lb's family scan has all its output times."""
    root = DATA / FAM_SCANS[lb]
    if not (root / "setups").exists():
        return False
    for s in (root / "setups").glob("*.h5"):
        with h5py.File(s) as fh:
            n_t = len(fh["times"])
        if len(list((root / "gpu" / s.stem).glob("result_*.h5"))) < n_t:
            return False
    return True


def _fam_task_saved(args):
    """_fam_task with its result stored per case (CACHE/fam_parts/...), so an interrupted build resumes."""
    lb, gs, nb, rz = args
    part = CACHE / "fam_parts" / f"lb{LB_TAG[lb]}" / f"gs{gs:02d}_nb{nb:04d}_r{rz:02d}.pkl"
    if part.exists():
        return (gs, nb, rz), pickle.load(open(part, "rb"))
    key, out = _fam_task(args)
    part.parent.mkdir(parents=True, exist_ok=True); pickle.dump(out, open(part, "wb"))
    return key, out


def build_family(lb, workers=8):
    from multiprocessing import Pool
    build_family_weights(lb, workers)
    res = {}
    with Pool(workers) as pool:
        for key, out in pool.imap_unordered(_fam_task_saved, [(lb, *c) for c in sorted(fam_cases(lb), key=lambda c: -c[1])]):
            res[key] = out
    gsf = {gs: (res[(gs, FAM_NB, 0)]["R"], res[(gs, FAM_NB, 0)]["L"],
                {k: np.array([res[(gs, FAM_NB, rz)][k] for rz in range(N_REAL)]) for k in ("damped", "undamped")})
           for gs in FAM_GS}
    nbf = {(nb, rz): (res[(NB_GS, nb, rz)]["L"], {k: res[(NB_GS, nb, rz)][k] for k in ("damped", "undamped")})
           for nb in NB_LIST for rz in range(N_REAL)}
    hist = {gs: {k: np.concatenate([res[(gs, FAM_NB, rz)]["hist"][k] for rz in range(N_REAL)])
                 for k in res[(gs, FAM_NB, 0)]["hist"]} | dict(R=res[(gs, FAM_NB, 0)]["R"]) for gs in FAM_GS}
    decs = [res[(NB_GS, DEC_NB, rz)] for rz in range(N_REAL)]
    dec = dict(h_t=sum(x["dec"]["h_t"] for x in decs), h_g=sum(x["dec"]["h_g"] for x in decs),
               t=decs[0]["dec"]["t"], edges_g=decs[0]["dec"]["edges_g"], spec=np.array([x["damped"] for x in decs]),
               R=decs[0]["R"], L=decs[0]["L"], n_real=N_REAL, gamma_star=NB_GS)
    return dict(gs=gsf, nb=nbf, hist=hist, dec=dec, R_nb=fam_rstar(lb, NB_GS))


def family(lb, rebuild=False):
    return cached(f"fam_lb{LB_TAG[lb]}", lambda: build_family(lb), rebuild)


def fam_available():
    """lambda_bar values whose family cache exists or whose scan is complete (in FAM_LBS order)."""
    return [lb for lb in FAM_LBS if (CACHE / f"fam_lb{LB_TAG[lb]}.pkl").exists() or fam_scan_complete(lb)]


def load_families(rebuild=False):
    """Fig 6: gs[(lb, gamma_*)] = (R_*, L, {'damped','undamped': (N_REAL, N_PLOT)}), nb[(lb, N_b, r)] = (L, {...})."""
    gs, nb = {}, {}
    for lb in fam_available():
        F = family(lb, rebuild)
        gs |= {(lb, k): v for k, v in F["gs"].items()}; nb |= {(lb, *k): v for k, v in F["nb"].items()}
    return gs, nb


def load_weight_hists(rebuild=False):
    """Fig 7: {(lb, gamma_*): pooled pair gamma_ij, integrated weight, last collision time (raw and damped)}."""
    return {(lb, gs): h for lb in fam_available() for gs, h in family(lb, rebuild)["hist"].items()}


def load_decomposition(rebuild=False):
    """Fig 8: {lb: per-time-step and per-gamma_ij power of the (NB_GS, DEC_NB) case, summed over realizations}."""
    return {lb: family(lb, rebuild)["dec"] for lb in fam_available()}


# ----------------------------------------------------------------------------- per-pair spectral power (fig 7, power-weighted)
def _pair_power_task(args):
    """Damped surrogate power of every pair of one gamma_*-family realization: integrated over ln k (kR_* = 1-100)
    and at the peak of the realization's summed spectrum. Same pair order as the weights file (and the fig 7 hist)."""
    lb, gs, nb, rz = args
    part = CACHE / "fam_pairpow" / f"lb{LB_TAG[lb]}" / f"gs{gs:02d}_nb{nb:04d}_r{rz:02d}.pkl"
    if part.exists():
        return (gs, rz), pickle.load(open(part, "rb"))
    f, sg, tb, dg = _fam_interp(lb, gs)
    d = fam_dir(lb, gs, nb, rz)
    with h5py.File(d / "weights_input.h5") as fh:
        t = fh["t"][:]; R = float(fh.attrs["R_star"])
    w, _, _, g = load_weights(d / "weights_output.h5", len(t))
    wd, td = damp(w, t, g, lb, t_end=tb[1], spacing=R / 150.)
    C = np.array([_accumulate(f, tb, gg, row, td) for gg, row in zip(g, wd)])        # (n_pairs, N_PLOT)
    ipk = int(np.argmax(C.sum(axis=0)))
    out = dict(P_int=np.trapz(C, np.log(X_PLOT), axis=1), P_peak=C[:, ipk], kR_peak=float(X_PLOT[ipk]), gamma=g)
    part.parent.mkdir(parents=True, exist_ok=True); pickle.dump(out, open(part, "wb"))
    return (gs, rz), out


def build_pair_power(lb, workers=8):
    """{gamma_*: pooled per-pair damped power over the N_REAL realizations of the gamma_* family}."""
    from multiprocessing import Pool
    res = {}
    with Pool(workers) as pool:
        for key, out in pool.imap_unordered(_pair_power_task, [(lb, gs, FAM_NB, rz) for gs in FAM_GS[::-1] for rz in range(N_REAL)]):
            res[key] = out
    return {gs: {k: np.concatenate([res[(gs, rz)][k] for rz in range(N_REAL)]) for k in ("P_int", "P_peak", "gamma")}
            for gs in FAM_GS}


def pair_power(lb, rebuild=False):
    return cached(f"fam_pairpow_lb{LB_TAG[lb]}", lambda: build_pair_power(lb), rebuild)


# ----------------------------------------------------------------------------- surrogate for the Cutting et al. 2020 setup (comparison figure)
CUT_GS, CUT_NB, CUT_T, CUT_NREAL = 4, 512, 4.0, 16   # their Fig. 10(d): gamma_* = 4, N_b = 512, t/R_* = 4


def _cutting_task(args):
    """Damped surrogate spectrum at t = CUT_T R_* for one random placement of CUT_NB bubbles at gamma_* = CUT_GS."""
    lb, rz = args
    part = CACHE / "cutting_parts" / f"lb{LB_TAG[lb]}_r{rz:02d}.pkl"
    if part.exists():
        return rz, pickle.load(open(part, "rb"))
    _fam_weights_task((lb, CUT_GS, CUT_NB, rz))
    f, sg, tb, dg = _fam_interp(lb, CUT_GS)
    d = fam_dir(lb, CUT_GS, CUT_NB, rz)
    with h5py.File(d / "weights_input.h5") as fh:
        t = fh["t"][:]; R = float(fh.attrs["R_star"]); L = float(fh.attrs["L"])
    w, _, _, g = load_weights(d / "weights_output.h5", len(t))
    T = min(CUT_T * R, tb[1])
    wd, td = damp(w, t, g, lb, t_end=T, spacing=R / 150.)
    tbT = np.array([tb[0], T])
    out = dict(R=R, L=L, T=T, damped=sum(_accumulate(f, tbT, gg, row, td) for gg, row in zip(g, wd)))
    part.parent.mkdir(parents=True, exist_ok=True); pickle.dump(out, open(part, "wb"))
    return rz, out


def build_cutting_surrogate(lb, workers=8):
    from multiprocessing import Pool
    with Pool(workers) as pool:
        res = dict(pool.map(_cutting_task, [(lb, rz) for rz in range(CUT_NREAL)]))
    ens = np.array([to_chw(res[rz]["damped"], res[rz]["R"], lb, res[rz]["L"]**3) for rz in range(CUT_NREAL)])
    return dict(kR=X_PLOT, ens=ens, T_over_R=res[0]["T"] / res[0]["R"])


def cutting_surrogate(lb, rebuild=False):
    return cached(f"cutting_surr_lb{LB_TAG[lb]}", lambda: build_cutting_surrogate(lb), rebuild)


# ----------------------------------------------------------------------------- weights figure: undamped and damped W_ij(t) of three example cases
def build_weights_examples(many_ic, fam_case):
    """W_ij(t) raw and damped (as in the damped spectra) for the N_b=3 run (0.84), one N_b>3 lattice run and one family case."""
    c = N23[0.84]; _, wb = c["trio"]
    with h5py.File(DATA / f"weights_in_{wb}.h5") as f:
        t = f["t"][:]
    w, _, _, g = load_weights(DATA / f"weights_out_{wb}_rnum.h5", len(t))
    R = float(np.mean([_pair_sep(c, i, j) for (i, j), _, _ in c["pairs"]]))
    wd, td = damp(w, t, g, 0.84)
    out = dict(n3=dict(t=t, w=w, g=g, R=R, td=td, wd=wd))
    r = [r for r in NMANY if r["ic"] == many_ic][0]
    w, t, g, t_ext = nmany_weights(r); R = run_meta(r)["R"]
    wd, td = damp(w, t, g, r["lb"], t_end=t_ext, spacing=(R / 150. if t_ext else None))
    out["many"] = dict(t=t, w=w, g=g, R=R, td=td, wd=wd)
    lb, gs, nb, rz = fam_case; d = fam_dir(lb, gs, nb, rz)
    with h5py.File(d / "weights_input.h5") as f:
        t = f["t"][:]; R = float(f.attrs["R_star"])
    w, _, _, g = load_weights(d / "weights_output.h5", len(t))
    t_max = (_FAM_SCAN.get(lb) or _FAM_SCAN.setdefault(lb, load_scan(DATA / FAM_SCANS[lb]))).t_max
    wd, td = damp(w, t, g, lb, t_end=t_max, spacing=R / 150.)
    out["fam"] = dict(t=t, w=w, g=g, R=R, td=td, wd=wd)
    return out


def weights_examples(many_ic, fam_case, rebuild=False):
    lb, gs, nb, rz = fam_case
    return cached(f"weights_examples_{many_ic}_lb{LB_TAG[lb]}_gs{gs}_nb{nb}_r{rz}",
                  lambda: build_weights_examples(many_ic, fam_case), rebuild)
