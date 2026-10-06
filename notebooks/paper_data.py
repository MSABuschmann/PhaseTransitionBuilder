"""Data for the paper figures (notebook 29_paper_figures.ipynb).

Every figure's numbers are built once by a `build_*` function and stored in
data/paper_cache/<name>.pkl; `cached(name, builder)` only rebuilds when the
file is missing (or rebuild=True). The notebook itself only loads and plots.

Conventions (agreed for the paper):
  * lattice spectra: (A+B)/2 of sledgehamr's gw_spectra and gw_spec_u_times_k.
  * surrogate: pair spectra at wall_points=100 (notebook 30): N_b=2,3 from dedicated runs, one per lattice pair at its
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

SCANS = {0.069: "runtime_scan_lb0.069_gs1-6_kR1-100_nw32",          # N_b > 3: wall_points 100, time_factor 2.5 (notebook 30)
         LB845: "runtime_scan_lb0.844777_gs1-6_kR1-100_nw32"}
N23_RUNS = {0.84: "n2_dedicated_lb0.84_w100",                      # N_b = 2, 3: one row per lattice pair at its exact gamma_ij,
            0.069: "n2_dedicated_lb0.069_w100"}                     # with t_m as an output time (notebook 30)
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
    dict(lb=LB845, ic="Cutting_lb0.845_N64_g3.94", sh="Cutting_lb0.84_N64_g3.94_long", panel=4, col="C0"),
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
    m = 0.35 * L; imgs = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                Q = P + L * np.array([dx, dy, dz]); imgs.append(Q[np.all((Q > -m) & (Q < L + m), axis=1)])
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
    """Median / min / max over snapshots t >= t_from (raw units, interpolated onto the last snapshot's k)."""
    late = np.where(times >= t_from)[0]
    k_ref, _ = lattice_snapshot(out, late[-1], L)
    S = np.array([np.interp(k_ref, *lattice_snapshot(out, q, L), left=np.nan, right=np.nan) for q in late])
    return k_ref, np.nanmedian(S, axis=0), np.nanmin(S, axis=0), np.nanmax(S, axis=0), times[late]


def build_nmany_one(r):
    m = run_meta(r); R, L, lb = m["R"], m["L"], r["lb"]
    w, t, g, t_ext = nmany_weights(r)
    t_coll_end = float(t[np.where(w.sum(axis=0) > 0.01)[0][-1]])
    wd, td = damp(w, t, g, lb, t_end=t_ext, spacing=(R / 150. if t_ext else None))
    wr = np.array([np.interp(td, t, row, left=0., right=0.) for row in w])     # undamped, on the same grid
    sh = load_averaged(r["sh"]); times = np.array(sh.GetTimesOfGravitationalWaveSpectra())
    # Fig 4: late-time lattice band vs surrogate at min(weights end, last snapshot)
    T = min(float(td[-1]), float(times[-1]))
    k_ref, med, lo, hi, tband = late_time_band(sh, times, t_coll_end, L)
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
    return dict(run=r, N_b=m["N_b"], L=L, R=R, gamma_star=m["gamma_star"], t_coll_end=t_coll_end, T_surr=T, t_band=(tband[0], tband[-1]),
                kR_sh=k_ref * R, y_med=norm(med), y_lo=norm(lo), y_hi=norm(hi), kR_rec=k_out * R, y_rec=y_rec,
                shell_mask=mask, rec_shells=Yn, t=times, I_lat=I_lat, I_red=I_red,
                y_rec_raw=y_rec_raw, rec_shells_raw=Yn_raw, I_red_raw=I_red_raw)


def build_nmany():
    return {(r["ic"]): build_nmany_one(r) for r in NMANY}


# ----------------------------------------------------------------------------- gamma_* and N_b families (fig 6)
def rho_chw(k, dE, R, lb, V):
    return to_chw(dE, R, lb, V)


def load_families():
    """The cached damped/undamped reconstructions of notebooks 20 and 23 (see data/*D36_and_undamped*.pkl)."""
    gs = pickle.load(open(DATA / "gamma_star_N128_ens16_D36_and_undamped_reconstructed.pkl", "rb"))
    nb = pickle.load(open(DATA / "n_b_family_gstar20_ens16_D36_and_undamped_reconstructed.pkl", "rb"))
    return gs, nb


# ----------------------------------------------------------------------------- weight histograms (fig 7)
FAMILY_ROOT = {0.84: DATA / "gamma_star_weights_lb0p84_N128_ens16", 0.069: DATA / "gamma_star_weights_lb0p069_N128_ens16"}
GAMMA_STARS = list(range(2, 21))


def _hist_task(args):
    """Per (lambda_bar, gamma_*): pair gamma_ij, integrated weight and last collision time, raw and D^3.6-damped,
    pooled over the 16 realizations. Damped 'last time' = last time the pair's damped weight is >= 1% of its maximum
    (damped weights never return exactly to zero)."""
    lb, gs = args
    gdir = FAMILY_ROOT[lb] / f"gamma_star_{gs:03d}"
    with h5py.File(gdir / "realization_00" / "weights_input.h5") as f: R = float(f.attrs["R_star"])
    rec = {k: [] for k in ("gamma", "W_raw", "tmax_raw", "W_damp", "tmax_damp")}
    for rz in range(16):
        rdir = gdir / f"realization_{rz:02d}"
        with h5py.File(rdir / "weights_input.h5") as f: t = f["t"][:]
        w, _, _, g = load_weights(rdir / "weights_output.h5", len(t))
        wd, _ = damp(w, t, g, lb)
        for name, ww, thr in (("raw", w, 0.), ("damp", wd, 0.01)):
            on = ww > thr * ww.max(axis=1, keepdims=True) if thr > 0 else ww > 0
            last = np.where(on.any(axis=1), t[len(t) - 1 - np.argmax(on[:, ::-1], axis=1)], np.nan)
            rec[f"W_{name}"].append(np.trapz(ww, t, axis=1)); rec[f"tmax_{name}"].append(last / R)
        rec["gamma"].append(g)
    return (lb, gs), {k: np.concatenate(v) for k, v in rec.items()} | dict(R=R)


def build_weight_hists():
    from multiprocessing import Pool
    with Pool(8) as pool:
        return dict(pool.map(_hist_task, [(lb, gs) for gs in GAMMA_STARS[::-1] for lb in (0.84, 0.069)]))


# ----------------------------------------------------------------------------- spectral decomposition (fig 8)
N_PLOT, X_PLOT = 200, np.geomspace(1., 100., 200)
NB_ROOT = {0.84: DATA / "n_b_family_lb0p84_gstar20_ens16", 0.069: DATA / "n_b_family_lb0p069_gstar20_ens16"}
SCAN20 = {0.84: DATA / "runtime_scan_lb0.84_gs1-20_kR1-100_nw32", 0.069: DATA / "runtime_scan_lb0.069_gs1-20_kR1-100_nw32"}
_DEC = {}


def _dec_setup(lb):
    """Notebook 23/25's (gamma_ij, t) cube interpolator on k = X_PLOT / R_* for the gamma_*=20, N_b=512 case."""
    from scipy.interpolate import interp1d, RegularGridInterpolator
    sc = load_scan(SCAN20[lb]); sg = np.array([r.gamma_ij for r in sc.rows])
    with h5py.File(NB_ROOT[lb] / "n_b_0512" / "realization_00" / "weights_input.h5") as f: R = float(f.attrs["R_star"]); L = float(f.attrs["L"])
    k_out = X_PLOT / R; NT = max(len(r.times) for r in sc.rows); tc = np.linspace(0., sc.t_max, NT)
    spec = [np.array([np.interp(k_out, r.wlist, r.spectrum[i], left=0., right=0.) for i in range(len(r.times))]) for r in sc.rows]
    floor = max(s.max() for s in spec) * 1e-12; lf = np.log(floor); cube = np.empty((len(sc.rows), NT, N_PLOT))
    for ir, (r, s) in enumerate(zip(sc.rows, spec)):
        ls = np.log(np.maximum(s, floor)); cube[ir] = interp1d(r.times, ls, axis=0, bounds_error=False, fill_value=(lf, ls[-1]))(tc)
    rgi = RegularGridInterpolator((sg, tc), cube, bounds_error=False, fill_value=lf)
    f = lambda pts: np.exp(rgi(np.column_stack([np.clip(pts[:, 0], sg.min(), sg.max()), pts[:, 1]])))
    return dict(f=f, sg=sg, tb=np.array([min(r.times[0] for r in sc.rows), sc.t_max]), R=R, L=L, dg=float(np.diff(sg).mean()))


def _dec_task(args):
    """One realization: per-time-step and per-gamma_ij contributions of every pair to the reconstructed spectrum
    (notebook 25's exact-index accumulation), with D^3.6-damped weights."""
    lb, rz = args
    if lb not in _DEC: _DEC[lb] = _dec_setup(lb)
    S = _DEC[lb]; rdir = NB_ROOT[lb] / "n_b_0512" / f"realization_{rz:02d}"
    with h5py.File(rdir / "weights_input.h5") as f: t = f["t"][:]
    w, _, _, g = load_weights(rdir / "weights_output.h5", len(t)); wd, _ = damp(w, t, g, lb)
    edges_g = np.arange(S["sg"].min() - S["dg"] / 2, 60.0, S["dg"])
    h_t = np.zeros((len(t) - 1, N_PLOT)); h_g = np.zeros((len(edges_g) - 1, N_PLOT)); spec = np.zeros(N_PLOT)
    t_hi = min(t[-1], S["tb"][1])
    for wr, gg in zip(wd, g):
        nz = np.where(wr > 0)[0]
        if len(nz) == 0: continue
        lo, hi = nz[0], min(nz[-1] + 2, len(t)); tw, ww = t[lo:hi], wr[lo:hi]
        msk = (tw >= S["tb"][0]) & (tw <= t_hi); tcmp, wcmp = tw[msk], ww[msk]; lo_eff = lo + (np.argmax(msk) if msk.any() else 0)
        app = len(tcmp) == 0 or tcmp[-1] < t_hi
        if app: tcmp = np.append(tcmp, t_hi); wcmp = np.append(wcmp, np.interp(t_hi, t, wr))
        if len(tcmp) < 2: continue
        sp = S["f"](np.column_stack([np.full(len(tcmp), float(gg)), tcmp])); wt = wcmp[:-1, None] * np.diff(sp, axis=0)
        spec += wt.sum(axis=0); n_clean = len(wt) - 1 if app else len(wt)
        np.add.at(h_t, np.arange(lo_eff, lo_eff + n_clean), wt[:n_clean])
        gb = int(np.digitize([gg], edges_g)[0]) - 1
        if 0 <= gb < len(edges_g) - 1: h_g[gb] += wt.sum(axis=0)
    return lb, h_t, h_g, spec, t, edges_g


def build_decomposition():
    from multiprocessing import Pool
    acc = {}
    with Pool(8) as pool:
        for lb, h_t, h_g, spec, t, eg in pool.imap_unordered(_dec_task, [(lb, rz) for lb in (0.84, 0.069) for rz in range(16)]):
            a = acc.setdefault(lb, dict(h_t=0., h_g=0., spec=[], t=t, edges_g=eg))
            a["h_t"] = a["h_t"] + h_t; a["h_g"] = a["h_g"] + h_g; a["spec"].append(spec)
    for lb, a in acc.items():
        with h5py.File(NB_ROOT[lb] / "n_b_0512" / "realization_00" / "weights_input.h5") as f: a["R"] = float(f.attrs["R_star"]); a["L"] = float(f.attrs["L"])
        a["n_real"] = len(a["spec"]); a["spec"] = np.array(a["spec"])
    return acc
