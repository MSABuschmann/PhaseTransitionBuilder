"""
Measured patch-fraction damping of pair weights.

Optional post-processing of each pair's scalar collision weight w(t) before
it enters reconstruct_pair_spectrum: whenever part of the weight is lost at
time u, that part keeps contributing F((t - u)/d_pair)**alpha, where F is a
measured patch-fraction table (gamma=4 calibration, one per lambda_bar).

Ported from reference_codes/patch_fraction_implementation_handoff (see its
patch_fraction_implementation.pdf for the prescription and its caveats):
response/load_response/effective_weights are the handoff's portable
reference, unchanged. The reconstruction itself (signed left-endpoint power
increments) is the existing reconstruct_pair_spectrum.
"""
import json
from pathlib import Path

import numpy as np

DEFAULT_CALIBRATION = Path(__file__).parent.parent / "data" / "patch_fraction_calibration.json"


def response(age_knots, fractions, alpha=2, tail="zero"):
    """Interpolate raw F in laboratory age/d, then take F**alpha."""
    x = np.asarray(age_knots, dtype=float)
    y = np.asarray(fractions, dtype=float)
    if (x.ndim != 1 or len(x) < 2 or y.shape != x.shape
            or not np.isfinite(x).all() or not np.isfinite(y).all()
            or x[0] != 0 or np.any(np.diff(x) <= 0)
            or y[0] != 1 or np.any((y < 0) | (y > 1))):
        raise ValueError("Expected finite increasing ages and bounded raw F with F(0)=1")
    if alpha not in (1, 2) or tail not in ("zero", "terminal"):
        raise ValueError("Specify alpha=1 or 2 and tail='zero' or 'terminal'")

    def evaluate(age):
        age = np.asarray(age, dtype=float)
        if not np.isfinite(age).all() or np.any(age < -1e-12):
            raise ValueError("Nonfinite or noncausal laboratory age")
        raw = np.interp(np.maximum(age, 0), x, y,
                        right=0.0 if tail == "zero" else y[-1])
        return raw**alpha

    return evaluate


def load_response(lambda_bar, geometry="fixed_annulus", alpha=2, tail="zero",
                  path=DEFAULT_CALIBRATION):
    """Load a gamma=4 template; runtime gamma interpolation is NOT implemented."""
    payload = json.loads(Path(path).read_text())
    if (payload.get("schema") != 1
            or payload.get("coordinate") != "lab_age_over_d"
            or payload.get("gamma_policy") != "gamma4_similarity_at_actual_pair_d"):
        raise ValueError("Wrong calibration convention")
    if geometry not in ("fixed_annulus", "fixed_rapidity"):
        raise ValueError("Unknown cohort geometry")
    rows = [row for row in payload["cases"]
            if abs(row["lambda_bar"] - lambda_bar) < 1e-12]
    if len(rows) != 1 or rows[0]["gamma"] != 4.0:
        raise ValueError("Require one supplied potential case calibrated at gamma=4")
    row = rows[0]
    return response(row["age_d"], row["kernels"][geometry]["K"], alpha, tail)


def effective_weights(weights, times, d_pair, kernel):
    """Exact sampled-level memory; kernel=None returns the original weights."""
    w, t = np.asarray(weights, dtype=float), np.asarray(times, dtype=float)
    if (w.ndim != 1 or t.shape != w.shape or len(t) < 2
            or not np.isfinite(w).all() or not np.isfinite(t).all()
            or np.any(np.diff(t) <= 0) or np.any((w < 0) | (w > 1))):
        raise ValueError("Invalid sampled weight/time history")
    if not np.isfinite(d_pair) or d_pair <= 0:
        raise ValueError("Target pair separation must be finite and positive")
    if kernel is None:
        return w.copy()
    edges = np.unique(np.r_[0., w, 1.])
    widths = np.diff(edges)
    levels = (edges[:-1] + edges[1:]) / 2
    active = levels < w[0]
    last_off = np.full(len(levels), np.nan)
    out = w.copy()
    for n in range(1, len(t)):
        now = levels < w[n]
        last_off[active & ~now] = t[n]
        last_off[now] = np.nan
        active = now
        keep = np.isfinite(last_off)
        if np.any(keep):
            survival = np.asarray(kernel((t[n] - last_off[keep]) / d_pair))
            if (survival.shape != last_off[keep].shape
                    or not np.isfinite(survival).all()
                    or np.any((survival < 0) | (survival > 1))):
                raise ValueError("Kernel is incompatible with this bounded pilot")
            out[n] += np.dot(widths[keep], survival)
    if np.any(out < w - 2e-14) or np.any(out > np.maximum.accumulate(w) + 2e-14):
        raise AssertionError("Sampled-level bounds failed")
    return out


def pair_separation(gamma, rin_0, rout_0, rmid_0):
    """
    Invert the weights binary's width-defined gamma_ij back to the pair
    separation d (textbook radii, mid-radius collision convention), with the
    round-trip check the handoff requires.
    """
    g = float(gamma)
    ric = 0.5 * (g * (rout_0 + rin_0) - (rout_0 - rin_0) / g)
    sc2 = ric**2 - rin_0**2
    if sc2 < -1e-9 * rin_0**2:
        raise ValueError(f"gamma={g} gives negative s_c^2={sc2}")
    sc2 = max(sc2, 0.)
    back = (rout_0 - rin_0) / (np.sqrt(rout_0**2 + sc2) - np.sqrt(rin_0**2 + sc2))
    if not np.isclose(back, g, rtol=1e-9, atol=1e-12):
        raise AssertionError(f"gamma round trip failed: {g} -> {back}")
    return 2. * np.sqrt(sc2 + rmid_0**2)


def refined_grid(times, t_end, refine=2, extension_spacing=None):
    """
    Weights grid truncated or extended to t_end, with refine-1 points
    inserted per interval. An extension past the last node continues at the
    grid's own (uniform) spacing, or at extension_spacing, which is required
    for a nonuniform grid; t_end itself is included exactly once.
    """
    t = np.asarray(times, dtype=float)
    if t_end > t[-1]:
        if extension_spacing is None:
            dt = np.diff(t)
            if not np.allclose(dt, dt[0], rtol=1e-9, atol=0.):
                raise ValueError("nonuniform weights grid: pass extension_spacing to extend it")
            h = dt[0]
        else:
            h = float(extension_spacing)
            if not h > 0:
                raise ValueError("extension_spacing must be positive")
        n = int(np.floor((t_end - t[-1]) / h))
        t = np.r_[t, t[-1] + h * np.arange(1, n + 1)]
    tol = 1e-9 * np.min(np.diff(t)) if len(t) > 1 else 0.
    t = np.r_[t[t < t_end - tol], t_end]
    if refine > 1:
        frac = np.arange(refine) / refine
        t = np.r_[(t[:-1, None] + np.diff(t)[:, None] * frac).ravel(), t[-1]]
    return t


def damped_weights(weights, weights_times, gammas, profile, kernel, t_end=None, refine=1):
    """
    Apply effective_weights to every pair of a weights file.

    weights: (n_pairs, n_t) from load_weights; gammas: their gamma_ij;
    profile: instanton (rin_0, rout_0, rmid_0). kernel: a response, or a
    load_gamma_response(...) result (picked per pair by its gamma_ij).
    Extending past the weights file (t_end > last time) pads with zero
    weights, allowed only when every pair's final weight is already zero.
    Returns (w_eff, times) on the
    (optionally refined, truncated at t_end) grid; the native weights are
    linearly interpolated onto it first, as in the handoff's pilot.
    """
    if t_end is not None and t_end > weights_times[-1] and np.any(np.asarray(weights)[:, -1] > 0):
        raise ValueError("cannot extend past the weights file with zero weights: some pairs are "
                         "still active at its last time; recompute the weights further out")
    t = refined_grid(weights_times, weights_times[-1] if t_end is None else t_end, refine)
    w = np.array([np.interp(t, weights_times, row, left=0., right=0.) for row in weights])
    out = np.empty_like(w)
    for p, (row, g) in enumerate(zip(w, gammas)):
        d = pair_separation(g, profile.rin_0, profile.rout_0, profile.rmid_0)
        k = kernel(g) if getattr(kernel, "per_gamma", False) else kernel
        out[p] = effective_weights(row, t, d, k)
    return out, t


# ---------------------------------------------------------------------------
# Calibration from bubblemaster --patch-moments output
# ---------------------------------------------------------------------------

def _linear_moments(x, y, a, b):
    """
    Exact integrals over [a, b] of s*f(s) and f(s)/s for the
    piecewise-linear interpolant f through (x, y): returns
    (int s f ds, int f/s ds).
    """
    if b <= a:
        return 0., 0.
    k = np.r_[a, x[(x > a) & (x < b)], b]
    f = np.interp(k, x, y)
    s0, s1, f0, f1 = k[:-1], k[1:], f[:-1], f[1:]
    beta = (f1 - f0) / (s1 - s0)
    alpha = f0 - beta * s0
    int_sf = np.sum(alpha * (s1**2 - s0**2) / 2 + beta * (s1**3 - s0**3) / 3)
    int_f_over_s = np.sum(alpha * np.log(s1 / s0) + beta * (s1 - s0))
    return int_sf, int_f_over_s


def region_energy(t, rho_a, rho_b, sA, A, sB, B):
    """
    Spatial gradient energy of the collision region with transverse radii
    rho_a < rho < rho_b at lab time t:
      E = pi * int_{s-}^{s+} [ s A(s) + (t^2/s - s) B(s) ] ds,
    s- = sqrt(t^2 - rho_b^2), s+ = sqrt(t^2 - rho_a^2); A, B piecewise linear.
    """
    s_lo, s_hi = np.sqrt(t**2 - rho_b**2), np.sqrt(t**2 - rho_a**2)
    a_sf, _ = _linear_moments(sA, A, s_lo, s_hi)
    b_sf, b_fs = _linear_moments(sB, B, s_lo, s_hi)
    return np.pi * (a_sf + t**2 * b_fs - b_sf)


def calibrate_from_moments(path, d, n_ages=2049, t_end=None):
    """
    Patch-fraction tables K(age/d) for both cohort geometries from a
    patch_moments.h5 file (s_c is read from the file). Cohort: original
    contact times 1.2 s_c < t_c < 1.3 s_c, tracked from t0 = 1.3 s_c.
    """
    import h5py
    with h5py.File(path, "r") as f:
        s_c = float(f.attrs["s_c"])
        sA, A, sB, B = f["s_A"][:], f["A"][:], f["s_B"][:], f["B"][:]
    t0 = 1.3 * s_c
    t_max = min(sA[-1], sB[-1]) if t_end is None else t_end
    ages = np.linspace(0., (t_max - t0) / d, n_ages)
    rho_a0, rho_b0 = s_c * np.sqrt(1.2**2 - 1), s_c * np.sqrt(1.3**2 - 1)
    out = {"fixed_annulus": [], "fixed_rapidity": []}
    for a in ages:
        t = t0 + a * d
        full = region_energy(t, 0., np.sqrt(t**2 - s_c**2), sA, A, sB, B)
        out["fixed_annulus"].append(region_energy(t, rho_a0, rho_b0, sA, A, sB, B) / full)
        out["fixed_rapidity"].append(region_energy(t, rho_a0 * t / t0, rho_b0 * t / t0, sA, A, sB, B) / full)
    K = {g: np.array(v) / v[0] for g, v in out.items()}
    return ages, K, dict(s_c=s_c, t0=t0, t_max=t_max)


# ---------------------------------------------------------------------------
# Multi-gamma calibration: F interpolated in gamma_ij at each age, then **alpha
# ---------------------------------------------------------------------------

def write_calibration(path, cases, convention):
    """
    cases: list of dicts with lambda_bar, gamma, d, s_c, delta_c, t0, ages,
    K (dict geometry -> array). Written in the handoff's coordinate
    (age/d, raw F, F(0)=1), one entry per (lambda_bar, gamma), with F capped at 1.
    """
    payload = dict(schema=2, coordinate="lab_age_over_d", gamma_policy="interpolate_F_in_gamma",
                   convention=convention, cases=[])
    for c in cases:
        payload["cases"].append(dict(
            lambda_bar=float(c["lambda_bar"]), gamma=float(c["gamma"]), d=float(c["d"]),
            s_c=float(c["s_c"]), delta_c=float(c["delta_c"]), t0=float(c["t0"]),
            age_d=[float(a) for a in c["ages"]],
            # F capped at 1 (the level bookkeeping requires F <= 1); the
            # uncapped maximum is recorded -- low-gamma cohorts briefly gain
            # energy share just after t0.
            kernels={g: {"K": [float(min(v, 1.)) for v in k], "max_uncapped": float(np.max(k))}
                     for g, k in c["K"].items()}))
    Path(path).write_text(json.dumps(payload))


def load_gamma_response(path, lambda_bar, geometry="fixed_annulus", alpha=2, tail="zero"):
    """
    Returns kernel_for(gamma): the response F**alpha for a pair with this
    gamma_ij, F linearly interpolated in gamma between the two nearest
    calibrated gammas (each with its own support and tail policy), clamped to
    the calibrated range.
    """
    payload = json.loads(Path(path).read_text())
    if payload.get("schema") != 2 or payload.get("coordinate") != "lab_age_over_d":
        raise ValueError("Wrong calibration convention")
    rows = sorted((r for r in payload["cases"] if abs(r["lambda_bar"] - lambda_bar) < 1e-12),
                  key=lambda r: r["gamma"])
    if not rows:
        raise ValueError(f"No calibration for lambda_bar={lambda_bar}")
    raw = [response(r["age_d"], r["kernels"][geometry]["K"], alpha=1, tail=tail) for r in rows]
    gammas = np.array([r["gamma"] for r in rows])

    def kernel_for(gamma):
        g = float(np.clip(gamma, gammas[0], gammas[-1]))
        j = int(np.clip(np.searchsorted(gammas, g) - 1, 0, len(gammas) - 2)) if len(gammas) > 1 else 0
        if len(gammas) == 1:
            return lambda age: raw[0](age)**alpha
        w = (g - gammas[j]) / (gammas[j+1] - gammas[j])
        return lambda age: ((1 - w) * raw[j](age) + w * raw[j+1](age))**alpha

    kernel_for.per_gamma = True
    return kernel_for
