"""Diagnostics for choosing BubbleMaster gamma and time scan domains."""

from __future__ import annotations

import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
from scipy.optimize import brentq

from .analysis import load_weights, write_weights_input
from .physics import BubbleKinematics, InstantonProfile


def place_simultaneous_bubbles(n_b, box_size, seed, min_separation):
    """Periodic random sequential placement with a hard midpoint separation."""
    rng = np.random.default_rng(seed)
    positions = np.empty((n_b, 3), dtype=float)
    positions[0] = rng.uniform(0.0, box_size, 3)
    for i in range(1, n_b):
        for _ in range(1_000_000):
            proposal = rng.uniform(0.0, box_size, 3)
            delta = np.abs(positions[:i] - proposal)
            delta = np.minimum(delta, box_size - delta)
            if np.all(np.linalg.norm(delta, axis=1) > min_separation):
                positions[i] = proposal
                break
        else:
            raise RuntimeError(f"Could not place bubble {i} of {n_b}")
    return positions


def load_kinematics(instanton_path):
    with h5py.File(instanton_path, "r") as f:
        profile = InstantonProfile(
            f["R"][:], f["Phi"][:], float(f.attrs["rin_0"]),
            float(f.attrs["rmid_0"]), float(f.attrs["rout_0"]), None,
        )
    return BubbleKinematics(profile)


def characteristic_scale(kinematics, gamma_star):
    t_star = brentq(
        lambda t: float(kinematics.Gamma(t)) - gamma_star,
        0.0, 1.0e6,
    )
    r_star = 2.0 * float(kinematics.R(t_star, r="mid"))
    return t_star, r_star


def prepare_case(root, kinematics, gamma_star, n_b, n_t, max_t_over_rstar,
                 base_seed):
    _, r_star = characteristic_scale(kinematics, gamma_star)
    box_size = r_star * n_b ** (1.0 / 3.0)
    seed = int(base_seed + gamma_star * 1_000_003 + n_b)
    positions = place_simultaneous_bubbles(
        n_b, box_size, seed,
        min_separation=2.0 * kinematics.profile.rmid_0,
    )
    times = np.linspace(0.0, max_t_over_rstar * r_star, n_t)
    case_dir = root / f"gamma_star_{int(gamma_star):03d}"
    case_dir.mkdir(parents=True, exist_ok=True)
    input_path = case_dir / "weights_input.h5"
    output_path = case_dir / "weights_output.h5"
    write_weights_input(
        input_path, positions, times, kinematics, box_size,
        collision_radius="mid",
    )
    with h5py.File(input_path, "a") as f:
        f.attrs["gamma_star"] = float(gamma_star)
        f.attrs["R_star"] = r_star
        f.attrs["placement_seed"] = seed
    return input_path, output_path


def run_grid(repo_root, gamma_stars=range(2, 65), n_b=128, n_t=1024,
             max_t_over_rstar=2.5, base_seed=640128, workers=4,
             omp_threads=4, lambda_bar=0.84):
    repo_root = Path(repo_root)
    lb_tag = f"{lambda_bar:g}".replace(".", "p")
    root = repo_root / "data" / f"gamma_star_weights_lb{lb_tag}_N{n_b}"
    root.mkdir(parents=True, exist_ok=True)
    kin = load_kinematics(
        repo_root / f"data/phi4_lambda_bar{lambda_bar:g}/instanton.h5"
    )
    cases = [prepare_case(root, kin, float(g), n_b, n_t,
                          max_t_over_rstar, base_seed)
             for g in gamma_stars]

    binary = repo_root / "cpp/weights/weights"
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(omp_threads)

    def run_one(paths):
        inp, out = paths
        subprocess.run([str(binary), str(inp), str(out)], check=True,
                       env=env, stdout=subprocess.DEVNULL)
        return out

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_one, paths) for paths in cases]
        for i, future in enumerate(as_completed(futures), 1):
            print(f"[{i:02d}/{len(futures)}] {future.result().parent.name}",
                  flush=True)
    return collect_grid(root, kin, n_t)


def collect_grid(root, kinematics, n_t):
    root = Path(root)
    records = []
    for case_dir in sorted(root.glob("gamma_star_*")):
        inp = case_dir / "weights_input.h5"
        out = case_dir / "weights_output.h5"
        with h5py.File(inp, "r") as f:
            gamma_star = float(f.attrs["gamma_star"])
            r_star = float(f.attrs["R_star"])
            box_size = float(f.attrs["L"])
            times = f["t"][:]
            positions = np.column_stack(
                (f["xlocs"][:], f["ylocs"][:], f["zlocs"][:])
            )
        weights, pair_i, pair_j, pair_gamma = load_weights(out, n_t)
        contact_times = np.empty(len(pair_i))
        for p, (i, j) in enumerate(zip(pair_i, pair_j)):
            delta = positions[int(i)] - positions[int(j)]
            delta -= box_size * np.round(delta / box_size)
            contact_times[p] = kinematics.collision_time(
                float(np.linalg.norm(delta)), collision_radius="mid"
            )
        integrated_weight = np.trapz(weights, times, axis=1)
        nonzero = weights > 0.0
        has_nonzero = nonzero.any(axis=1)
        first_nonzero_index = np.argmax(nonzero, axis=1)
        last_nonzero_index = len(times) - 1 - np.argmax(nonzero[:, ::-1], axis=1)
        pair_first_nonzero = np.full(len(weights), np.nan)
        pair_last_nonzero = np.full(len(weights), np.nan)
        pair_first_nonzero[has_nonzero] = times[first_nonzero_index[has_nonzero]]
        pair_last_nonzero[has_nonzero] = times[last_nonzero_index[has_nonzero]]
        active = np.where(weights.sum(axis=0) > 1.0e-10)[0]
        records.append(dict(
            gamma_star=gamma_star, r_star=r_star,
            pair_gamma=pair_gamma, contact_times=contact_times,
            integrated_weight=integrated_weight,
            pair_first_nonzero=pair_first_nonzero,
            pair_last_nonzero=pair_last_nonzero,
            earliest_contact=float(contact_times.min()),
            latest_contact=float(contact_times.max()),
            last_active=float(times[active[-1]]) if len(active) else np.nan,
            reached_time_boundary=bool(len(active) and active[-1] == len(times)-1),
        ))
    save_diagnostics(root / "diagnostics.h5", records)
    return records


def save_diagnostics(path, records):
    with h5py.File(path, "w") as f:
        f.attrs["collision_radius"] = "mid"
        for record in records:
            group = f.create_group(f"gamma_star_{int(record['gamma_star']):03d}")
            for key in ("gamma_star", "r_star", "earliest_contact",
                        "latest_contact", "last_active",
                        "reached_time_boundary"):
                group.attrs[key] = record[key]
            for key in ("pair_gamma", "contact_times", "integrated_weight",
                        "pair_first_nonzero", "pair_last_nonzero"):
                group.create_dataset(key, data=record[key])


def plot_diagnostics(records, output_prefix):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    output_prefix = Path(output_prefix)
    stars = np.array([r["gamma_star"] for r in records])
    all_gamma = np.concatenate([r["pair_gamma"] for r in records])
    y_edges = np.linspace(0.0, max(65.0, all_gamma.max() * 1.03), 81)
    x_edges = np.arange(stars.min() - 0.5, stars.max() + 1.5)
    count = np.zeros((len(y_edges)-1, len(stars)))
    weighted = np.zeros_like(count)
    for ix, record in enumerate(records):
        count[:, ix], _ = np.histogram(record["pair_gamma"], bins=y_edges)
        weighted[:, ix], _ = np.histogram(
            record["pair_gamma"], bins=y_edges,
            weights=record["integrated_weight"],
        )
        if weighted[:, ix].sum() > 0:
            weighted[:, ix] /= weighted[:, ix].sum()

    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    positive = count[count > 0]
    pcm = axes[0].pcolormesh(
        x_edges, y_edges, count, shading="auto",
        norm=LogNorm(vmin=1, vmax=positive.max()), cmap="viridis",
    )
    fig.colorbar(pcm, ax=axes[0], label="number of contributing pairs")
    positive_w = weighted[weighted > 0]
    pcm = axes[1].pcolormesh(
        x_edges, y_edges, weighted, shading="auto",
        norm=LogNorm(vmin=positive_w.min(), vmax=positive_w.max()), cmap="magma",
    )
    fig.colorbar(pcm, ax=axes[1], label="fraction of integrated weight")
    for ax in axes:
        ax.plot(stars, stars, color="white", ls="--", lw=1.0,
                label=r"$\gamma_{ij}=\gamma_*$")
        ax.plot(stars, 2.0 * stars, color="cyan", ls=":", lw=1.2,
                label=r"$\gamma_{ij}=2\gamma_*$")
        ax.plot(stars, 2.5 * stars, color="lime", ls="-.", lw=1.2,
                label=r"$\gamma_{ij}=2.5\gamma_*$")
        ax.plot(stars, 3.0 * stars, color="orange", ls=(0, (5, 2)), lw=1.2,
                label=r"$\gamma_{ij}=3\gamma_*$")
        ax.set_ylabel(r"pair collision $\gamma_{ij}$")
        ax.legend(frameon=False)
    axes[1].set_xlabel(r"target $\gamma_*$")
    fig.tight_layout()
    fig.savefig(output_prefix.with_name(output_prefix.name + "_gamma_hist.pdf"))

    earliest = np.array([r["earliest_contact"]/r["r_star"] for r in records])
    latest = np.array([r["latest_contact"]/r["r_star"] for r in records])
    last_active = np.array([r["last_active"]/r["r_star"] for r in records])
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    series = (
        (earliest, "earliest pair contact", "C0"),
        (latest, "latest contributing-pair contact", "C1"),
        (last_active, "last nonzero total weight", "C2"),
    )
    for ax, (values, label, color) in zip(axes, series):
        ax.plot(stars, values, color=color, lw=1.8)
        ax.set_ylabel(r"time / $R_*$")
        ax.set_title(label, loc="left")
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel(r"target $\gamma_*$")
    fig.tight_layout()
    fig.savefig(output_prefix.with_name(output_prefix.name + "_times.pdf"))

    # Required BubbleMaster time domain as a function of pair collision gamma.
    pair_gamma = np.concatenate([r["pair_gamma"] for r in records])
    pair_last = np.concatenate([r["pair_last_nonzero"] for r in records])
    valid = np.isfinite(pair_last)
    pair_gamma, pair_last = pair_gamma[valid], pair_last[valid]
    gamma_edges = np.arange(1.0, np.ceil(pair_gamma.max()) + 2.0, 1.0)
    gamma_centres = 0.5 * (gamma_edges[:-1] + gamma_edges[1:])
    bin_index = np.digitize(pair_gamma, gamma_edges) - 1
    bin_max = np.full(len(gamma_centres), np.nan)
    for i in range(len(bin_max)):
        selected = bin_index == i
        if np.any(selected):
            bin_max[i] = pair_last[selected].max()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(pair_gamma, pair_last, s=3, alpha=0.08, color="C0",
               rasterized=True, label="individual contributing pairs")
    ax.step(gamma_centres, bin_max, where="mid", color="black", lw=1.4,
            label=r"maximum in $\Delta\gamma_{ij}=1$ bins")
    ax.set_xlabel(r"pair collision $\gamma_{ij}$")
    ax.set_ylabel(r"last nonzero weight time $t_{\rm last}$")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_prefix.with_name(output_prefix.name + "_cutoff_envelope.pdf"))

    # Cumulative envelopes reveal which target-gamma realizations require the
    # long cutoff-time tail.
    subsets = (
        (lambda g: g < 4, r"$\gamma_*<4$"),
        (lambda g: g < 8, r"$\gamma_*<8$"),
        (lambda g: g < 16, r"$\gamma_*<16$"),
        (lambda g: g < 32, r"$\gamma_*<32$"),
        (lambda g: g <= 64, r"$\gamma_*\leq64$"),
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    for selector, label in subsets:
        selected_records = [r for r in records if selector(r["gamma_star"])]
        gamma = np.concatenate([r["pair_gamma"] for r in selected_records])
        last = np.concatenate([r["pair_last_nonzero"] for r in selected_records])
        valid = np.isfinite(last)
        gamma, last = gamma[valid], last[valid]
        edges = np.arange(1.0, np.ceil(gamma.max()) + 2.0, 1.0)
        centres = 0.5 * (edges[:-1] + edges[1:])
        indices = np.digitize(gamma, edges) - 1
        envelope = np.full(len(centres), np.nan)
        for i in range(len(envelope)):
            in_bin = indices == i
            if np.any(in_bin):
                envelope[i] = last[in_bin].max()
        ax.step(centres, envelope, where="mid", lw=1.4, label=label)
    ax.set_xlabel(r"pair collision $\gamma_{ij}$")
    ax.set_ylabel(r"maximum last nonzero weight time $t_{\rm last}$")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output_prefix.with_name(
        output_prefix.name + "_cutoff_envelope_by_gamma_star.pdf"))

    # Earliest required scan time for the same cumulative target-gamma cuts.
    fig, ax = plt.subplots(figsize=(10, 6))
    for selector, label in subsets:
        selected_records = [r for r in records if selector(r["gamma_star"])]
        gamma = np.concatenate([r["pair_gamma"] for r in selected_records])
        first = np.concatenate([r["pair_first_nonzero"] for r in selected_records])
        valid = np.isfinite(first)
        gamma, first = gamma[valid], first[valid]
        edges = np.arange(1.0, np.ceil(gamma.max()) + 2.0, 1.0)
        centres = 0.5 * (edges[:-1] + edges[1:])
        indices = np.digitize(gamma, edges) - 1
        envelope = np.full(len(centres), np.nan)
        for i in range(len(envelope)):
            in_bin = indices == i
            if np.any(in_bin):
                envelope[i] = first[in_bin].min()
        ax.step(centres, envelope, where="mid", lw=1.4, label=label)
    ax.set_xlabel(r"pair collision $\gamma_{ij}$")
    ax.set_ylabel(r"minimum first nonzero weight time $t_{\rm first}$")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output_prefix.with_name(
        output_prefix.name + "_first_weight_by_gamma_star.pdf"))

    # Same cumulative envelope at every second target gamma.  Use a colorbar
    # instead of an impractically large legend.
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    limits = list(range(3, 64, 2)) + [64]
    norm = Normalize(vmin=limits[0], vmax=limits[-1])
    cmap = plt.get_cmap("viridis")
    fig, ax = plt.subplots(figsize=(10, 6))
    for limit in limits:
        selected_records = [r for r in records if r["gamma_star"] <= limit]
        gamma = np.concatenate([r["pair_gamma"] for r in selected_records])
        last = np.concatenate([r["pair_last_nonzero"] for r in selected_records])
        valid = np.isfinite(last)
        gamma, last = gamma[valid], last[valid]
        edges = np.arange(1.0, np.ceil(gamma.max()) + 2.0, 1.0)
        centres = 0.5 * (edges[:-1] + edges[1:])
        indices = np.digitize(gamma, edges) - 1
        envelope = np.full(len(centres), np.nan)
        for i in range(len(envelope)):
            in_bin = indices == i
            if np.any(in_bin):
                envelope[i] = last[in_bin].max()
        ax.step(centres, envelope, where="mid", lw=1.0,
                color=cmap(norm(limit)), alpha=0.9)
    colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    colorbar.set_label(r"largest included target $\gamma_*$")
    ax.set_xlabel(r"pair collision $\gamma_{ij}$")
    ax.set_ylabel(r"maximum last nonzero weight time $t_{\rm last}$")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_prefix.with_name(
        output_prefix.name + "_cutoff_envelope_every2.pdf"))

    # Collapse each cumulative envelope to the single latest required time.
    target_limits = np.arange(int(stars.min()), int(stars.max()) + 1)
    required_tmax = np.empty(len(target_limits))
    for i, limit in enumerate(target_limits):
        selected_records = [r for r in records if r["gamma_star"] <= limit]
        required_tmax[i] = max(
            np.nanmax(r["pair_last_nonzero"]) for r in selected_records
        )
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.step(target_limits, required_tmax, where="post", color="C0", lw=1.6)
    ax.scatter(target_limits, required_tmax, color="C0", s=12, zorder=3)
    slope, intercept = np.polyfit(target_limits, required_tmax, 1)
    fitted = slope * target_limits + intercept
    residual = required_tmax - fitted
    r_squared = 1.0 - np.sum(residual**2) / np.sum(
        (required_tmax - required_tmax.mean())**2
    )
    ax.plot(target_limits, fitted, color="C1", ls="--", lw=1.5,
            label=(rf"fit: $t_{{\max}}={slope:.2f}\gamma_*"
                   rf"+{intercept:.2f}$, $R^2={r_squared:.4f}$"))
    ax.set_xlabel(r"maximum target $\gamma_*$")
    ax.set_ylabel(r"required maximum scan time $t_{\max}$")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_prefix.with_name(
        output_prefix.name + "_required_tmax.pdf"))


def plot_required_tmax_comparison(record_sets, output_path):
    """Overlay cumulative required-time curves for multiple potentials."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for label, records in record_sets.items():
        stars = np.array([r["gamma_star"] for r in records])
        limits = np.arange(int(stars.min()), int(stars.max()) + 1)
        required = np.array([
            max(np.nanmax(r["pair_last_nonzero"])
                for r in records if r["gamma_star"] <= limit)
            for limit in limits
        ])
        slope, intercept = np.polyfit(limits, required, 1)
        line, = ax.step(limits, required, where="post", lw=1.6,
                        label=label)
        ax.plot(limits, slope * limits + intercept, ls="--", lw=1.2,
                color=line.get_color(),
                label=rf"{label} fit: ${slope:.2f}\gamma_*+{intercept:.2f}$")
    ax.set_xlabel(r"maximum target $\gamma_*$")
    ax.set_ylabel(r"required maximum scan time $t_{\max}$")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path)
