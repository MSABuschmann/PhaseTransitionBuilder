# PhaseTransitionBuilder

Python/C++ code for predicting the gravitational-wave spectrum of a first-order vacuum phase transition by
building it up from individual two-bubble collisions:

1. **Pair spectra.** `bubblemaster` simulates the collision of two bubbles in 1+1D and computes its
   gravitational-wave spectrum as a function of time. A scan covers a grid of collision Lorentz factors
   $\gamma_{ij}$ (CPU and GPU versions).
2. **Geometric weights.** For a 3D population of bubbles, `weights` computes, for every colliding pair
   (including periodic images), the fraction $\mathcal W_{ij}(t)$ of its collision surface that is not inside
   any other bubble.
3. **Surrogate spectrum.** The spectrum of the full transition is the sum of the pair spectra, interpolated in
   $\gamma_{ij}$ and weighted by $\mathcal W_{ij}(t)$, optionally with a damping of the weights calibrated on 1+1D
   runs.

The surrogate is validated against 3D lattice simulations with
[sledgehamr](https://github.com/MSABuschmann/sledgehamr).

If you use this code, please cite: *[paper reference, arXiv number]*.

## Repository layout

| Path | Contents |
|---|---|
| `ptbuilder/` | Python package: potentials and instantons, scan setup, weights I/O, spectrum reconstruction, damping |
| `cpp/bubblemaster/` | 1+1D pair evolution and gravitational-wave integrals (`bubblemaster`, `bubblemaster_filon`, `bubblemaster_gpu`) |
| `cpp/weights/` | Geometric collision weights |
| `cpp/solver_1d/` | 1D field solver |
| `notebooks/00_production_scan.ipynb` | Set up a pair-spectrum scan and write its SLURM script |
| `notebooks/01_paper_runs.ipynb` | Set up all scans and dedicated runs used in the paper |
| `notebooks/02_paper_figures.ipynb` | All paper figures; data from `notebooks/paper_data.py` |
| `data/` | Scan results, weights, wall-radius tables, damping kernel, extracted literature data, figure caches |
| `sledgehamr_runs/` | Gravitational-wave spectra and initial conditions of the lattice runs used in the paper |
| `tests/` | Unit tests (`python -m pytest tests`) |

## Requirements

**Python** ≥ 3.9 with numpy, scipy, h5py, matplotlib, pybind11 and
[cosmoTransitions](https://github.com/clwainwright/CosmoTransitions) (instanton solver).
Reading lattice output additionally needs `pySledgehamr` from the
[sledgehamr](https://github.com/MSABuschmann/sledgehamr) repository: clone it next to this repository
(`../sledgehamr`), or set `Config.pysledgehamr_path`.

**C++:** a C++17 compiler with OpenMP and the HDF5 C++ library; GSL for `bubblemaster`, Eigen for `weights`,
and CUDA for `bubblemaster_gpu` (compiled for A100, `-arch=sm_80`; adjust in `cpp/bubblemaster/Makefile`).

## Installation

```bash
git clone https://github.com/MSABuschmann/PhaseTransitionBuilder.git
cd PhaseTransitionBuilder
pip install -e .
make                    # CPU binaries into bin/, plus the ptbuilder._potential extension
make bubblemaster_gpu   # GPU binary (needs CUDA)
```

The `bubblemaster` and `solver_1d` Makefiles use the Cray compiler wrapper `CC` and take HDF5 from `HDF5_DIR`
(as set by the HDF5 module on Cray systems such as NERSC Perlmutter); change `CXX` for other systems.
The `weights` Makefile also supports macOS with Homebrew (`llvm`, `hdf5`, `libomp`, `eigen`).

## Reproducing the paper figures

```bash
cd notebooks
jupyter notebook 02_paper_figures.ipynb
```

The notebook only loads and plots. Each figure's data is built by a `build_*` function in `paper_data.py` and
cached in `data/paper_cache/`; the committed caches let the notebook run without any of the raw data below.
Set `REBUILD = True` (or delete a cache file) to recompute from the raw data:

- the $N_b=2$, $N_b=3$ and $N_b>3$ comparisons with the lattice use the data in `data/` and `sledgehamr_runs/`
  (requires `pySledgehamr` and the compiled `weights` binary);
- the predictions for larger $\gamma_*$ and $N_b$, the weight distributions, the power decomposition, the
  comparison with Cutting et al. (2020) and the weights figure additionally need the $\gamma_*=1$–$16$ scans
  and the family weights, which are too large for this repository: *[data archive link]*.

## Running new scans

`notebooks/00_production_scan.ipynb` builds a scan with `ptbuilder.ic.generate_surrogate` and writes a SLURM
script to `scripts/`. The SLURM header is written for NERSC Perlmutter GPU nodes; set your account and adapt
the header for other clusters. Submit the script with `sbatch`; results are written to `data/`.
`notebooks/01_paper_runs.ipynb` does the same for every scan and dedicated run used in the paper.

## Configuration

`pt.Config()` sets where results go and where the binaries and `pySledgehamr` are found:

```python
import ptbuilder as pt

config = pt.Config(
    results_dir       = "data",
    bubblemaster_bin  = "bin/bubblemaster",
    solver_1d_bin     = "bin/solver_1d",
    weights_bin       = "bin/weights",
    pysledgehamr_path = "../sledgehamr",
)
```
