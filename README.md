# PhaseTransitionBuilder

A Python/C++ package for computing gravitational wave spectra from cosmological
first-order phase transitions via the bubble collision (envelope) approximation.

The workflow is:
1. Define a scalar field potential and compute the instanton (bounce) profile.
2. Run a 2D Milne-coordinate evolution of colliding bubble pairs with the
   `bubblemaster` binary over a grid of Lorentz factors γ.
3. Sample a 3D bubble population and compute geometric collision weights.
4. Bootstrap the predicted GW spectrum from the 2D scan results.
5. Optionally validate against a full 3D sledgehamr simulation.

## Requirements

### Python
- Python ≥ 3.9
- numpy, scipy, h5py, matplotlib
- [cosmoTransitions](https://github.com/clwainwright/CosmoTransitions) (instanton solver)

### C++ binaries
The three binaries (`bubblemaster`, `solver_1d`, `weights`) require:
- GCC with C++17 and OpenMP support
- [HDF5](https://www.hdfgroup.org/) C++ bindings (`libhdf5-cpp-dev`)
- [GSL](https://www.gnu.org/software/gsl/) (`libgsl-dev`)

On a Debian/Ubuntu cluster:
```bash
sudo apt install libhdf5-serial-dev libgsl-dev
```

## Installation

```bash
git clone https://github.com/MSABuschmann/PhaseTransitionBuilder.git
cd PhaseTransitionBuilder

# Install the Python package
pip install -e .

# Build the C++ binaries (requires HDF5 + GSL)
make
```

The Makefile assumes HDF5 headers at `/usr/include/hdf5/serial` and libraries at
`/usr/lib/x86_64-linux-gnu/hdf5/serial`.  Edit `cpp/*/Makefile` if your cluster
uses different paths (e.g. module-loaded HDF5).

## Usage

The main reconstruction workflow is:

| Notebook | Description |
|---|---|
| `notebooks/00_validate_instanton.ipynb` | Validate instanton profile and IC generation (no binaries needed) |
| `notebooks/01_scan.ipynb` | Run the 2D BubbleMaster scan over γ |
| `notebooks/02_realisation.ipynb` | Generate bubble population and bootstrap GW spectrum |
| `notebooks/03_validate_3d.ipynb` | Compare bootstrap against a sledgehamr 3D run |
| `notebooks/05_n3_reconstruction.ipynb` | Validate the reconstruction with three bubbles |
| `notebooks/06_n64_reconstruction.ipynb` | Compare the reconstructed and 3+1D N=64 spectra |

The remaining notebooks contain targeted solver and field-validation checks.
On a SLURM cluster, the scan and reconstruction notebooks only prepare inputs
and submission scripts in `scripts/`; submit the printed `sbatch` command and
continue the notebook after the job completes.

## Configuration

`pt.Config()` controls where results are written and where the binaries live:

```python
import ptbuilder as pt

config = pt.Config(
    results_dir      = "data",          # where HDF5 outputs go
    bubblemaster_bin = "bin/bubblemaster",
    solver_1d_bin    = "bin/solver_1d",
    weights_bin      = "bin/weights",
)
```
