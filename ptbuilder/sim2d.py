"""
Subprocess wrapper for the bubblemaster binary.
"""
import subprocess
from pathlib import Path


def run_bubblemaster(setup_path: Path, output_dir: Path, config,
                     save_fields: bool = False,
                     timeout: int = 3600) -> None:
    """
    Run the bubblemaster binary for a single setup file.

    Signature: ./bubblemaster <setup.h5> <output_dir/> [--save-fields]

    Parameters
    ----------
    setup_path  : path to the HDF5 setup file written by write_2d_setup()
    output_dir  : directory where result_*.h5 files will be written
    config      : Config instance (provides bubblemaster_bin path)
    save_fields : if True, pass --save-fields to also write fields.h5
    timeout     : max wall time in seconds (default 1 hour)
    """
    config.check_binaries()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(config.bubblemaster_bin),
        str(setup_path),
        str(output_dir) + "/",
    ]
    if save_fields:
        cmd.append("--save-fields")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(
            f"bubblemaster failed (exit {result.returncode}):\n"
            f"  cmd:    {' '.join(cmd)}\n"
            f"  stdout: {result.stdout[-2000:]}\n"
            f"  stderr: {result.stderr[-2000:]}"
        )


def run_solver_1d(setup_path: Path, output_path: Path, config,
                  timeout: int = 600) -> None:
    """
    Run the 1D spherical solver for validation.

    Signature: ./solver_1d <setup.h5> <output.h5>
    """
    config.check_binaries()
    cmd = [
        str(config.solver_1d_bin),
        str(setup_path),
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(
            f"solver_1d failed (exit {result.returncode}):\n"
            f"  stdout: {result.stdout[-2000:]}\n"
            f"  stderr: {result.stderr[-2000:]}"
        )


def run_weights(input_path: Path, output_path: Path, config,
                timeout: int = 3600) -> None:
    """
    Compute geometric GW weights for all bubble pairs.

    Signature: ./weights <bubbles.h5> <weights.h5>
    """
    config.check_binaries()
    cmd = [
        str(config.weights_bin),
        str(input_path),
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(
            f"weights failed (exit {result.returncode}):\n"
            f"  stdout: {result.stdout[-2000:]}\n"
            f"  stderr: {result.stderr[-2000:]}"
        )
