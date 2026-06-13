from dataclasses import dataclass, field
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent


@dataclass
class Config:
    results_dir: Path = Path("data")
    bubblemaster_bin: Path = _REPO_ROOT / "bin/bubblemaster"
    solver_1d_bin: Path    = _REPO_ROOT / "bin/solver_1d"
    weights_bin: Path      = _REPO_ROOT / "bin/weights"

    def __post_init__(self):
        self.results_dir     = Path(self.results_dir)
        self.bubblemaster_bin = Path(self.bubblemaster_bin)
        self.solver_1d_bin   = Path(self.solver_1d_bin)
        self.weights_bin     = Path(self.weights_bin)

    def check_binaries(self):
        missing = [
            p for p in (self.bubblemaster_bin, self.solver_1d_bin, self.weights_bin)
            if not p.exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"Compiled binaries not found: {missing}\nRun `make` from the repo root."
            )
