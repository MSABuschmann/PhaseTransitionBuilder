from dataclasses import dataclass, field
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
_SLEDGEHAMR_DEFAULT = _REPO_ROOT.parent / "sledgehamr"


@dataclass
class Config:
    results_dir: Path             = _REPO_ROOT / "data"
    bubblemaster_bin: Path        = _REPO_ROOT / "bin/bubblemaster"
    bubblemaster_filon_bin: Path  = _REPO_ROOT / "bin/bubblemaster_filon"
    bubblemaster_gpu_bin: Path    = _REPO_ROOT / "bin/bubblemaster_gpu"
    solver_1d_bin: Path           = _REPO_ROOT / "bin/solver_1d"
    weights_bin: Path             = _REPO_ROOT / "bin/weights"
    pysledgehamr_path: Path       = _SLEDGEHAMR_DEFAULT

    def __post_init__(self):
        self.results_dir            = Path(self.results_dir)
        self.bubblemaster_bin       = Path(self.bubblemaster_bin)
        self.bubblemaster_filon_bin = Path(self.bubblemaster_filon_bin)
        self.bubblemaster_gpu_bin   = Path(self.bubblemaster_gpu_bin)
        self.solver_1d_bin          = Path(self.solver_1d_bin)
        self.weights_bin            = Path(self.weights_bin)
        self.pysledgehamr_path      = Path(self.pysledgehamr_path)

    def check_binaries(self):
        missing = [
            p for p in (self.bubblemaster_bin, self.solver_1d_bin, self.weights_bin)
            if not p.exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"Compiled binaries not found: {missing}\nRun `make` from the repo root."
            )
