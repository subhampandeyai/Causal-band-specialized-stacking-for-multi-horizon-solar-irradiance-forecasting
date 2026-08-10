"""
FAME — Configuration Loader
============================
AUDIT FIX #1: Central config replaces ALL hardcoded paths.
AUDIT FIX #13: Run manifest generation for provenance.

Usage in any stage script:
    from utils.config import cfg, get_path, manifest

    data_dir = get_path("data_raw")
    seed = cfg["seeds"]["global"]
"""
import os, sys, yaml, json, hashlib, time, platform
from pathlib import Path
from typing import Optional
from datetime import datetime

# ── Locate project root (directory containing the config file) ─────
# The config file ships at configs/config.yaml. A bare config.yaml at the
# project root is also accepted, so older checkouts keep working.
_CONFIG_RELPATHS = (Path("configs") / "config.yaml", Path("config.yaml"))


def find_config(root: Path) -> Optional[Path]:
    """Return the config file under `root`, or None if it is not there."""
    for rel in _CONFIG_RELPATHS:
        candidate = root / rel
        if candidate.exists():
            return candidate
    return None


def _find_project_root() -> Path:
    """Walk upward from this file to find the directory holding the config."""
    current = Path(__file__).resolve().parent.parent
    for _ in range(5):
        if find_config(current) is not None:
            return current
        current = current.parent
    # Fallback: use environment variable
    env_root = os.environ.get("FAME_PROJECT_ROOT")
    if env_root and Path(env_root).exists():
        return Path(env_root)
    raise FileNotFoundError(
        "Cannot find configs/config.yaml. Set the FAME_PROJECT_ROOT env "
        "variable or run from the project directory."
    )

PROJECT_ROOT = _find_project_root()

# ── Load config ────────────────────────────────────────────────────
def load_config(config_path: Path = None) -> dict:
    """Load the config file and return it as a dict."""
    path = config_path or find_config(PROJECT_ROOT)
    if path is None or not path.exists():
        raise FileNotFoundError(
            f"Config not found under {PROJECT_ROOT}; expected one of: "
            + ", ".join(str(r) for r in _CONFIG_RELPATHS))
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

cfg = load_config()

# ── Path resolver (always returns absolute paths) ──────────────────
def get_path(key: str, create: bool = True) -> Path:
    """
    Resolve a path key from config to absolute path.
    Creates directory if create=True.
    
    Example: get_path("data_raw") → /abs/path/to/project/data/raw
    """
    relative = cfg["paths"].get(key)
    if relative is None:
        raise KeyError(f"Path key '{key}' not found in config.yaml")
    absolute = PROJECT_ROOT / relative
    if create and not absolute.suffix:  # Only mkdir for directories, not files
        absolute.mkdir(parents=True, exist_ok=True)
    return absolute

# ── Seed management ────────────────────────────────────────────────
def set_all_seeds(seed: int = None):
    """Set seeds for numpy, random, tensorflow, torch."""
    import random, numpy as np
    seed = seed or cfg["seeds"]["global"]
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(cfg["seeds"].get("tensorflow", seed))
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(cfg["seeds"].get("torch", seed))
    except ImportError:
        pass
    return seed

# ── File hashing (for provenance) ──────────────────────────────────
def file_hash(filepath: Path, algo: str = "sha256") -> str:
    """Compute hash of a file for provenance tracking."""
    h = hashlib.new(algo)
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]  # Short hash is sufficient

# ── Run Manifest (AUDIT FIX #13) ───────────────────────────────────
class RunManifest:
    """
    Records provenance for each stage run.
    Writes JSON manifest to outputs/artifacts/
    """
    def __init__(self, stage_name: str):
        self.stage_name = stage_name
        self.start_time = datetime.now().isoformat()
        self.data = {
            "stage": stage_name,
            "timestamp_start": self.start_time,
            "timestamp_end": None,
            "seed": cfg["seeds"]["global"],
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "config_hash": file_hash(find_config(PROJECT_ROOT)),
            "input_files": {},
            "output_files": {},
            "metrics": {},
            "parameters": {},
        }
    
    def log_input(self, name: str, filepath: Path):
        """Record an input file with its hash."""
        self.data["input_files"][name] = {
            "path": str(filepath),
            "hash": file_hash(filepath) if filepath.exists() else "MISSING",
        }
    
    def log_output(self, name: str, filepath: Path):
        """Record an output file with its hash."""
        self.data["output_files"][name] = {
            "path": str(filepath),
            "hash": file_hash(filepath) if filepath.exists() else "NOT_YET",
        }
    
    def log_metric(self, name: str, value):
        """Record a computed metric."""
        self.data["metrics"][name] = value
    
    def log_param(self, name: str, value):
        """Record a parameter used."""
        self.data["parameters"][name] = value
    
    def save(self):
        """Write manifest to outputs/artifacts/."""
        self.data["timestamp_end"] = datetime.now().isoformat()
        # Update output file hashes now that files exist
        for name, info in self.data["output_files"].items():
            p = Path(info["path"])
            if p.exists():
                info["hash"] = file_hash(p)
        
        artifact_dir = get_path("outputs_artifacts")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        manifest_path = artifact_dir / f"{self.stage_name}_{ts}.json"
        with open(manifest_path, 'w') as f:
            json.dump(self.data, f, indent=2, default=str)
        
        # Also write "latest" symlink-style file
        latest_path = artifact_dir / f"{self.stage_name}_latest.json"
        with open(latest_path, 'w') as f:
            json.dump(self.data, f, indent=2, default=str)
        
        return manifest_path

# ── Convenience: create a manifest ─────────────────────────────────
def manifest(stage_name: str) -> RunManifest:
    return RunManifest(stage_name)
