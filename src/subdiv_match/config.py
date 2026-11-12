"""Load configs/subdiv_match.yaml.

Private inputs (splits, predictions, try08 docs) are not named here: their file names come from the local MP_RUN_FILES
JSON (groups splits / pred / try08_docs, relative to MP_RUN_DIR) via dev/private_paths.py; OCR and the partner plat
layers come from MP_OCR_ROOT / MP_PLAT_CONFIG. Everything in the YAML's paths section resolves against the repo root.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT = REPO_ROOT / "configs" / "subdiv_match.yaml"
PRIVATE = ("splits", "pred", "try08_docs")     # groups read from MP_RUN_FILES


def _pp():
    if str(REPO_ROOT / "dev") not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / "dev"))
    import private_paths
    return private_paths


def resolve(p):
    p = Path(p).expanduser()
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


class _Unavailable:
    """Stands in for a private group when MP_RUN_FILES cannot be read; raises the original error on use, so
    public-only steps still load the config."""

    def __init__(self, err):
        self.err = err

    def __getitem__(self, key):
        raise self.err


def private_files():
    pp = _pp()
    try:
        files = pp.run_files()
    except pp.PrivatePathError as err:
        return {g: _Unavailable(err) for g in PRIVATE}
    return {g: files.get(g, _Unavailable(pp.PrivatePathError(f"MP_RUN_FILES has no group '{g}'"))) for g in PRIVATE}


def ocr_root():
    return _pp().ocr_root()


def plat_layers():
    """-> (root, {county: (relative layer path, name field, alias fields)}, rule pairs) from MP_PLAT_CONFIG."""
    _pp()
    import plat_layers as pl
    return pl.layer_config()


def load_config(path=DEFAULT):
    import yaml
    cfg = yaml.safe_load(Path(path).read_text())
    leaked = [k for k in cfg["paths"] if k in PRIVATE]
    if leaked:
        raise ValueError(f"{leaked}: private file names belong in MP_RUN_FILES, not in the YAML")
    cfg["paths"] = {**{k: resolve(v) for k, v in cfg["paths"].items()}, **private_files()}
    return cfg


def versions(packages=("numpy", "pandas", "geopandas", "shapely", "pyarrow", "rapidfuzz", "scikit-learn", "pyyaml")):
    import platform
    from importlib import metadata
    out = {"python": platform.python_version()}
    for p in packages:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = None
    return out
