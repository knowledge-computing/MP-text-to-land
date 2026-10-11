"""
Locations of the private project data, read from environment variables (or from the repo's git-ignored
.env file when a variable is not set in the environment). Nothing about the private folder layout is
written in this repository; see .env.example and plat_config.example.json for placeholders.

    MP_DATA_ROOT    root folder of the private data (absolute path)
    MP_RUN_DIR      the 2025 run folder (splits, PLSS layers, try08 model and outputs);
                    absolute, or relative to MP_DATA_ROOT
    MP_OCR_ROOT     local root of the OCR text files; absolute, or relative to MP_DATA_ROOT. Its folder
                    name is also the marker used to map MSI absolute paths onto it
    MP_PLAT_CONFIG  local JSON (keep it out of git) describing the partner plat layers:
                    {"root": dir (absolute or relative to MP_DATA_ROOT),
                     "layers": {county: {"path", "name_field", "alias_fields"}},
                     "rule_pairs": {county: [name_field, alias_field]}}

A missing variable or a variable that points to a missing path raises PrivatePathError with a message
naming the variable; there is no fallback to a built-in path. Paths are resolved only when used
(LazyPath), so importing a module never needs the variables. Standard library only.
"""
import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = REPO_ROOT / ".env"
RELATIVE_TO_DATA_ROOT = {"MP_RUN_DIR", "MP_OCR_ROOT"}


class PrivatePathError(RuntimeError):
    pass


def _from_env_file(name, env_file):
    try:
        with open(env_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("export "):
                    line = line[7:].lstrip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                if key.strip() == name:
                    return value.strip().strip('"').strip("'") or None
    except OSError:
        return None
    return None


def get(name, env_file=None):
    """Value of a private-path variable from the environment, else from .env; raises if unset."""
    env_file = env_file or ENV_FILE
    value = os.environ.get(name) or _from_env_file(name, env_file)
    if not value:
        raise PrivatePathError(
            f"{name} is not set. Set it in the environment or in {Path(env_file).name} "
            f"(see .env.example); there is no built-in default.")
    return value


def path(name, env_file=None, must_exist=True):
    p = Path(os.path.expanduser(get(name, env_file)))
    if name in RELATIVE_TO_DATA_ROOT and not p.is_absolute():
        p = path("MP_DATA_ROOT", env_file) / p
    if must_exist and not p.exists():
        raise PrivatePathError(f"{name} points to a path that does not exist (check your .env).")
    return p


class LazyPath(os.PathLike):
    """A path under a private-path variable, resolved when used (os.fspath / str / Path)."""

    def __init__(self, var, *parts):
        self.var, self.parts = var, tuple(str(p) for p in parts)

    def resolve(self):
        return path(self.var).joinpath(*self.parts)

    def __fspath__(self):
        return str(self.resolve())

    __str__ = __fspath__

    def __truediv__(self, other):
        return LazyPath(self.var, *self.parts, other)

    def __repr__(self):
        return f"LazyPath(${self.var}/{'/'.join(self.parts)})"

    def __eq__(self, other):
        return isinstance(other, LazyPath) and (self.var, self.parts) == (other.var, other.parts)

    def __hash__(self):
        return hash((self.var, self.parts))


def run_dir(*parts):
    return LazyPath("MP_RUN_DIR", *parts)


def ocr_root():
    return LazyPath("MP_OCR_ROOT")


def plat_config(env_file=None):
    """Parsed MP_PLAT_CONFIG with `root` resolved to an absolute Path; validates the structure."""
    cfg_path = path("MP_PLAT_CONFIG", env_file)
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    for key in ("root", "layers", "rule_pairs"):
        if key not in cfg:
            raise PrivatePathError(f"MP_PLAT_CONFIG: missing key '{key}' (see plat_config.example.json)")
    root = Path(os.path.expanduser(cfg["root"]))
    if not root.is_absolute():
        root = path("MP_DATA_ROOT", env_file) / root
    if not root.exists():
        raise PrivatePathError("MP_PLAT_CONFIG: 'root' does not exist")
    for county, layer in cfg["layers"].items():
        missing = [k for k in ("path", "name_field", "alias_fields") if k not in layer]
        if missing:
            raise PrivatePathError(f"MP_PLAT_CONFIG: layer '{county}' lacks {missing}")
    cfg["root"] = root
    return cfg
