"""Deed OCR text by deed key, built exactly as dev/infer_try08.py does (first row's saved_path, remapped to the
local OCR root MP_OCR_ROOT, pages whitespace-collapsed and joined with newlines). Never printed."""
import sys
from pathlib import Path

from .data import deed_key, read_split

_DEV = Path(__file__).resolve().parents[2] / "dev"


def _infer():
    if str(_DEV) not in sys.path:
        sys.path.insert(0, str(_DEV))
    import infer_try08
    return infer_try08


def deed_texts(split_path, ocr_root, stats=None):
    it = _infer()
    rows = read_split(split_path, columns=["image_ids", "cnty_name", "saved_path"])
    out = {}
    for r in rows:
        k = deed_key(r)
        if k in out:
            continue
        try:
            text, missing = it.read_and_concat([it.remap_path(p, ocr_root) for p in it.parse_list(r.get("saved_path"))])
            if stats is not None:
                stats["pages_missing"] = stats.get("pages_missing", 0) + missing
        except Exception as e:  # noqa: BLE001
            if stats is not None:
                stats[f"read_error:{type(e).__name__}"] = stats.get(f"read_error:{type(e).__name__}", 0) + 1
            text = ""
        out[k] = text
    return out
