"""
Run the try08 spaCy NER model (the paper's model) on the train or val split, one prediction per deed.

Re-implements the inference of the 2025 run folder's load_test_v2.py ($MP_RUN_DIR; it stays unchanged)
with four differences:
  * one document per deed: split rows are grouped by (county, set of image_ids); a split file holds one
    row per parcel geometry, so val has 2,436 rows but 491 deeds;
  * OCR paths in `saved_path` (MSI absolute paths) are remapped onto the local OCR root ($MP_OCR_ROOT): the
    part after the root's folder name is kept;
  * entity character offsets are kept (`ner_spans`), so PLSS components can be paired by position;
  * nothing from the data is printed (load_test_v2 printed OCR text and predictions).

The test split is frozen (CLAUDE.md, "Test-set freeze"): --split only accepts train/val, and any input
file whose name contains "test" is refused.

Output (a data file; write it under $MP_DATA_ROOT, never into this repo): GeoJSON FeatureCollection,
one feature per deed, geometry = union of the deed's row geometries, properties = the kept metadata
columns + n_rows, n_parts, n_pages, n_pages_missing, chars, SECTION_p ... CITY_p (lists of strings, the
same merge rule as load_test_v2.extract_kv) and ner_spans ([label, text, start, end] after merging).

Privacy: prints and writes to --summary only COUNTS, timing and library versions. Errors inside data loops
report the exception type and a numeric deed index only.

    HF_HOME=~/MapPrejudice/hf_cache ~/MapPrejudice/venvs/thesis-spacy/bin/python dev/infer_try08.py \
        --split val --out "$MP_DATA_ROOT/<run dir>/output/thesis/try08_val_predicted.geojson" \
        --summary results/infer_try08/val.json [--limit 20]
"""

import argparse
import ast
import json
import os
import platform
import re
import subprocess
import sys
import time
from collections import Counter
from importlib import metadata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "dev"))
import private_paths as pp  # noqa: E402  (private locations come from environment variables)

SPLITS = {"train": pp.run_dir("train", "train6_2_2.geojson"), "val": pp.run_dir("val", "val6_2_2.geojson")}
DEFAULT_MODEL = pp.run_dir("output", "try08", "NER_models_msigpu", "model-best")
DEFAULT_OCR_ROOT = pp.ocr_root()
LABELS = ("STATE", "COUNTY", "CITY", "SUBDIVISION", "SECTION", "TOWNSHIP", "RANGE")
KEEP = ("image_ids", "cnty_name", "city", "state", "add_cov", "add_mod", "lot_cov", "block_cov")
PACKAGES = ("spacy", "spacy-transformers", "thinc", "torch", "transformers", "shapely")


# ---------------------------------------------------------------- pure helpers (unit-tested)
def parse_list(value):
    """'(a, b)' / '[a]' / list / tuple / None -> list of str."""
    if value is None:
        return []
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return [value]
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return [str(value)]


def deed_key(props):
    return (str(props.get("cnty_name") or "").strip().lower(), frozenset(parse_list(props.get("image_ids"))))


def remap_path(path, ocr_root):
    """MSI absolute OCR path -> local path under ocr_root: the part after "<ocr_root folder name>/" is kept
    (None if the path does not contain that folder name)."""
    s, marker = str(path), Path(ocr_root).name + "/"
    if marker not in s:
        return None
    return Path(ocr_root) / s.split(marker, 1)[1]


def read_and_concat(paths):
    """Same text construction as load_test_v2.read_and_concat; returns (text, n_missing)."""
    text, missing = "", 0
    for p in paths:
        if p is None or not Path(p).exists():
            missing += 1
            continue
        with open(p, "r", encoding="utf-8") as f:
            text += re.sub(r"\s+", " ", f.read().strip()) + "\n"
    return text, missing


def extract_spans(doc):
    """load_test_v2.extract_kv merge rule (adjacent or 1-char-apart same-label entities are joined),
    keeping offsets. Returns {label: [(text, start, end), ...]}."""
    raw = {k: [] for k in LABELS}
    for ent in doc.ents:
        if ent.label_ in raw:
            raw[ent.label_].append((ent.text.strip(), ent.start_char, ent.end_char))
    out = {}
    for label, items in raw.items():
        merged = []
        for text, start, end in items:
            if merged and start == merged[-1][2] + 1:
                merged[-1] = (merged[-1][0] + " " + text, merged[-1][1], end)
            elif merged and start == merged[-1][2]:
                merged[-1] = (merged[-1][0] + text, merged[-1][1], end)
            else:
                merged.append((text, start, end))
        out[label] = merged
    return out


def group_deeds(features, stats):
    """Rows -> deeds in first-appearance order: {'props', 'geoms', 'paths'}."""
    deeds, index = [], {}
    for i, feat in enumerate(features):
        try:
            props = feat.get("properties") or {}
            key = deed_key(props)
            paths = parse_list(props.get("saved_path"))
            if key not in index:
                index[key] = len(deeds)
                deeds.append({"props": props, "geoms": [], "paths": paths})
            else:
                stats["rows_merged_into_existing_deed"] += 1
                if paths != deeds[index[key]]["paths"]:
                    stats["rows_with_different_saved_path"] += 1
            if feat.get("geometry"):
                deeds[index[key]]["geoms"].append(feat["geometry"])
        except Exception as e:  # noqa: BLE001 - report type + index only
            stats[f"group_error:{type(e).__name__}"] += 1
            stats["group_error_first_index"] = stats.get("group_error_first_index") or i
    return deeds


def union_geometry(geoms):
    from shapely.geometry import mapping, shape
    from shapely.ops import unary_union
    if not geoms:
        return None, 0
    g = unary_union([shape(x) for x in geoms])
    return mapping(g), (len(g.geoms) if hasattr(g, "geoms") else 1)


def versions():
    out = {"python": platform.python_version()}
    for p in PACKAGES:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = None
    try:
        out["repo_commit"] = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
                                            capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception:  # noqa: BLE001
        out["repo_commit"] = None
    return out


def refuse_test(path):
    if "test" in Path(path).name.lower():
        sys.exit("refused: the test split is frozen (CLAUDE.md, Test-set freeze)")


# ---------------------------------------------------------------- main
def run(geojson, out, ocr_root, nlp, limit=None, batch_size=16):
    refuse_test(geojson)
    ocr_root = os.fspath(ocr_root)
    stats = Counter()
    with open(geojson, encoding="utf-8") as f:
        fc = json.load(f)
    features = fc.get("features", [])
    stats["rows_in"] = len(features)
    deeds = group_deeds(features, stats)
    if limit:
        deeds = deeds[:limit]
    stats["deeds"] = len(deeds)

    texts = []
    for i, d in enumerate(deeds):
        try:
            local = [remap_path(p, ocr_root) for p in d["paths"]]
            text, missing = read_and_concat(local)
            d.update(n_pages=len(local), n_pages_missing=missing, chars=len(text))
            stats["pages"] += len(local)
            stats["pages_missing"] += missing
            stats["deeds_with_missing_page"] += missing > 0
            stats["deeds_empty_text"] += not text.strip()
            texts.append(text)
        except Exception as e:  # noqa: BLE001
            stats[f"read_error:{type(e).__name__}"] += 1
            d.update(n_pages=0, n_pages_missing=0, chars=0)
            texts.append("")

    t0 = time.time()
    docs = nlp.pipe(texts, batch_size=batch_size)
    out_features, label_counts, deeds_with = [], Counter(), Counter()
    for i, (d, doc) in enumerate(zip(deeds, docs)):
        try:
            spans = extract_spans(doc)
            geom, parts = union_geometry(d["geoms"])
            props = {k: d["props"].get(k) for k in KEEP if k in d["props"]}
            props.update(n_rows=len(d["geoms"]), n_parts=parts, n_pages=d["n_pages"],
                         n_pages_missing=d["n_pages_missing"], chars=d["chars"])
            for label in LABELS:
                props[label + "_p"] = [s[0] for s in spans[label]]
                label_counts[label] += len(spans[label])
                deeds_with[label] += bool(spans[label])
            props["ner_spans"] = sorted([label, *s] for label in LABELS for s in spans[label])
            stats["deeds_multi_part"] += parts > 1
            out_features.append({"type": "Feature", "geometry": geom, "properties": props})
        except Exception as e:  # noqa: BLE001
            stats[f"ner_error:{type(e).__name__}"] += 1
    seconds = time.time() - t0

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "crs": fc.get("crs"), "features": out_features}, f)
    stats["deeds_written"] = len(out_features)
    return {
        "counts": dict(stats),
        "entities_per_label": dict(label_counts),
        "deeds_with_label": dict(deeds_with),
        "deeds_with_any_plss": sum(any(f["properties"][k + "_p"] for k in ("SECTION", "TOWNSHIP", "RANGE"))
                                   for f in out_features),
        "ner_seconds": round(seconds, 1),
        "deeds_per_second": round(len(out_features) / seconds, 2) if seconds else None,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--split", choices=sorted(SPLITS), help="train or val (test is frozen)")
    ap.add_argument("--geojson", help="override the split file (toy runs); names containing 'test' are refused")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="default: $MP_RUN_DIR/output/try08/NER_models_msigpu/model-best")
    ap.add_argument("--ocr-root", default=DEFAULT_OCR_ROOT, help="default: $MP_OCR_ROOT")
    ap.add_argument("--out", required=True, help="output GeoJSON (data: keep it outside the repo)")
    ap.add_argument("--summary", help="aggregate JSON (counts, timing, versions)")
    ap.add_argument("--limit", type=int, help="first N deeds only (smoke test)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--prefer-gpu", action="store_true", help="spacy.prefer_gpu() (MPS/CUDA if available)")
    args = ap.parse_args(argv)
    if not (args.split or args.geojson):
        ap.error("give --split or --geojson")
    geojson = args.geojson or str(SPLITS[args.split])
    refuse_test(geojson)
    if Path(args.out).resolve().is_relative_to(REPO_ROOT) and "output" not in Path(args.out).resolve().relative_to(REPO_ROOT).parts[:1]:
        sys.exit("refused: --out inside the repo must be under output/ (git-ignored)")

    import spacy
    gpu = spacy.prefer_gpu() if args.prefer_gpu else False
    t0 = time.time()
    nlp = spacy.load(os.fspath(args.model))
    load_s = time.time() - t0
    result = run(geojson, args.out, args.ocr_root, nlp, args.limit, args.batch_size)
    meta_path = Path(args.model) / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    result.update(split=args.split or "custom", limit=args.limit, batch_size=args.batch_size, gpu=bool(gpu),
                  model_load_seconds=round(load_s, 1), versions=versions(),
                  model_meta={k: meta.get(k) for k in ("name", "version", "spacy_version", "pipeline")})
    text = json.dumps(result, indent=1)
    print(text)
    if args.summary:
        Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary).write_text(text + "\n")


if __name__ == "__main__":
    main()
