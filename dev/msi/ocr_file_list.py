"""
List the OCR page files one split needs (for copying them to MSI; dev/msi/MSI_RUNBOOK.md). Run on the laptop, from the
repo root, in any venv with pyogrio and geopandas (thesis-hf or mapprejudice):

    python dev/msi/ocr_file_list.py --split test

Uses exactly the pages the extraction reads (src/subdiv_match/ocr.deed_texts: the first row of each deed, saved_path
remapped to MP_OCR_ROOT). Writes the paths, relative to MP_OCR_ROOT, to output/private/<split>_ocr_files.txt
(git-ignored, owner-only) and prints counts only.
"""
import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "dev"))
sys.path.insert(0, str(REPO_ROOT / "src"))  # not via src/__init__ (it needs dotenv)
import infer_try08 as it  # noqa: E402
import private_paths as pp  # noqa: E402
from subdiv_match.data import deed_key, read_split  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--split", choices=["val", "test"], required=True)
    args = ap.parse_args(argv)
    ocr = os.fspath(pp.ocr_root())
    split = os.fspath(pp.run_files()["splits"][args.split])
    seen, files, unmapped, missing = set(), set(), 0, 0
    for r in read_split(split, columns=["image_ids", "cnty_name", "saved_path"]):
        k = deed_key(r)
        if k in seen:
            continue                      # deed_texts reads the first row of each deed only
        seen.add(k)
        for p in it.parse_list(r.get("saved_path")):
            q = it.remap_path(p, ocr)
            if q is None:
                unmapped += 1
            elif not q.exists():
                missing += 1
            else:
                files.add(os.path.relpath(q, ocr))
    out = REPO_ROOT / "output" / "private" / f"{args.split}_ocr_files.txt"
    old = os.umask(0o077)
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(sorted(files)) + "\n")
    finally:
        os.umask(old)
    print(f"split {args.split} | deeds {len(seen)} | files {len(files)} | unmapped {unmapped} | missing locally {missing}")
    print(f"list: {out.relative_to(REPO_ROOT)} (paths relative to MP_OCR_ROOT; never print it)")


if __name__ == "__main__":
    main()
