"""
Oracle ceiling for PLSS normalization on the test split: is the gold value recoverable from the NER spans?

For each single-polygon test deed with a gold section (dev/derive_gold_trs.py CSV), the try08 NER spans
(SECTION_p / TOWNSHIP_p / RANGE_p of testdata_all_predicted.geojson) are read by
src/normalizer/candidates.py. A slot counts as recoverable when the gold value is among ANY candidate
value (a) from spans of that slot's label, (b) from any PLSS span. The full triple needs all three. Deeds
where the gold is not recoverable are NER-limited; the remaining gap to the normalizer's accuracy is the
normalizer's own error.

Counts are reported overall and per county, with two denominators: all test deeds with a gold section,
and test deeds with any PLSS span (the headline). Also: spans per slot, and whether the R spans state
the gold range direction.

Privacy: reads image_ids, cnty_name and the three span columns; prints and writes COUNTS ONLY. Errors
inside data loops report the exception type and a numeric index only.

    mapprejudice/bin/python dev/oracle_ceiling.py
"""

import argparse
import csv
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.normalizer import spans_from_fields  # noqa: E402
from src.normalizer.candidates import candidate_values, read_spans  # noqa: E402


def _load(name):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "dev" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rt3 = _load("recompute_table3")
cso = _load("check_split_overlap")
DEFAULTS = {"gold": REPO_ROOT / "output" / "derived" / "gold_trs_622.csv",
            "summary": REPO_ROOT / "results" / "normalizer" / "oracle_ceiling.json"}


def load_gold(path):
    gold = {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["split"] == "test" and r["single"] == "True" and r["status"] == "ok":
                gold[(r["county"], r["deed_key"])] = {"T": int(r["T"]), "R": int(r["R"]), "S": int(r["S"]), "Rdir": r["Rdir"]}
    return gold


def deed_counts(spans, g):
    c = Counter()
    readings = read_spans(spans)
    by_label = Counter(s.label for s in spans)
    for k in ("S", "T", "R"):
        c[f"spans_{k}_{min(by_label[k], 3)}"] += 1
    hit_label = {k: g[k] in candidate_values(readings, k, True) for k in ("T", "R", "S")}
    hit_any = {k: g[k] in candidate_values(readings, k, False) for k in ("T", "R", "S")}
    for k in ("T", "R", "S"):
        c[f"{k}_in_same_label_spans"] += hit_label[k]
        c[f"{k}_in_any_span"] += hit_any[k]
    c["TR_in_same_label_spans"] += hit_label["T"] and hit_label["R"]
    c["TRS_in_same_label_spans"] += all(hit_label.values())
    c["TRS_in_any_span"] += all(hit_any.values())
    c["R_spans_state_gold_Rdir"] += any(d[0] == g["Rdir"] for r in readings if r.label == "R" for d in r.dirs)
    c["R_spans_state_other_Rdir"] += any(d[0] in "EW" and d[0] != g["Rdir"] for r in readings if r.label == "R" for d in r.dirs)
    return c


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--raw", default=rt3.DEFAULTS["raw"], help="raw NER GeoJSON (testdata_all_predicted)")
    ap.add_argument("--gold", default=str(DEFAULTS["gold"]))
    ap.add_argument("--summary", default=str(DEFAULTS["summary"]))
    args = ap.parse_args(argv)

    gold = load_gold(args.gold)
    gdf = rt3.read(args.raw)
    errors, seen = Counter(), set()
    per = defaultdict(lambda: {"all": Counter(), "with_span": Counter()})
    rows = gdf.drop(columns="geometry").to_dict("records")
    for i, props in enumerate(rows):
        try:
            ids = cso.parse_image_ids(props.get("image_ids"))
            key = (cso.county_key(props.get("cnty_name")), "|".join(sorted(set(ids)))) if ids else None
            if key not in gold or key in seen:
                errors["not_in_gold_or_duplicate"] += 1
                continue
            seen.add(key)
            spans = spans_from_fields(props)
            c = deed_counts(spans, gold[key])
            for county in (key[0], "ALL"):
                per[county]["all"]["deeds"] += 1
                per[county]["all"].update(c)
                if spans:
                    per[county]["with_span"]["deeds"] += 1
                    per[county]["with_span"].update(c)
        except Exception as e:  # type and index only
            errors[f"row#{i}:{type(e).__name__}"] += 1
    result = {"inputs": {"raw_rows": len(rows), "gold_test_deeds": len(gold), "joined": len(seen),
                         "gold_not_found_in_raw": len(set(gold) - seen)},
              "counts": {k: {d: dict(sorted(v.items())) for d, v in per[k].items()} for k in sorted(per)}}
    for k, block in result["counts"].items():
        for d, v in block.items():
            n = v.get("deeds") or 1
            v["pct"] = {m: round(100 * x / n, 2) for m, x in v.items() if m.startswith(("T", "R", "S")) and "spans_" not in m}
    result["errors"] = dict(Counter(k.split(":")[-1] if "#" in k else k for k in errors.elements()))
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
