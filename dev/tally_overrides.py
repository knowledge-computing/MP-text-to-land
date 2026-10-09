"""
Aggregate error-type tally of the 81 manual overrides in the GPT-4o trial (Phase 0, normalizer plan).

Compares TRIPLE_p before the overrides (the GPT-4o batch files) with TRIPLE_p after them
(all_output.geojson), joined by deed key (county, sorted image_ids). Counts which parts changed
(S / T / R / direction / number of triples). With --gold (dev/derive_gold_trs.py CSV), also counts how
often the first triple and any triple equal the gold section before and after the overrides.

The overrides are never used as labels: this script outputs counts only. It reads image_ids,
cnty_name, TRIPLE_p and RANGE_p; parsing reuses dev/recompute_table3.py (unchanged). Errors inside
data loops report the exception type and a numeric index only.

    mapprejudice/bin/python dev/tally_overrides.py --gold output/derived/gold_trs_622.csv
"""

import argparse
import csv
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "dev" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rt3 = _load("recompute_table3")
cso = _load("check_split_overlap")
DEFAULT_SUMMARY = REPO_ROOT / "results" / "normalizer" / "phase0_override_tally.json"


def deed_triples(gdf, use_range_dir, errors, tag):
    out, dup = {}, 0
    cols = gdf.drop(columns="geometry", errors="ignore")
    for i, props in enumerate(cols.to_dict("records")):
        try:
            ids = cso.parse_image_ids(props.get("image_ids"))
            if not ids:
                errors[f"{tag}:no_ids"] += 1
                continue
            key = (cso.county_key(props.get("cnty_name")), "|".join(sorted(set(ids))))
            d = rt3.range_direction(props.get("RANGE_p")) if use_range_dir else None
            if key in out:
                dup += 1
            out[key] = rt3.as_triples(props.get("TRIPLE_p"), d)
        except Exception as e:  # type and index only
            errors[f"{tag}#{i}:{type(e).__name__}"] += 1
    return out, dup


def change_types(before, after):
    kinds = set()
    if not before and after:
        kinds.add("empty_to_nonempty")
    if before and not after:
        kinds.add("nonempty_to_empty")
    if len(before) != len(after):
        kinds.add("triple_count_changed")
    for b, a in zip(before, after):
        for name, j in (("S", 0), ("T", 1), ("R", 2), ("dir", 3)):
            if b[j] != a[j]:
                kinds.add(f"{name}_changed")
    return kinds


def load_gold(path):
    gold = {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["split"] == "test" and r["single"] == "True" and r["status"] == "ok":
                gold[(r["county"], r["deed_key"])] = (int(r["S"]), int(r["T"]), int(r["R"]), r["Rdir"])
    return gold


def correctness(triples, g):
    c = Counter()
    if not triples:
        return c
    top = triples[0]
    c["top1_S_T_R_dir"] += top[:3] == g[:3] and top[3] in (g[3], "")
    c["top1_S"] += top[0] == g[0]
    c["top1_T"] += top[1] == g[1]
    c["top1_R"] += top[2] == g[2]
    c["top1_dir"] += top[3] == g[3]
    c["any_S_T_R"] += any(t[:3] == g[:3] for t in triples)
    return c


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--batch", action="append", default=None, help="GPT-4o batch GeoJSONs (default: try08 batches 1-4)")
    ap.add_argument("--saved", default=str(rt3.DEFAULTS["pred"]), help="all_output.geojson (after overrides)")
    ap.add_argument("--gold", default=None, help="gold CSV from dev/derive_gold_trs.py")
    ap.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    args = ap.parse_args(argv)

    import pandas as pd
    errors = Counter()
    batches = pd.concat([rt3.read(p) for p in (args.batch or rt3.DEFAULTS["batches"])], ignore_index=True)
    before, dup_b = deed_triples(batches, True, errors, "batch")
    after, dup_a = deed_triples(rt3.read(args.saved), False, errors, "saved")
    common = set(before) & set(after)
    result = {"deeds_before": len(before), "deeds_after": len(after), "matched": len(common),
              "duplicate_keys_before": dup_b, "duplicate_keys_after": dup_a,
              "only_before": len(set(before) - common), "only_after": len(set(after) - common)}
    changed = [k for k in common if before[k] != after[k]]
    result["changed_deeds"] = len(changed)
    result["change_types"] = dict(sorted(Counter(t for k in changed for t in change_types(before[k], after[k])).items()))
    result["triples_per_deed_before_changed"] = dict(sorted(Counter(len(before[k]) for k in changed).items()))
    result["triples_per_deed_after_changed"] = dict(sorted(Counter(len(after[k]) for k in changed).items()))
    if args.gold:
        gold = load_gold(args.gold)
        for name, keys in (("changed_deeds", changed), ("all_matched_deeds", sorted(common))):
            with_gold = [k for k in keys if k in gold]
            b, a = Counter(), Counter()
            for k in with_gold:
                b.update(correctness(before[k], gold[k]))
                a.update(correctness(after[k], gold[k]))
            result[f"gold_{name}"] = {"with_gold": len(with_gold), "before": dict(sorted(b.items())), "after": dict(sorted(a.items()))}
    result["errors"] = dict(Counter(k.split(":")[-1] if "#" in k else k for k in errors.elements()))
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
