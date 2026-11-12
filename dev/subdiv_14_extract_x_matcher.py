"""
Extraction method x matcher on VAL (configs/extract_x_matcher.yaml; the test split is frozen and is not read).

Each pipeline = one extraction method (try08 NER, or a cached val extraction: a local LLM from
dev/subdiv_09_llm_extract.py or GLiNER from dev/subdiv_15_gliner_extract.py) followed by one matcher: paper_b1 (case-insensitive exact match to the partner plat
names, own county; first predicted string with a hit wins) or the frozen matcher_v1.0. No query fusion. paper_b1 is
a simple baseline, not the paper's method; the paper's 16.18% is the paper criterion below (no geometry).

Metrics are matcher_v1.0's pre-registered set, computed with its own code (dev/eval_matcher_val.evaluate on all
val deeds, weak labels from dev/weak_labels.py): centroid-in over (b) all eligible deeds [primary] and (a) eligible
deeds with >= 1 extracted string, plat accuracy vs the weak label, contains, intersects; plus the paper criterion
(dev/eval_matcher_test.paper_original: >= 1 plat name at fuzz.ratio >= 70, raw strings, all counties). Per
extractor: extraction recall (deeds with >= 1 extracted string), seconds per deed and hardware.

Writes results/extract_x_matcher/val_cells.csv (extractor x matcher x county) and val_extraction.csv. Prints
aggregates only. Run with the repo venv (mapprejudice), like dev/eval_matcher_val.py:

    mapprejudice/bin/python dev/subdiv_14_extract_x_matcher.py
"""
import argparse
import csv
import json
import re
import statistics
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "dev"))
import plat_layers as pl  # noqa: E402
import yaml  # noqa: E402
from eval_matcher_test import paper_original  # noqa: E402
from eval_matcher_val import evaluate, preds_of  # noqa: E402
from weak_labels import load_deeds, weak_labels  # noqa: E402

from src.matcher import PlatIndex, frozen  # noqa: E402
from src.matcher.match import Match  # noqa: E402
from src.subdiv_match.data import deed_key  # noqa: E402

MATCHERS = ("paper_b1", "matcher_v1.0")


def casefold(s):
    return re.sub(r"\s+", " ", str(s).strip().casefold())


class ExactIndex(PlatIndex):
    """paper_b1 (a simple baseline, not the paper's method): the first predicted string equal (casefold, collapsed whitespace) to a partner plat primary name in
    the deed's county; the plat identity is that name's (county, n1(name)), as in dev/subdiv_10_eval.py."""

    def top1(self, preds, county, cfg=None, triples=frozenset()):
        if not hasattr(self, "_exact"):
            self._exact = {}
            for i, p in enumerate(self.plats):
                self._exact.setdefault(p["county"], {}).setdefault(casefold(p["name"]), i)
        for q in preds:
            i = self._exact.get(county, {}).get(casefold(q))
            if i is not None:
                ident = self.ident[i]
                return Match(ident, list(self.by_ident[ident]), 100.0, 100.0)
        return None


def llm_names(cache):
    out, n_err = {}, 0
    for line in Path(cache).open():
        r = json.loads(line)
        if "error" in r:
            n_err += 1
            continue
        out[(r["county"], frozenset(r["image_ids"]))] = (r["names"], r.get("seconds"))
    return out, n_err


def pct(n, d):
    return round(100 * n / d, 2) if d else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "extract_x_matcher.yaml"))
    ap.add_argument("--out", default=str(REPO_ROOT / "results" / "extract_x_matcher"))
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text())
    # geopandas quotes the offending field value when a date column fails to parse; keep data values out of the output
    warnings.filterwarnings("ignore", message="Error parsing datetimes")
    t0, errors = time.time(), Counter()
    plats, pairs = pl.load_plats()
    norm3 = frozen.frozen_normalizer(pairs)                    # raises if the frozen name rules changed
    index = {"paper_b1": ExactIndex(plats, norm3), "matcher_v1.0": PlatIndex(plats, norm3)}
    deeds = load_deeds(pl.pp.run_files()["pred"]["val"])       # refuses a test file
    labels = weak_labels(deeds, plats, errors)
    eligible = [lab["status"] == "eligible" for lab in labels]
    no_trs = [frozenset()] * len(deeds)                        # matcher_v1.0 has no section constraint

    cells, ext_rows = [], []
    for name, ex in cfg["extractors"].items():
        if ex["kind"] == "ner":
            ds = deeds
            timing = json.loads((REPO_ROOT / ex["timing"]).read_text())
            secs = {"mean": timing["ner_seconds"] / max(1, timing["counts"]["deeds"]), "median": None, "p90": None,
                    "source": f"{ex['timing']} (total / deeds; model load {timing['model_load_seconds']} s excluded)"}
            missing = 0
        else:
            if not (REPO_ROOT / ex["cache"]).exists():
                print(f"  {name}: skipped (no cached extraction yet)", flush=True)
                continue
            cache, n_err = llm_names(REPO_ROOT / ex["cache"])
            errors[f"{name}:cached_errors"] += n_err
            ds, s, missing = [], [], 0
            for d in deeds:
                rec = cache.get(deed_key(d["props"]))
                missing += rec is None
                ds.append({**d, "props": {**d["props"], "SUBDIVISION_p": list(rec[0]) if rec else []}})
                if rec and rec[1] is not None:
                    s.append(rec[1])
            q = statistics.quantiles(s, n=10) if len(s) >= 10 else [None] * 9
            secs = {"mean": statistics.mean(s) if s else None, "median": statistics.median(s) if s else None,
                    "p90": q[8], "source": f"{ex['cache']} (per-deed wall clock, {len(s)} deeds)"}
        has = [bool(preds_of(d["props"])) for d in ds]
        ext_rows.append({"extractor": name, "kind": ex["kind"], "deeds": len(ds), "missing_in_cache": missing,
                         "with_extraction": sum(has), "extraction_recall_pct": pct(sum(has), len(ds)),
                         "eligible": sum(eligible),
                         "eligible_with_extraction": sum(h and e for h, e in zip(has, eligible)),
                         "extraction_recall_eligible_pct": pct(sum(h and e for h, e in zip(has, eligible)), sum(eligible)),
                         "sec_per_deed_mean": round(secs["mean"], 3) if secs["mean"] is not None else None,
                         "sec_per_deed_median": round(secs["median"], 3) if secs["median"] is not None else None,
                         "sec_per_deed_p90": round(secs["p90"], 3) if secs["p90"] is not None else None,
                         "timing_source": secs["source"], "hardware": ex["hardware"],
                         **{k: ex[k] for k in ("model_sha256", "prompt_sha256") if k in ex}})
        paper = paper_original(ds, labels, index["matcher_v1.0"], errors)   # matcher-independent criterion
        for m in MATCHERS:
            t1 = time.time()
            res = evaluate(ds, labels, no_trs, index[m], frozen.CONFIG, errors)
            match_secs = (time.time() - t1) / max(1, sum(eligible))
            for county, r in sorted(res.items(), key=lambda kv: (kv[0] == "ALL", kv[0])):
                c, p = r["counts"], paper.get(county, {})
                cells.append({"extractor": name, "matcher": m, "county": county, "eligible_b": c.get("b_all", 0),
                              "with_extraction_a": c.get("a_with_pred", 0), "top1": c.get("top1", 0),
                              "centroid_in": c.get("centroid_in", 0),
                              "centroid_in_pct_b": r["centroid_in_pct_b"], "centroid_in_pct_a": r["centroid_in_pct_a"],
                              "plat_acc_pct_b": r["plat_acc_b"], "plat_acc_pct_a": r["plat_acc_a"],
                              "contains_pct_b": r["contains_pct_b"], "intersects_pct_b": r["intersects_pct_b"],
                              "paper_criterion_pct_all": p.get("pct_all"), "paper_criterion_pct_a": p.get("pct_a"),
                              "paper_criterion_n_all": (p.get("counts") or {}).get("all_with_pred"),
                              "match_sec_per_eligible_deed": round(match_secs, 4) if county == "ALL" else None})
        print(f"  {name} done ({round(time.time() - t0)} s)", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for fname, rows in (("val_cells.csv", cells), ("val_extraction.csv", ext_rows)):
        fields = list(dict.fromkeys(k for r in rows for k in r))
        with open(out / fname, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
    import pandas as pd
    pd.set_option("display.width", 250)
    cf = pd.DataFrame(cells)
    print(cf[cf.county == "ALL"].drop(columns=["county"]).to_string(index=False))
    print(pd.DataFrame(ext_rows).drop(columns=["timing_source", "hardware"], errors="ignore").to_string(index=False))
    print(cf[cf.county != "ALL"][["extractor", "matcher", "county", "eligible_b", "with_extraction_a",
                                  "centroid_in_pct_b", "plat_acc_pct_b", "paper_criterion_pct_all"]].to_string(index=False))
    print("errors:", dict(errors), "| seconds:", round(time.time() - t0, 1))


if __name__ == "__main__":
    main()
