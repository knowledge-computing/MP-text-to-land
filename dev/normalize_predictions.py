"""
S3: PLSS readings for NER predictions with the local rule-based normalizer (frozen rule_v2.0, county
level; no LLM). Spans are built in list order from SECTION_p / TOWNSHIP_p / RANGE_p, exactly as in
Phase 2, so the frozen behaviour is unchanged.

Library use: readings(deeds) -> list of frozenset((S, T, R)) per deed (readings with a section only).
CLI prints counts only:

    python dev/normalize_predictions.py --pred <val predictions geojson> [--summary results/matcher/val_readings.json]
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "dev"))
from src.normalizer import spans_from_fields  # noqa: E402
from src.normalizer.constraints import Constraints  # noqa: E402
from src.normalizer.frozen import VERSION  # noqa: E402
from src.normalizer.rule_based import RuleBasedNormalizer  # noqa: E402

VALID = REPO_ROOT / "output" / "derived" / "plss_valid_trs.json"
COUNTY_STATE = {"anoka": "MN", "dakota": "MN", "olmsted": "MN", "sherburne": "MN", "washington": "MN", "milwaukee": "WI"}


def readings(deeds, valid=VALID, stats=None):
    norm = RuleBasedNormalizer(Constraints.from_json(valid), "county")
    stats = stats if stats is not None else Counter()
    out = []
    for i, d in enumerate(deeds):
        trs = frozenset()
        try:
            spans = spans_from_fields(d["props"])
            stats["deeds"] += 1
            stats["with_plss_span"] += bool(spans)
            if spans:
                res = norm.normalize(spans, d["county"], COUNTY_STATE.get(d["county"]))
                trs = frozenset((t.S, t.T, t.R) for t in res.top_k() if t.S is not None)
                stats["with_reading"] += bool(res.ranked)
                stats["with_section_reading"] += bool(trs)
                stats["reason:" + str(res.reason)] += not res.ranked
        except Exception as e:  # noqa: BLE001
            stats[f"error:{type(e).__name__}"] += 1
        out.append(trs)
    return out


def main(argv=None):
    from weak_labels import load_deeds
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--pred", required=True)
    ap.add_argument("--summary")
    args = ap.parse_args(argv)
    stats = Counter()
    r = readings(load_deeds(args.pred), stats=stats)
    res = {"normalizer": VERSION, "counts": dict(stats), "readings_per_deed": dict(Counter(len(x) for x in r))}
    text = json.dumps(res, indent=1)
    print(text)
    if args.summary:
        Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary).write_text(text + "\n")


if __name__ == "__main__":
    main()
