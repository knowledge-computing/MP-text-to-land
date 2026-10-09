"""
Baselines behind the same Normalizer interface.

RegexV1Normalizer wraps the local regex normalizer used for results/provenance/try08_ner_local_regex.json
(dev/recompute_table3.py: normalize_number, range_direction, zip_longest pairing), imported unchanged.
It ignores county and state. Triples may hold None for a slot that zip_longest padded.
The saved GPT-4o outputs are read from their GeoJSON files by the evaluation harness (Phase 2).
"""

import importlib.util
from itertools import zip_longest
from pathlib import Path

from .types import TRS, Result

_RT3 = None


def _rt3():
    global _RT3
    if _RT3 is None:
        path = Path(__file__).resolve().parents[2] / "dev" / "recompute_table3.py"
        spec = importlib.util.spec_from_file_location("recompute_table3", path)
        _RT3 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_RT3)
    return _RT3


class RegexV1Normalizer:
    name = "regex_v1"

    def normalize(self, spans, county=None, state=None) -> Result:
        rt3 = _rt3()
        by = {k: [s.text for s in spans if s.label == k] for k in ("S", "T", "R")}
        lists = [[n for n in (rt3.normalize_number(t) for t in by[k]) if n is not None] for k in ("S", "T", "R")]
        d = rt3.range_direction(by["R"])
        triples = [(s, t, r) for s, t, r in zip_longest(*lists)]
        if not triples:
            return Result([], "no_candidate" if spans else "no_spans")
        ranked = [(TRS(t, "", r, d, s), 1.0 - 0.01 * i) for i, (s, t, r) in enumerate(triples)]
        return Result(ranked, None)
