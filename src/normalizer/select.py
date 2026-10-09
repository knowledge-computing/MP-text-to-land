"""
Pick the reading(s) to return from validated candidates.

The scorer is pluggable: anything with `score(candidate) -> float` (a learned ranker over
`candidate.features` can replace RuleScorer). Selection keeps the best reading per distinct TRS, then
only readings scoring above `keep_ratio` x the best one, so a deed does not collect spurious extra
triples (any-of-k credit rewards those). Township-only readings are dropped when a section reading
survives.

Empty results carry a reason:
  no_spans      no Section / Township / Range span at all
  no_candidate  no number was read for Township or Range
  filtered_out  numbers were read but none is a plausible T / R / S value
  not_in_plss   plausible combinations exist but none is in the scope's PLSS layer
"""

from .types import Result


class RuleScorer:
    name = "rule"

    def score(self, cand):
        return cand.conf


def select(cands, info, scorer=None, keep_ratio=0.5, max_k=None) -> Result:
    scorer = scorer or RuleScorer()
    usable = [c for c in cands if c.valid == "section" or (c.valid == "township" and c.trs.S is None)]
    if not usable:
        if not info.get("spans"):
            reason = "no_spans"
        elif not info.get("pool_T") or not info.get("pool_R"):
            reason = "no_candidate"
        elif not info.get("tr_in_range"):
            reason = "filtered_out"
        else:
            reason = "not_in_plss"
        return Result([], reason, {**info, "valid_readings": []})
    best = {}
    for c in usable:
        sc = scorer.score(c)
        if c.trs not in best or best[c.trs] < sc:
            best[c.trs] = sc
    ranked = sorted(best.items(), key=lambda kv: (-kv[1], kv[0].T, kv[0].R, kv[0].S or 0, kv[0].Rdir))
    top = ranked[0][1]
    kept = [ranked[0]] + [kv for kv in ranked[1:] if kv[1] > keep_ratio * top]
    if any(t.S is not None for t, _ in kept):
        kept = [kv for kv in kept if kv[0].S is not None]
    if max_k:
        kept = kept[:max_k]
    return Result(kept, None, {**info, "valid_readings": [t for t, _ in ranked]})
