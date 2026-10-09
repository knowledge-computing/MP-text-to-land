"""
The main local normalizer: candidates -> constraints -> validate -> select (no LLM, deterministic).

    from src.normalizer.constraints import Constraints
    from src.normalizer.rule_based import RuleBasedNormalizer
    norm = RuleBasedNormalizer(Constraints.from_json("output/derived/plss_valid_trs.json"), level="county")
    result = norm.normalize(spans, county="anoka", state="MN")   # Result: ranked TRS + reason + info
"""

from .candidates import read_spans
from .select import RuleScorer, select
from .types import Result
from .validate import build


class RuleBasedNormalizer:
    def __init__(self, constraints, level="county", scorer=None, keep_ratio=0.5, max_k=None, max_values=3):
        self.constraints, self.level = constraints, level
        self.scorer = scorer or RuleScorer()
        self.keep_ratio, self.max_k, self.max_values = keep_ratio, max_k, max_values
        self.name = f"rule_v2[{level}]"

    def normalize(self, spans, county, state) -> Result:
        readings = read_spans([s for s in spans if s.label in ("S", "T", "R")])
        scope = self.constraints.scope(county, state, self.level)
        cands, info = build(readings, scope, self.max_values)
        info["scope"] = scope.name
        return select(cands, info, self.scorer, self.keep_ratio, self.max_k)
