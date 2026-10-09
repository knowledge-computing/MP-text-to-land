"""
matcher_v1.0 — frozen 2026-10-09 by the pre-registered rule in CLAUDE.md ("Matcher v1.0 selection rule").

Selection set: train ∪ val try08 predictions, deduplicated by deed key, 524 eligible deeds
(results/matcher/selection_v1.json). Exact McNemar on "deed centroid inside the predicted plat":
A (no section constraint) 405 vs B (soft section bonus k=5) 409, discordant A-only 0 / B-only 4,
p = 0.125 >= 0.05 -> A.

The step-3 name rules are learned at run time from the plat layers (Olmsted PLATNAME/LABEL, Sherburne
LEGAL_NAME/ALIAS, min frequency 5). RULES_SHA256 pins them: frozen_rules() refuses to run if the layers
yield different rules. A snapshot lives in output/derived/matcher_v1_rules.json (git-ignored; derived
from partner data). Any change to these values needs a new version name.
"""
import hashlib
import json

from .match import MatchConfig
from .names import NameNormalizer, learn_rules

VERSION = "matcher_v1.0"
CONFIG = MatchConfig(scorer="ratio", norm="n3", own_county=True, aliases=False, section="none", k=0.0, cutoff=70.0)
RULES_MIN_FREQ = 5
RULES_SHA256 = "c8ee1ca12b701fa81cc08a8e78035de44d687340dfffe779e5fcb9729c64ac66"
TIE_RULE = "max (final, base score); then earlier predicted string; then first key in the county pool"


def rules_digest(rules):
    return hashlib.sha256(json.dumps(rules, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def frozen_rules(pairs):
    """Learn the step-3 rules from (name, alias) pairs and check them against RULES_SHA256."""
    rules = learn_rules(pairs, RULES_MIN_FREQ)
    if rules_digest(rules) != RULES_SHA256:
        raise RuntimeError(f"{VERSION}: learned name rules differ from the frozen ones (layers or code changed)")
    return rules


def frozen_normalizer(pairs):
    return NameNormalizer(frozen_rules(pairs))
