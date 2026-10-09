"""
The frozen rule_v2 configuration (frozen 2026-10-08, before the single Phase 2 run on the real test split).

`current()` collects the live constants; `FROZEN` is the record. test/test_normalizer_frozen.py fails if
they differ, so a change is always deliberate. Any later tuning uses train/val aggregates only (never
test) and gets a new version name.
"""

VERSION = "rule_v2.0"

FROZEN = {
    "numbers": {"FUZZY_CUTOFF": 80, "UNIT_CUTOFF": 75, "COMPOUND_MARGIN": 10, "ORDINAL_MIN_LEN": 5},
    "candidates": {
        "BASE_CONF": {"words+digits": 0.99, "paren": 0.9, "compact": 0.9, "digits": 0.85, "words": 0.85,
                      "words_ocr": 0.75, "paren_repaired": 0.65, "words_fuzzy": 0.6, "digits_repaired": 0.6,
                      "split_digits": 0.35},
        "DISAGREE": {"words": 0.8, "paren": 0.7},
        "UNKNOWN_PREFIX": 1.5, "UNKNOWN_SUFFIX": 1.0, "SPLIT_MAX_EXTRA_PENALTY": 1.0, "SECOND_SPLIT": 0.6,
        "UNKNOWN_PART": 0.7, "LABEL_CUTOFF": 75, "DIR_CUTOFF": 80,
    },
    "validate": {"ORDER_DECAY": 0.6, "FOREIGN_HINT": 0.15, "CROSS": 0.5, "UNALIGNED_TR": 0.5, "UNALIGNED_S": 0.4,
                 "TOWNSHIP_ONLY": 0.3, "TOWNSHIP_FALLBACK": 0.05, "DIR_STATED": 1.0, "DIR_DEFAULT": 1.0,
                 "DIR_MISSING": 0.9, "DIR_CONFLICT": 0.7, "DIR_AMBIGUOUS": 0.6},
    "constraints": {"S_RANGE": [1, 36], "T_RANGE": [1, 200], "R_RANGE": [1, 80]},
    "normalizer": {"keep_ratio": 0.5, "max_k": None, "max_values": 3},
    "plss_table": {"min_frac": 0.01, "mn_rdir": "0=W"},
}


def current():
    import inspect
    from . import candidates, constraints, numbers, rule_based, validate
    defaults = inspect.signature(rule_based.RuleBasedNormalizer.__init__).parameters
    pick = lambda mod, names: {n: getattr(mod, n) for n in names}
    return {
        "numbers": pick(numbers, FROZEN["numbers"]),
        "candidates": pick(candidates, FROZEN["candidates"]),
        "validate": pick(validate, FROZEN["validate"]),
        "constraints": {k: list(v) for k, v in pick(constraints, FROZEN["constraints"]).items()},
        "normalizer": {k: defaults[k].default for k in FROZEN["normalizer"]},
        "plss_table": FROZEN["plss_table"],  # checked against dev/build_plss_table.py defaults in the test
    }
