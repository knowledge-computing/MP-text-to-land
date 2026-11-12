"""Deed records for matching and evaluation (no printing).

query_source "ner": try08 SUBDIVISION_p strings (dev/infer_try08.py); "add_cov": the volunteer
transcription of the addition. blocks/lots always come from the volunteer transcription (block_cov /
lot_cov): try08 has no LOT/BLOCK labels. Gold = the deed's add_mod values (addition of the modern parcel).
"""
from .blocklot import parse_block, parse_lots
from .data import deeds_from_rows, read_split


def leak_free_train(train, others):
    """Drop train deeds whose deed key or (county, doc_num) also occurs in `others` (val/test deeds)."""
    keys = set().union(*(o.keys() for o in others))
    docs = {(d["county"], n) for o in others for d in o.values() for n in d["doc_nums"]}
    kept = {k: d for k, d in train.items()
            if k not in keys and not any((d["county"], n) in docs for n in d["doc_nums"])}
    return kept, len(train) - len(kept)


def contributors(key, d):
    return {(d["county"], n) for n in d["doc_nums"]}


def make_deeds(deeds, preds, query_source, loo=False):
    out = []
    for key, d in deeds.items():
        p = preds.get(key) if preds is not None else None
        queries = (p or {}).get("SUBDIVISION_p", []) if query_source == "ner" else list(d["add_cov"])
        lots = set()
        for x in d["lot_cov"]:
            lots |= parse_lots(x)
        out.append({"key": key, "county": d["county"], "queries": queries,
                    "blocks": {b for b in (parse_block(x) for x in d["block_cov"]) if b}, "lots": lots,
                    "exclude": contributors(key, d) if loo else set(), "gold": list(d["add_mod"]),
                    "has_pred": p is not None})
    return out


def load_split_deeds(path):
    return deeds_from_rows(read_split(path))
