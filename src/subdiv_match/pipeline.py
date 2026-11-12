"""Shared setup for steps 5 and 6: gazetteer variants, train / val deed lists, gold entries."""
from .build import build_gazetteer, load_records
from .data import ID_COLUMNS, deeds_from_rows, read_predictions, read_split
from .deeds import leak_free_train, load_split_deeds, make_deeds
from .matchers import MatchIndex
from .normalize import NORMALIZERS


def make_index(cfg, sources, records=None):
    P = cfg["paths"]
    records = records if records is not None else load_records(P["derived"] / "records.parquet")
    gaz = build_gazetteer(records, NORMALIZERS[cfg["normalizer"]["version"]], sources=set(sources))
    t = cfg["matchers"]["tfidf"]
    return MatchIndex(gaz, analyzer=t["analyzer"], ngram_range=t["ngram_range"], min_df=t["min_df"],
                      sublinear_tf=t["sublinear_tf"])


def split_deeds(cfg, split, query_source):
    """split 'train' (leak-free vs val + test identifiers, leave-one-out aliases) or 'val'."""
    P = cfg["paths"]
    if split not in ("train", "val"):
        raise ValueError("only train / val (the test split is frozen)")
    preds = read_predictions(P["pred"][split]) if query_source == "ner" else None
    deeds = load_split_deeds(P["splits"][split])
    info = {"deeds": len(deeds)}
    if split == "train":
        val = load_split_deeds(P["splits"]["val"])
        test_ids = deeds_from_rows(read_split(P["splits"]["test"], columns=ID_COLUMNS))  # identifiers only
        deeds, info["dropped_dup_val_test"] = leak_free_train(deeds, [val, test_ids])
    out = make_deeds(deeds, preds, query_source, loo=(split == "train"))
    info["with_gold"] = sum(bool(d["gold"]) for d in out)
    out = [d for d in out if d["gold"]]
    return out, info


def gold_entries(index, deeds):
    return [index.gaz.gold_entries(d["county"], d["gold"], d["exclude"]) for d in deeds]
