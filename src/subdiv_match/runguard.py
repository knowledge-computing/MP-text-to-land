"""Run-once guard, cache checks and the single error retry for TEST extractions (docs/preregistration_test.md
sections 4.1 and 9). Used by dev/subdiv_09_llm_extract.py, dev/subdiv_15_gliner_extract.py and the test evaluation.

A cache is a JSON-lines file with one record per deed ({county, image_ids, names, ...}; an error record has an
"error" field). Everything here returns counts only; names are never returned or printed.

- A full pass may resume an incomplete cache, but is refused once the cache holds every deed.
- The retry (once per cache) needs a complete cache, re-queries only error records, replaces them in place
  (atomic rewrite, file mode kept) and writes a counts-only marker next to the cache.
"""
import csv
import json
import os
import tempfile
import time
from collections import Counter
from pathlib import Path


def key_of(rec):
    return (rec["county"], frozenset(rec["image_ids"]))


def read_records(path):
    """-> list of records. A line that is not valid JSON raises ValueError naming its index only."""
    out = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                raise ValueError(f"cache line {i} is not valid JSON") from None
    return out


def cache_status(path, keys):
    """Counts for a cache against the expected deed keys."""
    keys = set(keys)
    if not Path(path).exists():
        return {"exists": False, "lines": 0, "unique": 0, "duplicates": 0, "errors": 0, "with_thinking": 0,
                "missing": len(keys), "extra": 0}
    recs = read_records(path)
    seen = Counter(key_of(r) for r in recs)
    return {"exists": True, "lines": len(recs), "unique": len(seen), "duplicates": sum(v - 1 for v in seen.values()),
            "errors": sum("error" in r for r in recs), "with_thinking": sum(bool(r.get("thinking_chars")) for r in recs),
            "missing": len(keys - set(seen)), "extra": len(set(seen) - keys)}


def is_complete(status):
    return status["exists"] and status["missing"] == 0 and status["duplicates"] == 0 and status["extra"] == 0


def retry_marker(cache):
    return Path(str(cache) + ".retry.json")


def guard_pass(cache, keys):
    """Refuse a new full pass when the cache already holds every deed (resuming an incomplete cache is allowed)."""
    st = cache_status(cache, keys)
    if st["duplicates"] or st["extra"]:
        raise SystemExit(f"refused: cache has {st['duplicates']} duplicate and {st['extra']} unexpected records")
    if is_complete(st):
        raise SystemExit("refused: the test pass is already complete for this cache (run once; use --retry-errors "
                         "for the single retry of error records)")
    return st


def retry_errors(cache, keys, redo):
    """The single retry: redo(record) -> new record for each error record (redo must catch its own exceptions and
    return an error record). Refused if the cache is incomplete or the retry was already done. -> counts."""
    marker = retry_marker(cache)
    if marker.exists():
        raise SystemExit("refused: the retry was already done for this cache (once only)")
    st = cache_status(cache, keys)
    if not is_complete(st):
        raise SystemExit(f"refused: finish the pass first (missing {st['missing']}, duplicates {st['duplicates']}, "
                         f"extra {st['extra']})")
    recs = read_records(cache)
    out, retried = [], 0
    for r in recs:
        if "error" in r:
            new = dict(redo(r))
            new["retry_of_error"] = r["error"]
            out.append(new)
            retried += 1
        else:
            out.append(r)
    mode = os.stat(cache).st_mode & 0o777
    fd, tmp = tempfile.mkstemp(dir=Path(cache).parent, prefix=".retry_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            for r in out:
                fh.write(json.dumps(r) + "\n")
        os.chmod(tmp, mode)
        os.replace(tmp, cache)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    counts = {"errors_before": st["errors"], "retried": retried, "errors_after": sum("error" in r for r in out),
              "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    fd = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    with os.fdopen(fd, "w") as fh:
        fh.write(json.dumps(counts) + "\n")
    return counts


def append_log(path, row):
    """Append one row to a run-log CSV (header = union of all rows' fields, kept in first-seen order)."""
    path = Path(path)
    rows = list(csv.DictReader(path.open())) if path.exists() else []
    rows.append({k: v for k, v in row.items()})
    fields = list(dict.fromkeys(k for r in rows for k in r))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
