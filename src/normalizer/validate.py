"""
Build (T, R, S) readings from span candidates and check each against a constraints Scope.

Replaces regex v1's zip_longest pairing (dev/recompute_table3.py:105-109). Without character offsets,
pairing uses list order: the i-th Township span goes with the i-th Range span (and the i-th Section
span when there are several); other combinations are kept with a lower confidence and usually fail the
PLSS check. When offsets become available (try08 re-run), only the alignment test needs to change.

Every combination gets a confidence (product of value confidences and the factors below) and a
`valid` status: "section" (exists in the scope), "township" (township-only reading that exists), or
"none". The features dict is what a learned ranker would consume instead of `conf`.
"""

from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional

from .constraints import in_range
from .types import TRS

ORDER_DECAY = 0.6         # a later number in the same span, unless joined by "and" / ","
FOREIGN_HINT = 0.15       # number whose label word names another slot, kept in its own span's slot
CROSS = 0.5               # number handed to another slot by its label word
UNALIGNED_TR = 0.5        # T and R from spans at different list positions
UNALIGNED_S = 0.4         # S from a span at a different list position than T
TOWNSHIP_ONLY = 0.3       # township reading when no section number was read
TOWNSHIP_FALLBACK = 0.05  # township reading when section numbers were read but none exists
DIR_STATED, DIR_DEFAULT, DIR_MISSING, DIR_CONFLICT, DIR_AMBIGUOUS = 1.0, 1.0, 0.9, 0.7, 0.6


@dataclass(frozen=True)
class SlotValue:
    value: int
    conf: float
    span: int      # list position of the span it came from (within that span's label)
    reading: int   # index into the readings list
    cross: bool    # came from a span of another label
    kind: str


@dataclass
class TripleCandidate:
    trs: TRS
    conf: float
    valid: str  # "section", "township" or "none"
    features: dict = field(default_factory=dict)


def slot_pool(readings, slot, max_values=3) -> List[SlotValue]:
    best = {}
    for ri, r in enumerate(readings):
        for c in r.values:
            if r.label == slot:
                f = 1.0 if c.hint in (None, slot) else FOREIGN_HINT
                if c.order > 0 and not c.joined and c.hint != slot:
                    f *= ORDER_DECAY
                v = SlotValue(c.value, c.conf * f, r.index, ri, False, c.kind)
            elif c.hint == slot:
                v = SlotValue(c.value, c.conf * CROSS, r.index, ri, True, c.kind)
            else:
                continue
            key = (v.value, v.reading)
            if key not in best or best[key].conf < v.conf:
                best[key] = v
    per_reading = {}
    for v in sorted(best.values(), key=lambda v: -v.conf):
        per_reading.setdefault(v.reading, [])
        if len(per_reading[v.reading]) < max_values:
            per_reading[v.reading].append(v)
    return [v for vs in per_reading.values() for v in vs]


def _stated(reading, letters, after=None):
    for d, conf, pos in reading.dirs:
        if d in letters and (after is None or pos >= after):
            return d
    return None


def rdir_options(r_read, t_read, r_start, scope):
    stated = _stated(r_read, "EW", r_start) or _stated(r_read, "EW") or (t_read and _stated(t_read, "EW"))
    dirs = scope.directions()
    if scope.accept_all:
        return [(stated, DIR_STATED, "stated")] if stated else [("", DIR_MISSING, "missing")]
    if stated and stated in dirs:
        return [(stated, DIR_STATED, "stated")]
    if stated:
        return [(d, DIR_CONFLICT, "conflict") for d in dirs] or [(stated, DIR_CONFLICT, "conflict")]
    if len(dirs) == 1:
        return [(dirs[0], DIR_DEFAULT, "default")]
    return [(d, DIR_AMBIGUOUS, "ambiguous") for d in dirs] or [("", DIR_MISSING, "missing")]


def build(readings, scope, max_values=3):
    """All (T, R, S) combinations with confidence and validity, plus diagnostic counts."""
    pools = {slot: slot_pool(readings, slot, max_values) for slot in ("S", "T", "R")}
    n_spans = Counter(r.label for r in readings if r.values)
    info = Counter(spans=len(readings), **{f"pool_{k}": len(v) for k, v in pools.items()})
    cands = []
    S, T, R = pools["S"], pools["T"], pools["R"]
    for t in T:
        for r in R:
            if not in_range(T=t.value, R=r.value):
                info["tr_out_of_range"] += 1
                continue
            info["tr_in_range"] += 1
            aligned = t.span == r.span or (n_spans["T"] <= 1 and n_spans["R"] <= 1)
            pconf = t.conf * r.conf * (1.0 if aligned else UNALIGNED_TR)
            t_read, r_read = readings[t.reading], readings[r.reading]
            r_start = next((c.start for c in r_read.values if c.value == r.value), None)
            for rdir, dconf, dsrc in rdir_options(r_read, t_read, r_start, scope):
                tdir = "N" if not scope.accept_all else (_stated(t_read, "NS") or "")
                base = {"aligned_tr": aligned, "dir_source": dsrc, "t_kind": t.kind, "r_kind": r.kind,
                        "t_cross": t.cross, "r_cross": r.cross, "t_conf": t.conf, "r_conf": r.conf}
                twp_ok = scope.township_ok(t.value, r.value, rdir)
                info["townships_valid" if twp_ok else "townships_invalid"] += 1
                any_s = False
                for s in S:
                    if not in_range(S=s.value):
                        info["s_out_of_range"] += 1
                        continue
                    any_s = True
                    s_al = s.span == t.span or n_spans["S"] <= 1
                    conf = pconf * dconf * s.conf * (1.0 if s_al else UNALIGNED_S)
                    ok = scope.section_ok(t.value, r.value, s.value, rdir)
                    info["sections_valid" if ok else "sections_invalid"] += 1
                    cands.append(TripleCandidate(TRS(t.value, tdir, r.value, rdir, s.value), conf,
                                                 "section" if ok else "none",
                                                 {**base, "aligned_s": s_al, "s_kind": s.kind, "s_cross": s.cross,
                                                  "s_conf": s.conf, "twp_ok": twp_ok}))
                if twp_ok:
                    factor = TOWNSHIP_FALLBACK if any_s else TOWNSHIP_ONLY
                    cands.append(TripleCandidate(TRS(t.value, tdir, r.value, rdir, None), pconf * dconf * factor,
                                                 "township", {**base, "township_only": True}))
    return cands, dict(info)
