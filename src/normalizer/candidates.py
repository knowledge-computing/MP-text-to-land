"""
Per-span candidate extraction: every plausible number (with a confidence) and direction in a span.

Sources, in order of base confidence:
  words+digits  number words followed by the same number in parentheses ("Twenty-four (24)")
  paren         digits in parentheses, OCR look-alikes repaired when digits dominate ("(2l)" -> 21)
  compact       label/direction glued to digits ("T31N", "R24W", "S5")
  digits        bare digits (fractions such as 1/4, ordinals such as 5th, and 4+ digit runs are skipped)
  words         number words (src/normalizer/numbers.py), lower confidence when OCR-repaired or fuzzy
  split_digits  two short digit groups separated by one space, read as one number ("2 2" -> 22)
When words and the parenthesized digits disagree, both stay as candidates with lower confidence; the
PLSS validity check downstream usually settles it.

Each number also records `hint`: the slot named by the label word right before it (Section / Township /
Range, OCR-tolerant), so a Section span that ran on into "of Township 31" can hand 31 to the T slot.
"""

import re
from dataclasses import dataclass, field, replace
from typing import List, Optional, Sequence, Tuple

from .numbers import _repair, parse_number_words, word_values
from .ocr import TO_DIGIT
from .types import Span

BASE_CONF = {
    "words+digits": 0.99, "paren": 0.9, "compact": 0.9, "digits": 0.85, "words": 0.85,
    "words_ocr": 0.75, "paren_repaired": 0.65, "words_fuzzy": 0.6, "digits_repaired": 0.6, "split_digits": 0.35,
}
DISAGREE = {"words": 0.8, "paren": 0.7}  # multipliers when words and parenthesized digits disagree
LABEL_WORDS = {
    "S": {"section", "sections", "sec", "secs", "sect", "s"},
    "T": {"township", "townships", "twp", "twps", "town", "tp", "t"},
    "R": {"range", "ranges", "rge", "rng", "r"},
}
LABEL_LONG = {"section": "S", "township": "T", "range": "R"}
SKIP_BEFORE_NUMBER = {"no", "nos", "number", "numbered", "num"}
DIR_WORDS = {"north": "N", "south": "S", "east": "E", "west": "W"}
TOKEN = re.compile(r"[A-Za-z0-9|]+")
UNKNOWN_PREFIX, UNKNOWN_SUFFIX = 1.5, 1.0  # split penalties, in "repaired characters"
SPLIT_MAX_EXTRA_PENALTY = 1.0  # a token's second-best split is kept if its penalty is within this of the best
SECOND_SPLIT = 0.6             # confidence factor for that second-best split
UNKNOWN_PART = 0.7             # confidence factor for a split with an unknown prefix/suffix and no repair
LABEL_CUTOFF = 75              # rapidfuzz ratio for OCR-damaged label words (Township, Section, Range)
DIR_CUTOFF = 80                # rapidfuzz ratio for OCR-damaged direction words


@dataclass(frozen=True)
class Candidate:
    value: int
    kind: str
    conf: float
    start: int
    end: int
    hint: Optional[str] = None  # slot named by the preceding label word
    order: int = 0              # position among the span's numbers
    agree: bool = False         # words and digits agreed
    disagree: bool = False      # words and digits disagreed
    joined: bool = False        # linked to the previous number by "and" / "&" / ","


@dataclass
class SpanReading:
    label: str
    index: int  # position among this label's spans (list order)
    span: Span
    values: List[Candidate] = field(default_factory=list)
    dirs: List[Tuple[str, float, int]] = field(default_factory=list)  # (N/S/E/W, conf, char position)


def _label_of(token):
    t = _repair(token.lower().strip("."))
    for slot, words in LABEL_WORDS.items():
        if t in words:
            return slot
    if len(t) >= 5:
        from rapidfuzz import fuzz, process
        hit = process.extractOne(t, LABEL_LONG.keys(), scorer=fuzz.ratio, score_cutoff=LABEL_CUTOFF)
        if hit:
            return LABEL_LONG[hit[0]]
    return None


def _hint_before(text, pos):
    """Slot of the label word right before `pos` (skipping 'No.' / 'numbered')."""
    tokens = [m.group() for m in TOKEN.finditer(text[:pos])]
    while tokens and tokens[-1].lower() in SKIP_BEFORE_NUMBER:
        tokens.pop()
    return _label_of(tokens[-1]) if tokens else None


def _in_parens(text, start, end):
    return text[:start].rstrip().endswith("(") and text[end:].lstrip().startswith(")")


def _digits(group):
    """'2l' -> (21, repaired=1); None when look-alikes outnumber real digits."""
    real = sum(c.isdigit() for c in group)
    repaired = len(group) - real
    if real == 0 or repaired > real:
        return None
    try:
        return int("".join(c if c.isdigit() else TO_DIGIT[c] for c in group)), repaired
    except KeyError:
        return None


def _dir_of(suffix):
    t = _repair(suffix.lower().strip("."))
    if len(t) == 1 and t.upper() in "NSEW":
        return t.upper()
    if t in DIR_WORDS:
        return DIR_WORDS[t]
    if len(t) >= 4:
        from rapidfuzz import fuzz, process
        hit = process.extractOne(t, DIR_WORDS.keys(), scorer=fuzz.ratio, score_cutoff=DIR_CUTOFF)
        return DIR_WORDS[hit[0]] if hit else None
    return None


def split_token(tok):
    """Readings of a token that mixes letters and digits ('T31N', 'TOWNSHIP27', '8north', '2B', 'Z2'):
    [(penalty, value, repaired, prefix, direction)] best first. Penalty = repaired look-alikes, plus a
    penalty for an unknown one-letter prefix and for an unknown suffix."""
    out = []
    for k in range(len(tok)):
        pre = tok[:k]
        if pre[-1:].isdigit():
            continue
        for n in (3, 2, 1):
            num, suf = tok[k:k + n], tok[k + n:]
            if len(num) < n or any(c.isdigit() for c in suf) or not all(c.isdigit() or c in TO_DIGIT for c in num):
                continue
            parsed = _digits(num)
            if parsed is None or parsed[0] <= 0:
                continue
            value, repaired = parsed
            if re.fullmatch(r"(?i)(st|nd|rd|th)", suf):
                continue
            direction = _dir_of(suf) if suf else None
            penalty = repaired
            if pre and _label_of(pre) is None and len(pre) == 1:
                penalty += UNKNOWN_PREFIX
            if suf and direction is None:
                penalty += UNKNOWN_SUFFIX
            out.append((penalty, value, repaired, pre, direction))
    out.sort(key=lambda x: (x[0], -len(str(x[1]))))
    return out


def digit_candidates(text):
    out, dirs = [], []
    tokens = list(TOKEN.finditer(text))
    for i, m in enumerate(tokens):
        tok, s, e = m.group(), m.start(), m.end()
        if text[s - 1:s] == "/" or text[e:e + 1] == "/" or re.fullmatch(r"(?i)\d+(st|nd|rd|th)", tok):
            continue
        if not any(c.isdigit() for c in tok):
            splits = []
        elif sum(c.isalpha() for c in tok) > sum(c.isdigit() for c in tok) and (
                word_values(tok) is not None or (not tok[-1].isdigit() and _label_of(tok) is not None)):
            splits = []  # an OCR-damaged word ('e1even', 'SEC1ON'), not a number
        else:
            splits = split_token(tok)
        paren = _in_parens(text, s, e)
        for rank, (penalty, value, repaired, pre, direction) in enumerate(splits[:2]):
            if rank and penalty > splits[0][0] + SPLIT_MAX_EXTRA_PENALTY:
                break
            if pre or direction:
                kind = "compact" if not repaired else "digits_repaired"
            elif paren:
                kind = "paren_repaired" if repaired else "paren"
            else:
                kind = "digits_repaired" if repaired else "digits"
            conf = BASE_CONF[kind] * (SECOND_SPLIT if rank else 1.0) * (UNKNOWN_PART if penalty >= 1 and not repaired else 1.0)
            if value < 1000:
                out.append(Candidate(value, kind, conf, s, e, hint=_label_of(pre) if pre else None))
            if direction and rank == 0:
                dirs.append((direction, 0.9, e - 1))
        if not splits and paren and len(tok) <= 3 and all(c in TO_DIGIT or c.isdigit() for c in tok):
            value = int("".join(c if c.isdigit() else TO_DIGIT[c] for c in tok))
            if 0 < value < 1000:
                out.append(Candidate(value, "paren_repaired", BASE_CONF["paren_repaired"], s, e))
        if (i + 1 < len(tokens) and tok.isdigit() and len(tok) <= 2 and tokens[i + 1].group().isdigit()
                and len(tok) + len(tokens[i + 1].group()) <= 3 and text[e:tokens[i + 1].start()] == " "):
            nxt = tokens[i + 1]
            out.append(Candidate(int(tok + nxt.group()), "split_digits", BASE_CONF["split_digits"], s, nxt.end()))
    return out, dirs


def direction_candidates(text):
    out = []
    for m in TOKEN.finditer(text):
        tok = m.group()
        low = _repair(tok.lower())
        if len(tok) == 1 and tok.upper() in "NSEW" and not text[m.end():].lstrip().startswith(tuple("0123456789")):
            out.append((tok.upper(), 0.8, m.start()))
        elif low in DIR_WORDS:
            out.append((DIR_WORDS[low], 0.95, m.start()))
        elif len(low) >= 4 and low.isalpha():
            from rapidfuzz import fuzz, process
            hit = process.extractOne(low, DIR_WORDS.keys(), scorer=fuzz.ratio, score_cutoff=DIR_CUTOFF)
            if hit:
                out.append((DIR_WORDS[hit[0]], 0.8, m.start()))
    return out


def read_span(span: Span, index: int = 0) -> SpanReading:
    text = span.text or ""
    digits, compact_dirs = digit_candidates(text)
    words = [Candidate(w.value, {"exact": "words", "ocr": "words_ocr", "fuzzy": "words_fuzzy"}[w.quality],
                       BASE_CONF[{"exact": "words", "ocr": "words_ocr", "fuzzy": "words_fuzzy"}[w.quality]],
                       w.start, w.end) for w in parse_number_words(text)]
    # agreement: words immediately followed by a parenthesized number
    merged, used = [], set()
    for w in words:
        follow = next((d for d in digits if d.kind.startswith("paren") and d.start >= w.end
                       and re.fullmatch(r"[\s(]*", text[w.end:d.start])), None)
        if follow is None:
            merged.append(w)
        elif follow.value == w.value:
            merged.append(replace(w, kind="words+digits", conf=BASE_CONF["words+digits"], end=follow.end, agree=True))
            used.add(id(follow))
        else:
            merged.append(replace(w, conf=w.conf * DISAGREE["words"], disagree=True))
            merged.append(replace(follow, conf=follow.conf * DISAGREE["paren"], disagree=True))
            used.add(id(follow))
    merged += [d for d in digits if id(d) not in used]
    merged.sort(key=lambda c: (c.start, -c.conf))
    values, prev = [], None
    for c in merged:
        hint = c.hint or _hint_before(text, c.start)
        gap = _repair(text[prev.end:c.start].lower()) if prev is not None else "x"
        joined = prev is not None and re.fullmatch(r"[\s,]*(and|&|,)[\s,]*", gap or "x") is not None
        if hint is None and (joined or (c.disagree and prev is not None and prev.disagree)):
            hint = prev.hint
        c = replace(c, hint=hint, order=len(values), joined=joined)
        values.append(c)
        prev = c
    dirs = compact_dirs + direction_candidates(text)
    dirs.sort(key=lambda d: d[2])
    return SpanReading(span.label, index, span, values, dirs)


def read_spans(spans: Sequence[Span]) -> List[SpanReading]:
    """Readings in input order. `index` (the list-order pairing key) counts only the spans of the same
    label that yield a number, so an empty spurious span ('Government' tagged R) does not shift pairing;
    spans without a number get index -1."""
    counts, out = {}, []
    for s in spans:
        r = read_span(s, -1)
        if r.values:
            r.index = counts.get(s.label, 0)
            counts[s.label] = r.index + 1
        out.append(r)
    return out


def candidate_values(readings, slot, same_label_only=True):
    """All candidate values for a slot (oracle ceiling): from spans of that label, or from every span."""
    return {c.value for r in readings if (r.label == slot or not same_label_only) for c in r.values}
