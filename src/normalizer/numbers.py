"""
Number words <-> integers for PLSS numerals.

The parser extends the regex v1 normalizer (dev/recompute_table3.py:49-77: UNITS/TENS vocabulary,
'hundred', 'and', the 'fourty' spelling) with:
  * OCR-tolerant word matching: digit look-alikes inside words (f0rty), 'rn'/'vv'/'cl' confusions,
    then rapidfuzz ratio >= 80 for words of 4+ letters;
  * compounds without spaces (twentyfour, onehundredseventeen) by vocabulary segmentation;
  * words split by OCR (Sev en) by joining adjacent fragments;
  * ordinals (Fourth, 5th) are never numbers, so 'Fourth Principal Meridian' yields nothing;
  * structure-aware composition: 'twenty four' -> 24, 'one hundred and six' -> 106, but
    'one and seven' -> [1, 7] and 'twelve twelve' -> [12, 12].
"""

import re
from dataclasses import dataclass
from functools import lru_cache

UNITS = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
         "fifteen sixteen seventeen eighteen nineteen").split()
TENS = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")
HUNDRED = 100
AND = -1
TENS_VALUES = {20, 30, 40, 50, 60, 70, 80, 90}

VOCAB = {w: i for i, w in enumerate(UNITS) if i}
VOCAB.update({w: 10 * i for i, w in enumerate(TENS) if w})
VOCAB.update({"fourty": 40, "ninty": 90, "hundred": HUNDRED})
ORDINALS = set(("first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth thirteenth "
                "fourteenth fifteenth sixteenth seventeenth eighteenth nineteenth twentieth thirtieth fortieth "
                "fiftieth sixtieth seventieth eightieth ninetieth hundredth").split())
# OCR look-alikes inside words -> letters (multi-char first)
WORD_REPAIRS = (("rn", "m"), ("vv", "w"), ("cl", "d"), ("0", "o"), ("1", "l"), ("5", "s"), ("8", "b"),
                ("|", "l"), ("2", "z"), ("6", "b"))
FUZZY_CUTOFF = 80
UNIT_CUTOFF = 75  # a unit word right after a tens word ("Twenty-Nne") or inside a compound
COMPOUND_MARGIN = 10   # a fuzzy tens+unit compound wins unless a single word scores this much higher
ORDINAL_MIN_LEN = 5    # fuzzy ordinal check ("Fourth") only for words this long
UNIT_WORDS = {w: VOCAB[w] for w in UNITS[1:10]}
LABEL_PREFIXES = ("township", "section", "numbered", "number", "range", "sec", "twp", "rge", "rng")


@dataclass(frozen=True)
class WordNumber:
    value: int
    start: int
    end: int
    quality: str  # "exact", "ocr" (look-alike repair / joined fragments) or "fuzzy"


def int_to_words(n, joiner="-", hundred_and=False):
    """1..999 -> words. joiner goes between tens and units ("-", " " or ""); hundred_and adds "and"."""
    if not 0 < n < 1000:
        raise ValueError("expected 1..999")
    parts = []
    hundreds, rest = divmod(n, 100)
    if hundreds:
        parts.append(f"{UNITS[hundreds]} hundred")
        if rest and hundred_and:
            parts.append("and")
    if rest:
        if rest < 20:
            parts.append(UNITS[rest])
        else:
            tens, units = divmod(rest, 10)
            parts.append(TENS[tens] + (joiner + UNITS[units] if units else ""))
    return " ".join(parts)


def _repair(token):
    t = token.lower()
    for src, dst in WORD_REPAIRS:
        t = t.replace(src, dst)
    return t


@lru_cache(maxsize=4096)
def _segment(t):
    """Split t into vocabulary words (longest first); None if impossible."""
    if not t:
        return ()
    for n in range(min(len(t), 9), 2, -1):
        head = t[:n]
        if head in VOCAB or (head == "and" and len(t) > n):
            rest = _segment(t[n:])
            if rest is not None:
                return (VOCAB.get(head, AND),) + rest
    return None


def _is_ordinal(t):
    if t in ORDINALS:
        return True
    from rapidfuzz import fuzz, process
    if len(t) >= ORDINAL_MIN_LEN:
        o = process.extractOne(t, ORDINALS, scorer=fuzz.ratio)
        v = process.extractOne(t, VOCAB.keys(), scorer=fuzz.ratio)
        return o is not None and o[1] >= FUZZY_CUTOFF and (v is None or o[1] > v[1])
    return False


def word_values(token):
    """A token -> (tuple of vocab values, quality) or None. Values include HUNDRED and AND markers."""
    low = token.lower()
    if low in ORDINALS or re.fullmatch(r"\d+(st|nd|rd|th)", low):
        return None
    if low in VOCAB:
        return (VOCAB[low],), "exact"
    seg = _segment(low)
    if seg and any(v != AND for v in seg):
        return seg, "exact"
    fixed = _repair(low)
    if fixed != low:
        if fixed in VOCAB:
            return (VOCAB[fixed],), "ocr"
        seg = _segment(fixed)
        if seg and any(v != AND for v in seg):
            return seg, "ocr"
    for prefix in LABEL_PREFIXES:  # label glued to the number: 'SECTIONTwenty'
        if fixed.startswith(prefix) and len(fixed) - len(prefix) >= 3:
            rest = word_values(token[len(prefix):])
            if rest:
                return rest
    if len(fixed) >= 4 and fixed.isalpha() and not _is_ordinal(fixed):
        from rapidfuzz import fuzz, process
        single = process.extractOne(fixed, VOCAB.keys(), scorer=fuzz.ratio, score_cutoff=FUZZY_CUTOFF)
        compound = None  # tens + unit with OCR damage: 'tvventynne', 'twentywo', 'twentycight'
        for cut in range(4, len(fixed) - 1):
            a = process.extractOne(fixed[:cut], TENS[2:], scorer=fuzz.ratio, score_cutoff=FUZZY_CUTOFF)
            b = process.extractOne(fixed[cut:], UNIT_WORDS.keys(), scorer=fuzz.ratio, score_cutoff=UNIT_CUTOFF)
            if a and b and (compound is None or min(a[1], b[1]) > compound[0]):
                compound = (min(a[1], b[1]), VOCAB[a[0]], UNIT_WORDS[b[0]])
        if compound and (single is None or compound[0] >= single[1] - COMPOUND_MARGIN):
            return (compound[1], compound[2]), "fuzzy"
        if single:
            return (VOCAB[single[0]],), "fuzzy"
    return None


def unit_after_tens(token):
    """A damaged unit word right after a tens word ('Nne' in 'Twenty-Nne') -> value or None."""
    t = _repair(token.lower())
    if len(t) < 2 or not t.isalpha() or t in ORDINALS:
        return None
    from rapidfuzz import fuzz, process
    hit = process.extractOne(t, UNIT_WORDS.keys(), scorer=fuzz.ratio, score_cutoff=UNIT_CUTOFF)
    return UNIT_WORDS[hit[0]] if hit else None


def compose(values):
    """[(vocab value, start, end)] (HUNDRED / AND markers allowed) -> [(int, start, end)], split where the
    word structure breaks ('one and seven' -> 1, 7; 'one hundred and six' -> 106)."""
    out, i = [], 0
    while i < len(values):
        v, s, e = values[i]
        i += 1
        if v == AND:
            continue
        if v == HUNDRED:
            total = 100
        elif v < 10 and i < len(values) and values[i][0] == HUNDRED:
            total, e = v * 100, values[i][2]
            i += 1
        elif v >= 20 and v % 10 == 0:
            if i < len(values) and 1 <= values[i][0] <= 9:
                v, e = v + values[i][0], values[i][2]
                i += 1
            out.append((v, s, e))
            continue
        else:
            out.append((v, s, e))
            continue
        j = i + 1 if i < len(values) and values[i][0] == AND else i  # after "hundred": optional "and"
        if j < len(values) and 1 <= values[j][0] < 100:
            rest, e = values[j][0], values[j][2]
            j += 1
            if rest >= 20 and rest % 10 == 0 and j < len(values) and 1 <= values[j][0] <= 9:
                rest, e = rest + values[j][0], values[j][2]
                j += 1
            total += rest
            i = j
        out.append((total, s, e))
    return out


TOKEN = re.compile(r"[A-Za-z0-9|&]+")
GAP = re.compile(r"^[\s\-‐–,]*$")


def _is_and(tok):
    low = tok.lower()
    return low in ("and", "&") or _repair(low) == "and"


def parse_number_words(text):
    """All numbers written in words in `text`, in order, as WordNumber (start/end cover their words)."""
    tokens = [(m.group(), m.start(), m.end()) for m in TOKEN.finditer(text)]
    items, i = [], 0  # (values, start, end, quality) or None for a non-number token
    while i < len(tokens):
        tok, s, e = tokens[i]
        if _is_and(tok):
            items.append(((AND,), s, e, "exact"))
            i += 1
            continue
        letters = sum(c.isalpha() for c in tok)
        wv = word_values(tok) if letters > sum(c.isdigit() for c in tok) else None
        if (wv is None or wv[1] == "fuzzy") and i + 1 < len(tokens) and tok.isalpha():  # OCR split: 'Sev en'
            nxt, s2, e2 = tokens[i + 1]
            nwv = word_values(nxt) if nxt.isalpha() else None
            if (nxt.isalpha() and " " in text[e:s2] and GAP.match(text[e:s2]) and (len(tok) < 3 or len(nxt) < 3)
                    and (nwv is None or nwv[1] == "fuzzy")):
                joined = word_values(tok + nxt)
                if joined and (wv is None or joined[1] != "fuzzy"):
                    items.append((joined[0], s, e2, "ocr"))
                    i += 2
                    continue
        if wv is None and items and items[-1] and len(items[-1][0]) == 1 and items[-1][0][0] in TENS_VALUES \
                and GAP.match(text[items[-1][2]:s]):
            unit = unit_after_tens(tok)
            if unit:
                wv = ((unit,), "fuzzy")
        items.append((wv[0], s, e, wv[1]) if wv else None)
        i += 1

    out, run = [], []

    def flush():
        while run and run[-1][0] == (AND,):
            run.pop()
        if run:
            quality = next((q for q in ("fuzzy", "ocr") if any(it[3] == q for it in run)), "exact")
            flat = [(v, it[1], it[2]) for it in run for v in it[0]]
            out.extend(WordNumber(n, s, e, quality) for n, s, e in compose(flat) if n > 0)
        run.clear()

    for it in items:
        if it is None:
            flush()
        elif run and not GAP.match(text[run[-1][2]:it[1]]) and it[0] != (AND,) and run[-1][0] != (AND,):
            flush()
            run.append(it)
        elif it[0] == (AND,) and not run:
            continue
        else:
            run.append(it)
    flush()
    return out
