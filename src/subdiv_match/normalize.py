"""Addition / plat-name normalization.

standardize_addition_mp: a faithful port of standardize_addition() from UMNLibraries/racial_covenants_processor
(apps/parcel/utils/parcel_utils.py), the Mapping Prejudice production normalizer. That code is
MIT-licensed, Copyright (c) 2023 Regents of the University of Minnesota; the port keeps its behaviour,
including substring ordinal replacement (no word boundaries) and dropping "addition"/"subdivision".

standardize (version std_v1) = OCR cleanup -> number words -> replat/rearrangement/RLS canonical forms ->
trailing boilerplate removal -> standardize_addition_mp -> token cleanup. Output is lower case.
"""
import re

from rapidfuzz import fuzz

VERSION = "std_v1"

# ------------------------------------------------------------------ port (MIT, see module docstring)
_MP_ORDINALS = {"first": "1st", "second": "2nd", "third": "3rd", "fourth": "4th", "fifth": "5th",
                "sixth": "6th", "seventh": "7th", "eighth": "8th", "ninth": "9th", "tenth": "10th"}


def standardize_addition_mp(input_str):
    if input_str:
        for word, abbr in _MP_ORDINALS.items():
            input_str = re.sub(f"{word}", f"{abbr}", input_str, flags=re.IGNORECASE)
        input_str = re.sub(r"['`’]([sn])", r"\1", input_str, flags=re.IGNORECASE)
        input_str = re.sub(r"([sn])['`’]", r"\1", input_str, flags=re.IGNORECASE)
        input_str = re.sub(r"ADD(?:IT)?\.", "ADDITION", input_str, flags=re.IGNORECASE)
        input_str = re.sub(r" ADDN(\.)?", " ADDITION", input_str, flags=re.IGNORECASE)
        input_str = re.sub(r" ADD(\.)?$", " ADDITION", input_str, flags=re.IGNORECASE)
        input_str = re.sub(r"SUBDIVSION", "SUBDIVISION", input_str, flags=re.IGNORECASE)
        input_str = re.sub(r"RE-SUB", "RESUB", input_str, flags=re.IGNORECASE)
        input_str = re.sub(r"SUBD*\.*(?=\s|$)", "SUBDIVISION", input_str, flags=re.IGNORECASE)
        input_str = re.sub(r",(?: )?", " ", input_str, flags=re.IGNORECASE)
        input_str = re.sub(r" & ", " and ", input_str, flags=re.IGNORECASE)
        input_str = re.sub(r" no(\.)?\s*(?=\d+)", " ", input_str, flags=re.IGNORECASE)
        input_str = re.sub(r"#\s*(?=\d+)", "", input_str, flags=re.IGNORECASE)
        input_str = re.sub(r"(?<=.{5}) (?:an )?addition to (?:the city of )?.+", " ", input_str, flags=re.IGNORECASE)
        input_str = re.sub(r"\.\s*", " ", input_str, flags=re.IGNORECASE)
        input_str = re.sub(r" addition", " ", input_str, flags=re.IGNORECASE)
        input_str = re.sub(r" subdivision", " ", input_str, flags=re.IGNORECASE)
        input_str = re.sub(r"\s\s+", " ", input_str)
        return input_str.lower().strip()
    return ""


# ------------------------------------------------------------------ extensions (std_v1)
_UNITS = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen " \
         "sixteen seventeen eighteen nineteen".split()
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
_ORD_UNITS = "zeroth first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth thirteenth " \
             "fourteenth fifteenth sixteenth seventeenth eighteenth nineteenth".split()
_ORD_TENS = {"twentieth": 20, "thirtieth": 30, "fortieth": 40, "fiftieth": 50}
CARDINALS = {w: i for i, w in enumerate(_UNITS)} | _TENS
ORDINAL_WORDS = {w: i for i, w in enumerate(_ORD_UNITS)} | _ORD_TENS
_ROMAN = {"ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6, "vii": 7, "viii": 8, "ix": 9, "x": 10}
# keyword spellings repaired by fuzzy match (OCR noise); only tokens of >= 6 letters are tested
KEYWORDS = ["addition", "subdivision", "resubdivision", "rearrangement", "replat", "auditors", "outlots", "registered"]
KEYWORD_CUTOFF = 85


def ordinal(n):
    return f"{n}{'th' if 10 <= n % 100 <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def _number_words(s):
    """'twenty-first' -> '21st', 'twenty one' -> '21', 'second' -> '2nd', 'two' after no/number -> '2'."""
    tens = "|".join(_TENS)

    def compound(m):
        t, u = _TENS[m.group(1)], m.group(2)
        if u in ORDINAL_WORDS and 1 <= ORDINAL_WORDS[u] <= 9:
            return ordinal(t + ORDINAL_WORDS[u])
        return str(t + CARDINALS[u])
    units19 = "|".join(_UNITS[1:10] + _ORD_UNITS[1:10])
    s = re.sub(rf"\b({tens})[\s-]+({units19})\b", compound, s)
    s = re.sub(r"\b(" + "|".join(ORDINAL_WORDS) + r")\b", lambda m: ordinal(ORDINAL_WORDS[m.group(1)]), s)
    # cardinals become digits only where they act as a plat number: after no / number / #, or at the end
    card = "|".join(sorted(CARDINALS, key=len, reverse=True))
    s = re.sub(rf"\b(no|number|num|#)\.?\s*({card})\b", lambda m: f"no {CARDINALS[m.group(2)]}", s)
    s = re.sub(rf"\b(no|number|num|#)\.?\s*({'|'.join(_ROMAN)})\b", lambda m: f"no {_ROMAN[m.group(2)]}", s)
    s = re.sub(r"\bnumber\s*(?=\d)", "no ", s)
    return s


def _ocr_token(tok):
    """Digits inside a mostly-letter token are OCR noise (1->i, 0->o, 5->s): ADDIT1ON -> addition."""
    if re.fullmatch(r"\d+(st|nd|rd|th)?", tok) or not re.search(r"\d", tok):
        return tok
    letters = sum(c.isalpha() for c in tok)
    if letters >= 3 and letters >= 2 * sum(c.isdigit() for c in tok):
        return tok.translate(str.maketrans({"0": "o", "1": "i", "5": "s"}))
    return tok


def _keyword_fix(tok):
    if len(tok) < 6 or not tok.isalpha() or any(tok in k for k in KEYWORDS):   # truncations stay
        return tok
    best = max(KEYWORDS, key=lambda k: fuzz.ratio(tok, k))
    return best if fuzz.ratio(tok, best) >= KEYWORD_CUTOFF else tok


COUNTY_NAMES = ["anoka", "dakota", "hennepin", "milwaukee", "olmsted", "ramsey", "sherburne", "washington"]
_BOILER = [
    r"\s*,?\s*according to\b.*$",                         # according to the recorded plat thereof ...
    r"\s*,?\s*(?:now )?on file\b.*$",
    r"\s*,?\s*(?:as )?(?:of )?record(?:ed)? in\b.*$",
    r"\s*,?\s*(?:in |of )?(?:the )?county of\b.*$",
    r"\s*,?\s*(?:in |of )?(?:the )?state of\b.*$",
    r"(?<=\w)[\s,]+(?:in )?(?:the )?\b(?:" + "|".join(COUNTY_NAMES) + r") county\b.*$",   # not at the start
]


RULES = ("chars", "hyphen", "ocr_digits", "numbers", "replat", "rearrangement", "resub", "rls", "auditors",
         "outlots", "heights", "keyword_fix", "boilerplate", "leftover_words", "leading_the")


def standardize(s, skip=()):
    """std_v1 key of a plat / addition name ('' for empty input). `skip`: rule names to disable (ablation)."""
    if not isinstance(s, str) or not s.strip():
        return ""
    on = lambda r: r not in skip  # noqa: E731
    s = s.lower()
    if on("chars"):
        s = re.sub(r"[|~_\"“”«»*]", " ", s)
    if on("hyphen"):
        s = re.sub(r"(\w)-\s+(\w)", r"\1\2", s)                   # line-break hyphenation: addi- tion
    s = re.sub(r"\s+", " ", s).strip()
    if on("ocr_digits"):
        s = " ".join(_ocr_token(t) for t in s.split())
    if on("numbers"):
        s = _number_words(s)
    if on("replat"):
        s = re.sub(r"\bre[\s-]?plat(?:ted)?\b", "replat", s)
    if on("rearrangement"):
        s = re.sub(r"\bre[\s-]?arr(?:angement|gt|\.)?(?=\W|$)", "rearrangement", s)
    if on("resub"):
        s = re.sub(r"\bre[\s-]?sub(?:division|d)?\b\.?", "resubdivision", s)
    if on("rls"):
        s = re.sub(r"\b(?:registered land survey|r\.?\s?l\.?\s?s\.?)(?=\W|$)", "rls", s)
    if on("auditors"):
        s = re.sub(r"\baud(?:itor)?(?:'?s)?\b\.?", "auditors", s)     # aud / aud. / auditor's -> auditors
    if on("outlots"):
        s = re.sub(r"\bout\s?lots?\b", "outlots", s)
    if on("heights"):
        s = re.sub(r"\bhts\b|\bhghts?\b|\bhgts\b", "heights", s)
    if on("keyword_fix"):
        s = re.sub(r"[a-z]{6,}", lambda m: _keyword_fix(m.group(0)), s)
    if on("boilerplate"):
        for pat in _BOILER:
            s = re.sub(pat, "", s)
    s = s.replace("resubdivision", "resubdivisionx")                  # protect from the port's SUBD rule
    s = standardize_addition_mp(s)
    s = s.replace("resubdivisionx", "resubdivision")
    if on("leftover_words"):
        s = re.sub(r"\b(?:add|addn|addition|sub|subdiv|subdivision)\b", " ", s)   # e.g. a leading "addition"
    s = re.sub(r"\bno\s+(?=\d)", "", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    if on("leading_the"):
        s = re.sub(r"^the\s+", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


NORMALIZERS = {"mp": standardize_addition_mp, "std_v1": standardize}


def number_words(s):
    """Public wrapper: number words -> digits on a lower-case string (used by diagnostics)."""
    return _number_words(s.lower())


def plat_numbers(s):
    """Numbers that identify a plat (ordinals and bare numbers) from a std key: 'oak 2nd' -> {2}."""
    return {int(m.group(1)) for m in re.finditer(r"\b(\d+)(?:st|nd|rd|th)?\b", s or "")}
