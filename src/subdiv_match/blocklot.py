"""Block / lot strings -> comparable sets (in the spirit of get_blocks / get_lots in racial_covenants_processor).

'Block 03' -> '3'; 'A' -> 'A'; 'Lots 3 thru 5 and 9' -> {'3', '4', '5', '9'}; 'Lot 4A' -> {'4A'}.
"""
import re

from .normalize import CARDINALS

MAX_RANGE = 200


def _canon(tok):
    tok = tok.strip().upper()
    if re.fullmatch(r"\d+[A-Z]?", tok):
        return tok.lstrip("0") or "0"
    if tok.lower() in CARDINALS:
        return str(CARDINALS[tok.lower()])
    return tok if re.fullmatch(r"[A-Z]", tok) else None


def parse_block(s):
    if not isinstance(s, str) or not s.strip():
        return None
    s = re.sub(r"^\s*(?:bl(?:oc)?k|blk)\.?\s*", "", s, flags=re.IGNORECASE)
    m = re.match(r"\s*(\d+[A-Za-z]?|[A-Za-z]+)\b", s)
    return _canon(m.group(1)) if m else None


def parse_lots(s):
    if not isinstance(s, str) or not s.strip():
        return set()
    s = re.sub(r"\blots?\b\.?", " ", s, flags=re.IGNORECASE)
    out = set()
    for a, b in re.findall(r"(\d+)\s*(?:-|to|thru|through)\s*(\d+)", s, flags=re.IGNORECASE):
        a, b = int(a), int(b)
        if 0 <= b - a <= MAX_RANGE:
            out.update(str(i) for i in range(a, b + 1))
    s = re.sub(r"(\d+)\s*(?:-|to|thru|through)\s*(\d+)", " ", s, flags=re.IGNORECASE)
    for tok in re.split(r"[\s,;&/]+|\band\b", s, flags=re.IGNORECASE):
        c = _canon(tok) if tok else None
        if c and c not in {"PART", "OF"}:
            out.add(c)
    return out
