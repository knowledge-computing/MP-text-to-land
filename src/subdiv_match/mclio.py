"""Subdivision names from Milwaukee County (MCLIO) parcel legal descriptions (public data).

A legal description typically reads "<SUBDIVISION NAME> <quarter / section / T-R> <LOT ..> <BLK ..> <dims>", e.g.
"TOY HEIGHTS ADD 1 NE 1/4 SEC 3 5 22 LOT 9 BLK 4 64 X 141". The name is the prefix before the first locator
token. Descriptions that start with a metes-and-bounds or aliquot-part call, condominium units and certified
survey maps yield no name.
"""
import re

from .blocklot import parse_block, parse_lots

_Q = r"(?:NW|NE|SW|SE|N|S|E|W)"
STOP = re.compile(r"""\s(?:
    """ + _Q + r"""(?:\s*(?:&|AND)\s*""" + _Q + r""")*\s+1/[24]      # quarter section
  | (?:NW|NE|SW|SE)\s*\d                                          # quarter with the slash / space lost
  | \d{1,2}/\d{1,2}/\d{1,2}                                        # S/T/R
  | IN\s+(?:THE\s+)?""" + _Q + r"""\b
  | SEC(?:TION)?\.?\s*\d
  | \d{1,2}-\d{1,2}-\d{1,2}\b                                    # S-T-R
  | LOTS?\s+[\dA-Z]
  | (?:BLK|BLOCK)S?\.?\s+[\dA-Z]
  | UNIT\s | BLDG\s | OUTLOTS?\b | PARCEL\s+\d
  | V\d+\s*P\d+ | VOL\.?\s+\d
  | COMM?\.?\s+(?:AT|\d) | BEG\b
  | (?:THAT\s+)?(?:PART|PT)\.?\s+OF\b
  | (?:PART|PT|EXC|EXCEPT|ALSO|UND|UNDIV)\b\.?                     # part / exception / undivided calls
  | (?:[NSEW]{1,2}|NLY|SLY|ELY|WLY)\s+[\d.]+\s*(?:FT|FEET|')      # partial-lot call: N 50 FT OF ...
  | (?:[NSEW]{1,2}|NLY|SLY|ELY|WLY)\s+(?:\d/\d|HALF|QUARTER)\b      # N 1/2 OF / N HALF OF ...
  | [\d.]+\s*(?:FT|FEET)\b
)""", re.X)
BAD_START = re.compile(r"""^(?:COMM?\b|BEG\b|PART\b|PT\b|THAT\b|ALL\b|LOTS?\b|BLK\b|BLOCK\b|SEC\b|UNIT\b|CSM\b|
    CERTIFIED\b|C\s?S\s?M\b|LEGAL\s+DESCRIPTION\b|""" + _Q + r"""\s*\d|""" + _Q + r"""\s+1/|\d)""", re.X)
BLOCK = re.compile(r"\b(?:BLK|BLOCK)\.?\s+(\d+[A-Z]?|[A-Z])\b")
LOTS = re.compile(r"\bLOTS?\s+((?:\d+[A-Z]?|[A-Z])(?:\s*(?:&|AND|,|-|TO|THRU)\s*(?:\d+[A-Z]?|[A-Z])\b)*)")


def extract(legal, condo_name=None):
    """-> (subdivision name or None, block or None, set of lots)."""
    if not isinstance(legal, str) or not legal.strip():
        return None, None, set()
    s = re.sub(r"\s+", " ", legal.upper()).strip()
    b = BLOCK.search(s)
    lots = set()
    for m in LOTS.finditer(s):
        lots |= parse_lots(m.group(1))
    block = parse_block(b.group(1)) if b else None
    if (isinstance(condo_name, str) and condo_name.strip()) or "CONDOMINIUM" in s or BAD_START.match(s):
        return None, block, lots
    m = STOP.search(" " + s)
    name = (" " + s)[:m.start()].strip() if m else s
    name = re.sub(r"(?:[\s,;-]+|\s+ETC\.?)+$", "", name)
    if sum(c.isalpha() for c in name) < 4:
        return None, block, lots
    return name, block, lots
