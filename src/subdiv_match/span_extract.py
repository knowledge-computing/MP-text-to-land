"""Helpers for span-extraction models with a word limit (GLiNER): split a text into overlapping chunks that the model
can read whole, and merge the per-chunk entities back into one ordered list of names. No model code here.

GLiNER splits text with the regex below (gliner.data_processing.tokenizer.WhitespaceTokenSplitter) and silently drops
every token after config.max_len, so chunks are cut on the same tokens. Each chunk "owns" its tokens minus half of
the overlap on each inner side; an entity is kept only by the chunk that owns its first token, so an entity cut by a
chunk edge (and seen whole in the neighbour) is not kept twice or in truncated form.
"""
import re

TOKEN = re.compile(r"\w+(?:[-_]\w+)*|\S")
EDGE = " \t\r\n,;:"


def token_chunks(text, max_tokens, overlap):
    """-> list of (char_start, char_end, own_start, own_end): chunk span and the char range it owns."""
    if not 0 <= overlap < max_tokens:
        raise ValueError("need 0 <= overlap < max_tokens")
    toks = [m.span() for m in TOKEN.finditer(text)]
    if not toks:
        return []
    out, step, a = [], max_tokens - overlap, 0
    while True:
        b = min(a + max_tokens, len(toks))
        first, last = a == 0, b == len(toks)
        oa = a if first else a + overlap // 2
        ob = b if last else b - (overlap - overlap // 2)
        out.append((toks[a][0], toks[b - 1][1], toks[oa][0], toks[ob][0] if ob < len(toks) else len(text)))
        if last:
            return out
        a += step


def merge_entities(text, chunks, per_chunk):
    """chunks: from token_chunks; per_chunk: entity dicts with chunk-relative 'start' / 'end'.
    -> distinct names (case-insensitive, whitespace-collapsed) in order of their first position in text."""
    spans = set()
    for (cs, _, os_, oe), ents in zip(chunks, per_chunk):
        for e in ents:
            s, t = cs + int(e["start"]), cs + int(e["end"])
            if os_ <= s < oe:
                spans.add((s, t))
    names, seen = [], set()
    for s, t in sorted(spans):
        name = re.sub(r"\s+", " ", text[s:t]).strip(EDGE)
        key = name.casefold()
        if name and key not in seen:
            seen.add(key)
            names.append(name)
    return names
