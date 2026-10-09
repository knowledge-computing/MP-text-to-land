"""
OCR confusion map shared by the synthetic generator (noise) and the candidate extractor (digit repair).

The pairs are generic typewriter/OCR confusions, not learned from deed text.
"""

# digit -> characters OCR commonly reads it as
DIGIT_LOOKALIKES = {
    "0": "OoDQ", "1": "lIi|", "2": "Zz", "3": "8", "4": "A",
    "5": "Ss", "6": "bG", "7": "T", "8": "B3", "9": "gq",
}
# non-digit character -> the digit it most likely stands for inside a number
TO_DIGIT = {ch: d for d, chars in DIGIT_LOOKALIKES.items() for ch in chars if not ch.isdigit()}

# multi- or single-character confusions inside words (read left -> right)
LETTER_CONFUSIONS = (
    ("rn", "m"), ("m", "rn"), ("cl", "d"), ("d", "cl"), ("vv", "w"), ("w", "vv"),
    ("e", "c"), ("c", "e"), ("h", "b"), ("n", "u"), ("u", "n"), ("t", "f"), ("i", "l"),
    ("l", "1"), ("I", "1"), ("O", "0"), ("o", "0"), ("S", "5"), ("s", "5"), ("B", "8"), ("Z", "2"),
)

# operation weights for ocr_noise
_OPS = (("substitute", 0.6), ("delete", 0.1), ("duplicate", 0.05), ("insert_space", 0.1), ("delete_space", 0.15))


def _substitutions(text, i):
    out = []
    ch = text[i]
    if ch.isdigit():
        out.extend((1, c) for c in DIGIT_LOOKALIKES.get(ch, ""))
    for src, dst in LETTER_CONFUSIONS:
        if text.startswith(src, i):
            out.append((len(src), dst))
    return out


def ocr_noise(text, rng, p=0.05):
    """Corrupt `text` with probability `p` per character. `rng` is a random.Random (seeded by the caller)."""
    if p <= 0 or not text:
        return text
    ops, weights = zip(*_OPS)
    out, i = [], 0
    while i < len(text):
        ch = text[i]
        if rng.random() >= p:
            out.append(ch)
            i += 1
            continue
        op = rng.choices(ops, weights)[0]
        if op == "substitute":
            subs = _substitutions(text, i)
            if subs:
                n, dst = rng.choice(subs)
                out.append(dst)
                i += n
                continue
            out.append(ch)
        elif op == "delete":
            pass
        elif op == "duplicate":
            out.append(ch + ch)
        elif op == "insert_space":
            out.append(ch + " " if ch != " " else ch)
        elif op == "delete_space":
            if ch != " ":
                out.append(ch)
        i += 1
    return "".join(out)
