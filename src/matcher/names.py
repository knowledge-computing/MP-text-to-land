"""Plat-name normalization.

n1: upper-case, punctuation -> space, collapse spaces (ablation step 1).
NameNormalizer (step 3): n1, then 2->1 token merges (e.g. "AUDITOR S" -> "AUDITORS"), then every token is
mapped to the canonical member of its equivalence class. Classes come from union-find over 1:1 word
substitutions seen at least `min_freq` times between a plat layer's name field and its alias/label field,
plus a fixed ordinal table (FIRST..TWELFTH <-> 1ST..12TH). Pure-number substitutions are ignored (they are
plat-number changes, not spelling variants). Canonical member = a form containing a digit, then the shortest.
"""
import difflib
import re
from collections import Counter, defaultdict

_SUFFIX = {1: "ST", 2: "ND", 3: "RD"}
ORDINALS = {w: f"{i}{'TH' if 10 <= i % 100 <= 13 else _SUFFIX.get(i % 10, 'TH')}"
            for i, w in enumerate("FIRST SECOND THIRD FOURTH FIFTH SIXTH SEVENTH EIGHTH NINTH TENTH ELEVENTH "
                                  "TWELFTH".split(), 1)}
_NUM = re.compile(r"[\d ]+")


def n1(s):
    return re.sub(r"\s+", " ", re.sub(r"[^A-Z0-9 ]", " ", str(s).upper())).strip()


def substitution_counts(pairs):
    """pairs of (name, alias) strings -> Counter of (a, b) replacements: equal-length blocks token by token,
    otherwise phrases of <= 2 tokens per side."""
    counts = Counter()
    for a, b in pairs:
        if not (isinstance(a, str) and isinstance(b, str)):
            continue
        ta, tb = n1(a).split(), n1(b).split()
        for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, ta, tb, autojunk=False).get_opcodes():
            if op != "replace":
                continue
            if i2 - i1 == j2 - j1:  # equal-length block: pair tokens one to one
                counts.update(zip(ta[i1:i2], tb[j1:j2]))
            elif i2 - i1 <= 2 and j2 - j1 <= 2:
                counts[(" ".join(ta[i1:i2]), " ".join(tb[j1:j2]))] += 1
    return counts


def learn_rules(pairs, min_freq=5):
    """-> {"canon": {token: canonical}, "merges": {"A B": "AB"}, "used": {"a->b": freq}}."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    merges, used = {}, {}
    for (a, b), n in sorted(substitution_counts(pairs).items()):
        if n < min_freq or _NUM.fullmatch(a) or _NUM.fullmatch(b):
            continue
        if " " not in a and " " not in b:
            parent[find(a)] = find(b)
            used[f"{a}->{b}"] = n
        elif a.replace(" ", "") == b.replace(" ", ""):
            merges[a] = b
            used[f"{a}->{b}"] = n
    for w, o in ORDINALS.items():
        parent[find(w)] = find(o)
    classes = defaultdict(set)
    for x in list(parent):
        classes[find(x)].add(x)
    canon = {}
    for members in classes.values():
        rep = min(members, key=lambda m: (not re.search(r"\d", m), len(m), m))
        canon.update({m: rep for m in members})
    return {"canon": canon, "merges": merges, "used": used}


class NameNormalizer:
    def __init__(self, rules):
        self.canon = dict(rules.get("canon", {}))
        self.merges = [(re.compile(rf"\b{re.escape(k)}\b"), v) for k, v in
                       sorted(rules.get("merges", {}).items(), key=lambda kv: -len(kv[0]))]

    def __call__(self, s):
        s = n1(s)
        for pat, v in self.merges:
            s = pat.sub(v, s)
        return " ".join(self.canon.get(t, t) for t in s.split())
