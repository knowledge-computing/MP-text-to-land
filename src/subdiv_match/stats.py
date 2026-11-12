"""Paired tests for the extraction x matcher test run (docs/preregistration_test.md section 6). Standard library only."""
from math import comb


def mcnemar_exact(b, c):
    """Exact two-sided McNemar p: b, c = discordant counts (only X right, only Y right).
    p = min(1, 2 * sum_{k <= min(b, c)} C(n, k) / 2^n), n = b + c; p = 1 when n = 0. Integer arithmetic, so it is
    exact before the final division."""
    if b < 0 or c < 0:
        raise ValueError("discordant counts must be >= 0")
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(min(b, c) + 1))
    return min(1.0, 2 * tail / 2 ** n)


def holm(pvalues):
    """Holm step-down adjusted p-values, in the input order: sort ascending, adjusted p(i) = max_{j <= i}
    min(1, (m - j + 1) p(j))."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adj, running = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adj[i] = running
    return adj


def discordant(x, y):
    """Per-deed 0/1 outcomes of two pipelines on the same deeds -> (b, c) = (only x right, only y right)."""
    if len(x) != len(y):
        raise ValueError("outcome vectors differ in length")
    b = sum(1 for p, q in zip(x, y) if p and not q)
    c = sum(1 for p, q in zip(x, y) if q and not p)
    return b, c
