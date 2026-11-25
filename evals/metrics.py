"""Pure metric functions for the evaluation harness. No I/O, no network, no LLM."""
from collections import Counter


def recall(found, reference):
    """Share of reference items that were found. Both are sets of IDs or titles."""
    if not reference:
        raise ValueError("reference set is empty")
    return len(set(found) & set(reference)) / len(set(reference))


def _average_ranks(values):
    """1-based ranks; tied values share the average of their positions."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(xs, ys):
    """Spearman rank correlation. None if there are fewer than 2 points or either side is constant."""
    if len(xs) != len(ys):
        raise ValueError("xs and ys must have the same length")
    if len(xs) < 2:
        return None
    rx, ry = _average_ranks(xs), _average_ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx) ** 0.5
    vy = sum((b - my) ** 2 for b in ry) ** 0.5
    if vx == 0 or vy == 0:
        return None
    return cov / (vx * vy)


def agreement(scores, labels, threshold):
    """Share of papers where 'score >= threshold' matches 'label >= threshold'."""
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have the same length")
    if not scores:
        raise ValueError("no papers to compare")
    same = sum(1 for s, l in zip(scores, labels) if (s >= threshold) == (l >= threshold))
    return same / len(scores)


def saturation(scores):
    """Share of papers that received the most common score. 1.0 means every paper got the same score."""
    if not scores:
        raise ValueError("no scores")
    return Counter(scores).most_common(1)[0][1] / len(scores)
