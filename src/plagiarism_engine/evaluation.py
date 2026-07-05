from typing import Set, List, Tuple


def exact_jaccard(set_a: Set[str], set_b: Set[str]) -> float:
    if not set_a and not set_b:
        return 0.0

    intersection_size = len(set_a.intersection(set_b))
    union_size = len(set_a.union(set_b))

    if union_size == 0:
        return 0.0

    return intersection_size / union_size
