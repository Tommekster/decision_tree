#! /usr/bin/env python3

import math
from typing import List, Hashable

from .group_by import group_by


def renyi_entropy(rows: List[Hashable], alpha: float) -> float:
    assert alpha >= 0.0
    if alpha == 0:
        return max_entropy(rows)
    elif alpha == 1:
        return shannon_entropy(rows)
    elif alpha == float("inf"):
        return min_entropy(rows)
    probabilities = __get_probabilities__(rows)
    return math.log2(sum(math.pow(x, alpha) for x in probabilities)) / (1 - alpha)


def shannon_entropy(rows: List[Hashable]) -> float:
    probabilities = __get_probabilities__(rows)
    return -sum(p * math.log2(p) for p in probabilities)


def max_entropy(rows: List[Hashable]) -> float:
    return math.log2(len(rows))


def min_entropy(rows: List[Hashable]) -> float:
    probabilities = __get_probabilities__(rows)
    return -math.log2(max(probabilities))


def __get_probabilities__(rows: List[Hashable]) -> List[float]:
    groups = group_by(rows, lambda x: x)
    total_count = len(rows)
    return [
        len(group) / total_count
        for key, group in groups.items()
    ]
