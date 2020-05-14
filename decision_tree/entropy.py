#! /usr/bin/env python3

import math
from typing import List, Hashable

from .group_by import group_by


def shannon_entropy(rows: List[Hashable]) -> float:
    groups = group_by(rows, lambda x: x)
    total_count = len(rows)
    probabilities = [
        len(group) / total_count
        for key, group in groups.items()
    ]
    return -sum(p * math.log2(p) for p in probabilities)
