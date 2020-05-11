#! /usr/bin/env python3

import math
from typing import List, Tuple, Callable, Dict, Hashable


def parse_table(path: str, delimiter: str = ",") -> List[Tuple[str]]:
    def read_file():
        with open(path, "r") as f:
            for line in f:
                yield tuple(x.strip() for x in line.split(delimiter))

    return list(read_file())


def get_target_distribution(table: List[Tuple[str]]):
    groups = group_by(table, lambda x: x[-1])
    total_rows = len(table)
    return [
        (key, len(vals), len(vals) / total_rows)
        for key, vals in groups.items()
    ]


def entropy(rows: List[Hashable]):
    groups = group_by(rows, lambda x: x)
    total_count = len(rows)
    probabilities = [
        len(group) / total_count
        for key, group in groups.items()
    ]
    return -sum(p * math.log2(p) for p in probabilities)


def group_by(table: List[Tuple[str]], selector: Callable[[Tuple[str]], Hashable]) -> Dict[Hashable, List[Tuple[str]]]:
    groups = dict()
    for row in table:
        target = selector(row)
        groups.setdefault(target, []).append(row)
    return groups


if __name__ == "__main__":
    table = parse_table("../data/cars/car.data")
    print("\t".join(["Target", "Cnt", "%"]))
    for target, cnt, frac in get_target_distribution(table):
        print("\t".join([target, str(cnt), str(frac * 100)]))
    print("Total entropy", entropy(table))
    print("Total entropy", entropy([x[-1] for x in table]))

    marginals = [entropy([x[n] for x in table]) for n,_ in enumerate(table[0])]
    print("Marginal entropies", *marginals)

    joins = [entropy([(x[n],x[-1]) for x in table]) for n,_ in enumerate(table[0])]
    print("Joint entropies", *joins)

    conditionals = [H_x_y - H_y for H_x_y, H_y in zip(joins, marginals)]
    print("conditionals", *conditionals)

    gains = [marginals[-1] - H_xIy for H_xIy in conditionals]
    print("gains", *gains)