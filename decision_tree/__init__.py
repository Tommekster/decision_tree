#! /usr/bin/env python3

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
    pass
