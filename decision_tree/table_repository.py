#! /usr/bin/env python3

from typing import List, Tuple, Union


def parse_table(path: str, labels: Tuple[Union[int, str], ...] = None, delimiter: str = ",") \
        -> Tuple[List[Tuple[str]], Tuple[Union[int, str], ...]]:
    def read_file():
        with open(path, "r") as f:
            for line in f:
                yield tuple(x.strip() for x in line.split(delimiter))

    table = list(read_file())
    labels = labels or tuple(n for n, _ in enumerate(table[0]))
    return table, labels
