#! /usr/bin/env python3

from typing import List, Tuple


def parse_table(path: str, delimiter: str = ",") -> List[Tuple[str]]:
    def read_file():
        with open(path, "r") as f:
            for line in f:
                yield tuple(x.strip() for x in line.split(delimiter))

    return list(read_file())


if __name__ == "__main__":
    for l in parse_table("../data/cars/car.data"):
        print("\t".join(l))
    pass
