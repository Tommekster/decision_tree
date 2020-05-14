#! /usr/bin/env python3

from typing import List, Tuple, Union

from . import table_repository
from .decision_tree_generator import DecisionTreeGenerator
from .entropy import entropy
from .group_by import group_by


def get_target_distribution(table: List[Tuple[str]]):
    groups = group_by(table, lambda x: x[-1])
    total_rows = len(table)
    return [
        (key, len(vals), len(vals) / total_rows)
        for key, vals in groups.items()
    ]


def load_golf() -> Tuple[List[Tuple[str]], Tuple[Union[int, str], ...]]:
    return table_repository.parse_table(
        "data/golf/golf.data",
        ("Outlook", "Temperature", "Humidity", "Wind", "Play golf"))


def load_cars() -> Tuple[List[Tuple[str]], Tuple[Union[int, str], ...]]:
    return table_repository.parse_table(
        "data/cars/car.data",
        ("buying", "maint", "doors", "persons", "lug_boot", "safety", "class"))


if __name__ == "__main__":
    generator = DecisionTreeGenerator()
    table, labels = load_cars()
    print("\t".join(["Target", "Cnt", "%"]))
    for target, cnt, frac in get_target_distribution(table):
        print("\t".join([target, str(cnt), str(frac * 100)]))
    print("Total entropy", entropy(table))
    print("Total entropy for target", entropy([x[-1] for x in table]))
    print("max gain {} has index {}".format(*generator.select_feature_index(table)))
    tree = generator.create_tree(table, labels=labels)
    with open("output.json", "w") as f:
        f.write(tree.to_json() + "\n")
    print(tree.to_json(indent=2))
