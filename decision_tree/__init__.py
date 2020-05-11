#! /usr/bin/env python3

import math
from dataclasses import dataclass
from typing import List, Tuple, Callable, Dict, Hashable, Union, Any

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass(init=True, repr=True)
class Leaf:
    value: str
    count: int


@dataclass_json
@dataclass(init=True, repr=True)
class NodeBranch:
    value: str
    children: Union[List, Leaf]


@dataclass_json
@dataclass(init=True, repr=True)
class DecisionNode:
    label: Union[str, int]
    branches: List


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


def group_by(table: List[Tuple[str]], selector: Callable[[Tuple[str]], Hashable]) -> Dict[
    Union[str, Tuple[str]], List[Tuple[str]]]:
    groups = dict()
    for row in table:
        target = selector(row)
        groups.setdefault(target, []).append(row)
    return groups


def select_feature_index(table: List[Tuple[str]]) -> Tuple[int, float]:
    marginal_entropies = [entropy([x[n] for x in table]) for n, _ in enumerate(table[0])]
    joint_entropies = [entropy([(x[n], x[-1]) for x in table]) for n, _ in enumerate(table[0])]
    conditional_entropies = [H_x_y - H_y for H_x_y, H_y in zip(joint_entropies, marginal_entropies)]
    information_gains = [marginal_entropies[-1] - H_xIy for H_xIy in conditional_entropies]
    sorted_gains = sorted(enumerate(information_gains[:-1]), key=lambda x: x[1], reverse=True)
    return sorted_gains[0]


def create_tree(table: List[Tuple[str]], labels: Tuple[Union[str, int], ...] = None) -> Union[DecisionNode, List[Leaf]]:
    labels = labels or tuple(range(len(table[0])))
    if len(labels) == 1:
        raise RuntimeError
    feature, gain = select_feature_index(table)
    if gain == 0 or len(labels) == 2:
        return [Leaf(value=key, count=len(vals)) for key, vals in group_by(table, lambda x: x[-1]).items()]
    groups = group_by(table, lambda x: x[feature])
    label = labels[feature]
    return DecisionNode(
        label=label,
        branches=[
            NodeBranch(
                value=key,
                children=create_tree(subselect_table(values, feature, key), skip_index(labels, feature))
            )
            for key, values in groups.items()
        ])


def subselect_table(table: List[Tuple[str]], index, value):
    return [
        skip_index(row, index)
        for row in table
        if row[index] == value
    ]


def skip_index(row: Tuple[Any], skip_index) -> Tuple[Any]:
    return tuple(v for n, v in enumerate(row) if n != skip_index)


if __name__ == "__main__":
    table = parse_table("../data/golf/golf.data")
    print("\t".join(["Target", "Cnt", "%"]))
    for target, cnt, frac in get_target_distribution(table):
        print("\t".join([target, str(cnt), str(frac * 100)]))
    print("Total entropy", entropy(table))
    print("Total entropy for target", entropy([x[-1] for x in table]))
    print("max gain {} has index {}".format(*select_feature_index(table)))
    tree = create_tree(table, labels=("Outlook", "Temperature", "Humidity", "Wind", "Play golf"))
    print(tree.to_json(indent=2))
