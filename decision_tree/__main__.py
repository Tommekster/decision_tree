#! /usr/bin/env python3

from typing import List, Tuple, Union

import blockdiag.command
import graphviz

from . import entropy
from . import table_repository
from .decision_tree_generator import DecisionTreeGenerator
from .group_by import group_by
from .models import DecisionNode, Leaf


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
        ("Outlook", "Temperature", "Humidity", "Wind", "Play golf")
    )


def load_cars() -> Tuple[List[Tuple[str]], Tuple[Union[int, str], ...]]:
    return table_repository.parse_table(
        "data/cars/car.data",
        ("buying", "maint", "doors", "persons", "lug_boot", "safety", "class")
    )


def export_graph(tree: Union[DecisionNode, List[Leaf]]) -> str:
    graph = graphviz.Digraph()
    index = [0]

    def get_index():
        index[0] += 1
        return index[0]

    def add_node(node, node_index):
        if isinstance(node, DecisionNode):
            graph.node(str(node_index), node.label)
            for branch in node.branches:
                if isinstance(branch.children, DecisionNode):
                    child_index = get_index()
                    add_node(branch.children, child_index)
                    graph.edge(str(node_index), str(child_index), label=branch.value)
                elif isinstance(branch.children, list):
                    child_index = get_index()
                    graph.node(
                        str(child_index),
                        "\\n".join(
                            "{} ({})".format(child.value, child.count)
                            for child in branch.children
                        )
                    )
                    graph.edge(str(node_index), str(child_index), label=branch.value)
                else:
                    raise NotImplementedError
        else:
            raise NotImplementedError

    add_node(tree, get_index())

    return graph.source


if __name__ == "__main__":
    generator = DecisionTreeGenerator(entropy.shannon_entropy)
    table, labels = load_golf()
    print("\t".join(["Target", "Cnt", "%"]))
    for target, cnt, frac in get_target_distribution(table):
        print("\t".join([target, str(cnt), str(frac * 100)]))
    print("Total entropy", entropy.shannon_entropy(table))
    print("Total entropy for target", entropy.shannon_entropy([x[-1] for x in table]))
    print("max gain {} has index {}".format(*generator.select_feature_index(table)))
    tree = generator.create_tree(table, labels=labels)

    with open("output.json", "w") as f:
        f.write(tree.to_json() + "\n")
    with open("diag.diag", "w") as f:
        f.write(export_graph(tree).replace("digraph", "blockdiag"))
    blockdiag.command.main(["diag.diag"])
    # print(tree.to_json(indent=2))
