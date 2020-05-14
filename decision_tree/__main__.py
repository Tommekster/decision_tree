#! /usr/bin/env python3

from typing import List, Tuple, Union

from . import diagram
from . import entropy
from . import table_repository
from .decision_tree_generator import DecisionTreeGenerator
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
        ("Outlook", "Temperature", "Humidity", "Wind", "Play golf")
    )


def load_cars() -> Tuple[List[Tuple[str]], Tuple[Union[int, str], ...]]:
    return table_repository.parse_table(
        "data/cars/car.data",
        ("buying", "maint", "doors", "persons", "lug_boot", "safety", "class")
    )


def generate_trees(alphas: List[float], results: dict, json_results: bool, diagram_results: bool):
    for alpha in alphas:
        generator = DecisionTreeGenerator(lambda x: entropy.renyi_entropy(x, alpha))
        print("alpha =", alpha)
        print("Total entropy", generator.entropy(table))
        print("Total entropy for target", generator.entropy([x[-1] for x in table]))
        print("first max gain {1} has index {0}".format(*generator.select_feature_index(table)))
        print("")

        tree = generator.create_tree(table, labels=labels)
        results[alpha] = tree.to_json()
        if json_results:
            with open("output_{}.json".format(alpha), "w") as f:
                f.write(results[alpha] + "\n")

        if diagram_results:
            diagram_file = "diagram{}.png".format(alpha)
            graph = diagram.create_graph(tree)
            diagram.save_graph(graph, diagram_file)


def compare_results(alphas: List[float], results: dict):
    header = "\t".join([""] + [str(a) for a in alphas])
    print(header)
    for x in alphas:
        print("\t".join([str(x)] + ["=" if results[x] == results[y] else " " for y in alphas]))


if __name__ == "__main__":
    table, labels = load_golf()
    print("\t".join(["Target", "Cnt", "%"]))
    for target, cnt, frac in get_target_distribution(table):
        print("\t".join([target, str(cnt), str(frac * 100)]))
    print("")

    alphas = [0.0, 0.5, 1.0, 2.0, 99, float("inf")]
    results = {}

    generate_trees(alphas, results, json_results=False, diagram_results=True)
    compare_results(alphas, results)
