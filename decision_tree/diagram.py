#! /usr/bin/env python3
import tempfile
from typing import List, Union

import blockdiag.command
import graphviz

from .models import DecisionNode, Leaf


def create_graph(tree: Union[DecisionNode, List[Leaf]]) -> str:
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

    return graph.source.replace("digraph", "blockdiag")


def save_graph(graph: str, output_file: str) -> None:
    with tempfile.NamedTemporaryFile() as temp_file:
        temp_file.write(graph)
        temp_file.close()
        blockdiag.command.main(["-o", output_file, temp_file.name])
