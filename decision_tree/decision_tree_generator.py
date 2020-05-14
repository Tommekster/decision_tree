#! /usr/bin/env python3

from typing import List, Tuple, Union, Any, Callable, Hashable

from .group_by import group_by
from .models import Leaf, NodeBranch, DecisionNode


class DecisionTreeGenerator:
    entropy: Callable[[List[Hashable]], float]

    def __init__(self, entropy: Callable[[List[Hashable]], float]):
        self.entropy = entropy

    def create_tree(self, table: List[Tuple[str]], labels: Tuple[Union[str, int], ...] = None) \
            -> Union[DecisionNode, List[Leaf]]:
        labels = labels or tuple(range(len(table[0])))
        if len(labels) == 1:
            raise RuntimeError
        feature, gain = self.select_feature_index(table)
        if gain == 0 or len(labels) == 2:
            return [Leaf(value=key, count=len(vals)) for key, vals in group_by(table, lambda x: x[-1]).items()]
        groups = group_by(table, lambda x: x[feature])
        label = labels[feature]
        return DecisionNode(
            label=label,
            gain=gain,
            branches=[
                NodeBranch(
                    value=key,
                    children=self.create_tree(
                        self.__subselect_table__(values, feature, key),
                        self.__skip_index__(labels, feature)
                    )
                )
                for key, values in groups.items()
            ])

    def select_feature_index(self, table: List[Tuple[str]]) -> Tuple[int, float]:
        marginal_entropies = [self.entropy([x[n] for x in table]) for n, _ in enumerate(table[0])]
        joint_entropies = [self.entropy([(x[n], x[-1]) for x in table]) for n, _ in enumerate(table[0])]
        conditional_entropies = [H_x_y - H_y for H_x_y, H_y in zip(joint_entropies, marginal_entropies)]
        information_gains = [marginal_entropies[-1] - H_xIy for H_xIy in conditional_entropies]
        sorted_gains = sorted(enumerate(information_gains[:-1]), key=lambda x: x[1], reverse=True)
        return sorted_gains[0]

    def __subselect_table__(self, table: List[Tuple[str]], index, value):
        return [
            self.__skip_index__(row, index)
            for row in table
            if row[index] == value
        ]

    def __skip_index__(self, row: Tuple[Any], skip_index) -> Tuple[Any]:
        return tuple(v for n, v in enumerate(row) if n != skip_index)
