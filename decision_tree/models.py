#! /usr/bin/env python3

from dataclasses import dataclass
from typing import List, Union

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass(init=True, repr=True, eq=True)
class Leaf:
    value: str
    count: int


@dataclass_json
@dataclass(init=True, repr=True, eq=True)
class NodeBranch:
    value: str
    children: Union[List, Leaf]


@dataclass_json
@dataclass(init=True, repr=True)
class DecisionNode:
    label: Union[str, int]
    gain: float
    branches: List[NodeBranch]

    def __eq__(self, other):
        # skip grain
        if not isinstance(other, DecisionNode):
            return False
        return self.label == other.label and self.branches == other.branches
