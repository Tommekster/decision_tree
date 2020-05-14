#! /usr/bin/env python3

from dataclasses import dataclass
from typing import List, Union

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
    branches: List[NodeBranch]
