#! /usr/bin/env python3

from typing import List, Tuple, Callable, Dict, Hashable, Union


def group_by(table: List[Hashable], selector: Callable[[Union[Hashable, Tuple[str]]], Hashable]) \
        -> Dict[Union[str, Tuple[str]], List[Tuple[str]]]:
    groups = dict()
    for row in table:
        target = selector(row)
        groups.setdefault(target, []).append(row)
    return groups
