"""
Runtime-checkable protocols for tree, forest, and forest-sum types.

Both the non-planar (Tree/Forest/ForestSum) and planar
(PlanarTree/OrderedForest/OrderedForestSum) types satisfy these
structurally, enabling generic algebra code to work with either family.
"""
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class TreeLike(Protocol):
    list_repr: Any
    unlabelled_repr: Any

    def nodes(self) -> int: ...


@runtime_checkable
class ForestLike(Protocol):
    tree_list: tuple

    def simplify(self) -> "ForestLike": ...


@runtime_checkable
class ForestSumLike(Protocol):
    term_list: tuple

    def simplify(self) -> "ForestSumLike": ...
