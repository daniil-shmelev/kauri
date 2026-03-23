"""Ordered-tree basis objects for truncated verification."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import sympy

from kauri.trees import Tree, ForestSum
from kauri.utils import _check_valid, _nodes, _to_labelled_tuple, _to_unlabelled_tuple
from kauri._protocols import TreeLike


@dataclass(frozen=True)
class PlanarTree:
    """Ordered rooted tree; sibling order is part of identity."""

    list_repr: tuple | list | None = None
    unlabelled_repr: tuple | None = None

    def __post_init__(self) -> None:
        if self.list_repr is None:
            object.__setattr__(self, "unlabelled_repr", None)
            return
        if not _check_valid(self.list_repr):
            raise ValueError(f"{self.list_repr!r} is not a valid planar tree representation.")
        tuple_repr: tuple = _to_labelled_tuple(self.list_repr)
        object.__setattr__(self, "list_repr", tuple_repr)
        object.__setattr__(self, "unlabelled_repr", _to_unlabelled_tuple(tuple_repr))

    def nodes(self) -> int:
        return _nodes(self.unlabelled_repr)

    def as_ordered_forest(self) -> OrderedForest:
        return OrderedForest((self,))

    def to_nonplanar_tree(self) -> Tree:
        if self.list_repr is None:
            return Tree(None)
        return Tree(self.list_repr)


@dataclass(frozen=True)
class NoncommutativeForest:
    """Noncommutative forest (word) of planar trees."""

    tree_list: tuple[PlanarTree, ...] = tuple()

    def __post_init__(self) -> None:
        values: tuple[PlanarTree, ...] = tuple(self.tree_list)
        if len(values) == 0:
            values = (EMPTY_PLANAR_TREE,)
        object.__setattr__(self, "tree_list", values)

    def __iter__(self) -> Iterator[PlanarTree]:
        yield from self.tree_list

    def __getitem__(self, index: int) -> PlanarTree:
        return self.tree_list[index]

    def simplify(self) -> NoncommutativeForest:
        if len(self.tree_list) <= 1:
            return self
        filtered = tuple(tree for tree in self.tree_list if tree.list_repr is not None)
        if len(filtered) == 0:
            return EMPTY_ORDERED_FOREST
        if len(filtered) == len(self.tree_list):
            return self
        return NoncommutativeForest(filtered)

    def nodes(self) -> int:
        return sum(tree.nodes() for tree in self.tree_list)

    def __mul__(
        self, other: int | float | PlanarTree | NoncommutativeForest | ForestSum
    ) -> NoncommutativeForest | ForestSum:
        if isinstance(other, (int, float)):
            return ForestSum(((sympy.sympify(other), self),))
        if isinstance(other, TreeLike):
            return NoncommutativeForest(self.tree_list + (other,)).simplify()
        if isinstance(other, NoncommutativeForest):
            return NoncommutativeForest(self.tree_list + other.tree_list).simplify()
        if isinstance(other, ForestSum):
            terms = tuple(
                (coeff, NoncommutativeForest(self.tree_list + forest.tree_list).simplify())
                for coeff, forest in other.term_list
            )
            return ForestSum(terms).simplify()
        raise TypeError(f"Cannot multiply NoncommutativeForest and {type(other)}")

    def __rmul__(
        self, other: int | float | PlanarTree | NoncommutativeForest | ForestSum
    ) -> NoncommutativeForest | ForestSum:
        if isinstance(other, (int, float)):
            return ForestSum(((sympy.sympify(other), self),))
        if isinstance(other, TreeLike):
            return NoncommutativeForest((other,) + self.tree_list).simplify()
        if isinstance(other, NoncommutativeForest):
            return NoncommutativeForest(other.tree_list + self.tree_list).simplify()
        if isinstance(other, ForestSum):
            terms = tuple(
                (coeff, NoncommutativeForest(forest.tree_list + self.tree_list).simplify())
                for coeff, forest in other.term_list
            )
            return ForestSum(terms).simplify()
        raise TypeError(f"Cannot multiply {type(other)} and NoncommutativeForest")


OrderedForest = NoncommutativeForest
OrderedForestSum = ForestSum


EMPTY_PLANAR_TREE = PlanarTree(None)
EMPTY_ORDERED_FOREST = NoncommutativeForest((EMPTY_PLANAR_TREE,))
ZERO_ORDERED_FOREST_SUM = ForestSum(((0, EMPTY_ORDERED_FOREST),))


def validate_order(order: int, *, allow_zero: bool = True) -> None:
    if allow_zero:
        if order < 0:
            raise ValueError("order must be non-negative")
        return
    if order <= 0:
        raise ValueError("order must be positive")
