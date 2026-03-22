"""Ordered-tree basis objects for truncated verification."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import sympy

from kauri.trees import Tree
from kauri.utils import _check_valid, _nodes, _to_labelled_tuple, _to_unlabelled_tuple


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
class OrderedForest:
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

    def simplify(self) -> OrderedForest:
        if len(self.tree_list) <= 1:
            return self
        filtered = tuple(tree for tree in self.tree_list if tree.list_repr is not None)
        if len(filtered) == 0:
            return EMPTY_ORDERED_FOREST
        if len(filtered) == len(self.tree_list):
            return self
        return OrderedForest(filtered)

    def nodes(self) -> int:
        return sum(tree.nodes() for tree in self.tree_list)

    def __mul__(
        self, other: int | float | PlanarTree | OrderedForest | OrderedForestSum
    ) -> OrderedForest | OrderedForestSum:
        if isinstance(other, (int, float)):
            return OrderedForestSum(((sympy.sympify(other), self),))
        if isinstance(other, PlanarTree):
            return OrderedForest(self.tree_list + (other,)).simplify()
        if isinstance(other, OrderedForest):
            return OrderedForest(self.tree_list + other.tree_list).simplify()
        if isinstance(other, OrderedForestSum):
            terms = tuple(
                (coeff, OrderedForest(self.tree_list + forest.tree_list).simplify())
                for coeff, forest in other.term_list
            )
            return OrderedForestSum(terms).simplify()
        raise TypeError(f"Cannot multiply OrderedForest and {type(other)}")

    __rmul__ = __mul__


@dataclass(frozen=True)
class OrderedForestSum:
    """Linear combination of ordered forests."""

    term_list: tuple[tuple[sympy.core.basic.Basic, OrderedForest], ...] = tuple()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "term_list",
            tuple((sympy.sympify(c), f) for c, f in self.term_list),
        )

    def __iter__(self) -> Iterator[tuple[sympy.core.basic.Basic, OrderedForest]]:
        yield from self.term_list

    def simplify(self) -> OrderedForestSum:
        if len(self.term_list) == 0:
            return ZERO_ORDERED_FOREST_SUM
        merged: dict[tuple[PlanarTree, ...], sympy.core.basic.Basic] = {}
        forest_by_key: dict[tuple[PlanarTree, ...], OrderedForest] = {}
        for coeff, forest in self.term_list:
            key = forest.simplify().tree_list
            merged[key] = sympy.simplify(merged.get(key, sympy.Integer(0)) + sympy.sympify(coeff))
            forest_by_key[key] = forest.simplify()
        terms: list[tuple[sympy.core.basic.Basic, OrderedForest]] = []
        for key, coeff in merged.items():
            if sympy.simplify(coeff) != 0:
                terms.append((sympy.simplify(coeff), forest_by_key[key]))
        if len(terms) == 0:
            return ZERO_ORDERED_FOREST_SUM
        return OrderedForestSum(tuple(terms))


@dataclass(frozen=True)
class TensorOrderedSum:
    """Linear combination of tensor products of ordered forests."""

    term_list: tuple[tuple[sympy.core.basic.Basic, OrderedForest, OrderedForest], ...]

    def __iter__(self) -> Iterator[tuple]:
        yield from self.term_list

    def __len__(self) -> int:
        return len(self.term_list)


EMPTY_PLANAR_TREE = PlanarTree(None)
EMPTY_ORDERED_FOREST = OrderedForest((EMPTY_PLANAR_TREE,))
EMPTY_ORDERED_FOREST_SUM = OrderedForestSum(((sympy.Integer(1), EMPTY_ORDERED_FOREST),))
ZERO_ORDERED_FOREST_SUM = OrderedForestSum(((sympy.Integer(0), EMPTY_ORDERED_FOREST),))


def validate_order(order: int, *, allow_zero: bool = True) -> None:
    if allow_zero:
        if order < 0:
            raise ValueError("order must be non-negative")
        return
    if order <= 0:
        raise ValueError("order must be positive")
