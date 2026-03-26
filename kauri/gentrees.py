# Copyright 2025 Daniil Shmelev
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

"""
Functions for generating rooted trees in lexicographic order, based on the algorithms of :cite:`beyer1980constant`.
"""
from typing import Generator
from functools import lru_cache
from itertools import product

from .trees import Tree
from .utils import _level_sequence_to_list_repr, _apply_color_sequence

def trees_up_to_order(order : int) -> Generator[Tree, None, None]:
    """
    Yields the trees up to and including order :math:`n`, ordered by the lexicographic order.

    :param order: Maximum order
    :type order: int
    :yields: The next tree in lexicographic order, as long as the
        order of the tree does not exceed :math:`n`.
    :rtype: Tree

    **Example usage:**

    .. kauri-exec::

        trees = list(trees_up_to_order(5))
        for i in range(0, len(trees), 10):
            kr.display(*trees[i:i+10])
    """
    if not isinstance(order, int):
        raise TypeError("order must be an int, not " + str(type(order)))
    if order < 0:
        raise ValueError("order must be non-negative")

    t = Tree(None)
    while t.nodes() <= order:
        yield t
        t = next(t)

def trees_of_order(order : int) -> Generator[Tree, None, None]:
    """
    Yields the trees of order :math:`n`, ordered by the lexicographic order.

    :param order: Order
    :type order: int
    :yields: The next tree in lexicographic order, as long as the order of the tree is :math:`n`.
    :rtype: Tree

    **Example usage:**

    .. kauri-exec::

        kr.display(*trees_of_order(5))
    """
    if not isinstance(order, int):
        raise TypeError("order must be an int, not " + str(type(order)))
    if order < 0:
        raise ValueError("order must be non-negative")

    t = Tree(_level_sequence_to_list_repr(list(range(order))))
    while t.nodes() == order:
        yield t
        t = next(t)

def _ordered_compositions(total: int) -> Generator[tuple[int, ...], None, None]:
    """Yields ordered tuples of positive integers summing to total."""
    if total == 0:
        yield tuple()
        return
    for first in range(1, total + 1):
        for rest in _ordered_compositions(total - first):
            yield (first, *rest)


@lru_cache(maxsize=None)
def _planar_repr_of_order(order: int) -> tuple[tuple, ...]:
    """Canonical tuple representations for planar trees of fixed order."""
    if order == 1:
        return (tuple(),)

    out: list[tuple] = []
    for child_orders in _ordered_compositions(order - 1):
        per_child = tuple(_planar_repr_of_order(child_order) for child_order in child_orders)
        for child_tuple in product(*per_child):
            out.append(tuple(child_tuple))

    return tuple(sorted(out))


def planar_trees_of_order(order: int):
    """
    Yields planar rooted trees of fixed order.

    Order 0 contains only the empty planar tree.

    **Example usage:**

    .. kauri-exec::

        trees = list(planar_trees_of_order(5))
        for i in range(0, len(trees), 10):
            kr.display(*trees[i:i+10])
    """
    from .trees import EMPTY_PLANAR_TREE, PlanarTree, validate_order

    validate_order(order)
    if order == 0:
        yield EMPTY_PLANAR_TREE
        return

    for list_repr in _planar_repr_of_order(order):
        yield PlanarTree(list_repr)


def planar_trees_up_to_order(order: int):
    """Yields planar rooted trees of all orders from 0 through order.

    **Example usage:**

    .. kauri-exec::

        trees = list(planar_trees_up_to_order(5))
        for i in range(0, len(trees), 10):
            kr.display(*trees[i:i+10])
    """
    from .trees import validate_order

    validate_order(order)
    for current_order in range(order + 1):
        yield from planar_trees_of_order(current_order)


def _validate_num_colors(d):
    if not isinstance(d, int):
        raise TypeError("number of colors d must be an int, not " + str(type(d)))
    if d < 0:
        raise ValueError("number of colors d must be non-negative")


def _all_colorings(unlabelled, d: int, n: int, cls=Tree):
    """Yields a tree (of type *cls*) for every coloring of an unlabelled shape."""
    for coloring in product(range(d), repeat=n):
        yield cls(_apply_color_sequence(unlabelled, iter(coloring)))


def _color_all_variants(shape: Tree, d: int):
    """Yields all distinct colorings of an unlabelled tree shape with d colors."""
    n = shape.nodes()
    if n == 0:
        yield shape
        return
    unlabelled = shape.unlabelled_repr
    if shape.sigma() == 1:
        yield from _all_colorings(unlabelled, d, n)
    else:
        seen = set()
        for t in _all_colorings(unlabelled, d, n):
            if t not in seen:
                seen.add(t)
                yield t


def colored_trees_of_order(order: int, d: int):
    """
    Yields all distinct colored rooted trees of a given order with *d* colors.

    Each node is decorated with a color from {0, ..., d-1}.

    :param order: Number of nodes
    :type order: int
    :param d: Number of colors
    :type d: int
    :yields: Colored trees
    :rtype: Tree

    **Example usage:**

    .. kauri-exec::

        trees = list(colored_trees_of_order(4, 2))
        for i in range(0, len(trees), 10):
            kr.display(*trees[i:i+10])
    """
    _validate_num_colors(d)
    for shape in trees_of_order(order):
        yield from _color_all_variants(shape, d)


def colored_trees_up_to_order(order: int, d: int):
    """
    Yields all distinct colored rooted trees up to and including a given order with *d* colors.

    :param order: Maximum number of nodes
    :type order: int
    :param d: Number of colors
    :type d: int
    :yields: Colored trees
    :rtype: Tree

    **Example usage:**

    .. kauri-exec::

        trees = list(colored_trees_up_to_order(4, 2))
        for i in range(0, len(trees), 10):
            kr.display(*trees[i:i+10])
    """
    _validate_num_colors(d)
    for shape in trees_up_to_order(order):
        yield from _color_all_variants(shape, d)


def colored_planar_trees_of_order(order: int, d: int):
    """
    Yields all colored planar rooted trees of a given order with *d* colors.

    Each node is decorated with a color from {0, ..., d-1}.
    Planar trees have no symmetry, so every coloring is distinct.

    :param order: Number of nodes
    :type order: int
    :param d: Number of colors
    :type d: int
    :yields: Colored planar trees
    :rtype: PlanarTree

    **Example usage:**

    .. kauri-exec::

        trees = list(colored_planar_trees_of_order(4, 2))
        for i in range(0, len(trees), 10):
            kr.display(*trees[i:i+10])
    """
    from .trees import PlanarTree, EMPTY_PLANAR_TREE, validate_order

    validate_order(order)
    _validate_num_colors(d)
    if order == 0:
        yield EMPTY_PLANAR_TREE
        return
    for shape in planar_trees_of_order(order):
        yield from _all_colorings(shape.unlabelled_repr, d, order, PlanarTree)


def colored_planar_trees_up_to_order(order: int, d: int):
    """
    Yields all colored planar rooted trees up to and including a given order with *d* colors.

    :param order: Maximum number of nodes
    :type order: int
    :param d: Number of colors
    :type d: int
    :yields: Colored planar trees
    :rtype: PlanarTree

    **Example usage:**

    .. kauri-exec::

        trees = list(colored_planar_trees_up_to_order(4, 2))
        for i in range(0, len(trees), 10):
            kr.display(*trees[i:i+10])
    """
    from .trees import validate_order

    validate_order(order)
    _validate_num_colors(d)
    for current_order in range(order + 1):
        yield from colored_planar_trees_of_order(current_order, d)
