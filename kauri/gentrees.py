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
from functools import lru_cache, cache
from itertools import product

from .trees import Tree, EMPTY_TREE
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


# ---------------------------------------------------------------------------
# Colored tree indexing
# ---------------------------------------------------------------------------

@cache
def _colored_tree_list_cached(max_order: int, d: int) -> tuple:
    """Cached tuple of all colored trees up to max_order with d colors."""
    return tuple(colored_trees_up_to_order(max_order, d))


@cache
def _colored_tree_lookup_cached(max_order: int, d: int) -> dict:
    """Cached dict mapping Tree -> index."""
    trees = _colored_tree_list_cached(max_order, d)
    return {t: i for i, t in enumerate(trees)}


@cache
def _colored_planar_tree_list_cached(max_order: int, d: int) -> tuple:
    """Cached tuple of all colored planar trees up to max_order with d colors."""
    return tuple(colored_planar_trees_up_to_order(max_order, d))


@cache
def _colored_planar_tree_lookup_cached(max_order: int, d: int) -> dict:
    """Cached dict mapping PlanarTree -> index."""
    trees = _colored_planar_tree_list_cached(max_order, d)
    return {t: i for i, t in enumerate(trees)}


def colored_trees(d: int, max_order: int) -> list[Tree]:
    """
    Returns all distinct colored rooted trees up to a given order with *d* colors,
    starting with the empty tree.

    :param d: Number of colors (path dimension).
    :type d: int
    :param max_order: Maximum number of nodes.
    :type max_order: int
    :return: List of colored trees.
    :rtype: list[Tree]
    """
    _validate_num_colors(d)
    return list(_colored_tree_list_cached(max_order, d))


def colored_tree_to_idx(tree: Tree, d: int, max_order: int) -> int:
    """
    Returns the index of a colored tree in the canonical enumeration.

    Index 0 is the empty tree. Non-empty trees are enumerated by order,
    then by shape, then by coloring.

    :param tree: A colored rooted tree.
    :type tree: Tree
    :param d: Number of colors (path dimension).
    :type d: int
    :param max_order: Maximum number of nodes.
    :type max_order: int
    :return: Index in the enumeration.
    :rtype: int
    """
    _validate_num_colors(d)
    lookup = _colored_tree_lookup_cached(max_order, d)
    if tree not in lookup:
        raise ValueError(f"Tree {tree} not found in enumeration for d={d}, max_order={max_order}")
    return lookup[tree]


def idx_to_colored_tree(idx: int, d: int, max_order: int) -> Tree:
    """
    Returns the colored tree at a given index in the canonical enumeration.

    :param idx: Index (0 = empty tree).
    :type idx: int
    :param d: Number of colors (path dimension).
    :type d: int
    :param max_order: Maximum number of nodes.
    :type max_order: int
    :return: The colored tree at the given index.
    :rtype: Tree
    """
    _validate_num_colors(d)
    trees = _colored_tree_list_cached(max_order, d)
    if idx < 0 or idx >= len(trees):
        raise ValueError(f"idx {idx} out of range [0, {len(trees)}) for d={d}, max_order={max_order}")
    return trees[idx]


def colored_planar_tree_to_idx(tree, d: int, max_order: int) -> int:
    """
    Returns the index of a colored planar tree in the canonical enumeration.

    Planar analogue of :func:`colored_tree_to_idx`. Index 0 is the empty tree;
    non-empty trees are enumerated in the order emitted by
    :func:`colored_planar_trees_up_to_order`.

    :param tree: A colored planar rooted tree.
    :type tree: PlanarTree
    :param d: Number of colors (path dimension).
    :type d: int
    :param max_order: Maximum number of nodes.
    :type max_order: int
    :return: Index in the enumeration.
    :rtype: int
    """
    _validate_num_colors(d)
    lookup = _colored_planar_tree_lookup_cached(max_order, d)
    if tree not in lookup:
        raise ValueError(f"Planar tree {tree} not found in enumeration for d={d}, max_order={max_order}")
    return lookup[tree]


def idx_to_colored_planar_tree(idx: int, d: int, max_order: int):
    """
    Returns the colored planar tree at a given index in the canonical enumeration.

    Planar analogue of :func:`idx_to_colored_tree`. Index 0 is the empty planar tree.

    :param idx: Index (0 = empty planar tree).
    :type idx: int
    :param d: Number of colors (path dimension).
    :type d: int
    :param max_order: Maximum number of nodes.
    :type max_order: int
    :return: The colored planar tree at the given index.
    :rtype: PlanarTree
    """
    _validate_num_colors(d)
    trees = _colored_planar_tree_list_cached(max_order, d)
    if idx < 0 or idx >= len(trees):
        raise ValueError(f"idx {idx} out of range [0, {len(trees)}) for d={d}, max_order={max_order}")
    return trees[idx]


# ---------------------------------------------------------------------------
# Recursive tree ordering and canonical-recursive permutation
#
# The "recursive" ordering enumerates decorated trees bottom-up, cycling root
# labels innermost. Non-planar uses child multisets; planar uses ordered child
# sequences. Both orderings match the C++ enumeration in pySigLib's
# siglib/shared/branched_trees.h.
#
# The "canonical" ordering (used by colored_trees_of_order etc.) enumerates
# by shape first, then colorings.
# ---------------------------------------------------------------------------

def _enumerate_child_indices(target_nodes, min_idx, tree_nodes, total_count, *, ordered):
    """Enumerate tuples of child tree indices whose node counts sum to target_nodes.

    When ``ordered=False`` the tuples are non-decreasing in index (multiset
    semantics); when ``ordered=True`` any left-to-right ordering is allowed.
    """
    if target_nodes == 0:
        yield ()
        return
    start = 0 if ordered else min_idx
    for idx in range(start, total_count):
        n = tree_nodes[idx]
        if n > target_nodes:
            break
        for rest in _enumerate_child_indices(
            target_nodes - n, idx, tree_nodes, total_count, ordered=ordered,
        ):
            yield (idx,) + rest


@cache
def _enumerate_trees_recursive(d: int, max_order: int, planar: bool = False) -> tuple:
    """Enumerate decorated trees in recursive ordering (child-group first, root label innermost)."""
    trees = []
    tree_nodes = []
    for order in range(1, max_order + 1):
        if order == 1:
            for label in range(d):
                trees.append((1, label, ()))
                tree_nodes.append(1)
        else:
            current_count = len(trees)
            for children in _enumerate_child_indices(
                order - 1, 0, tree_nodes, current_count, ordered=planar,
            ):
                if not children:
                    continue
                for label in range(d):
                    trees.append((order, label, children))
                    tree_nodes.append(order)
    return tuple(trees)


def _all_recursive_kauri_trees(d: int, max_order: int, planar: bool):
    """Build every recursive-order tree as a kauri Tree/PlanarTree in one bottom-up pass.

    Because each tree's ``child_ids`` reference strictly smaller indices, we can
    build the parent directly from cached children, avoiding the O(n^2) blow-up
    of the naive per-tree recursion.
    """
    from .trees import Tree, Forest, PlanarTree, NoncommutativeForest
    rec = _enumerate_trees_recursive(d, max_order, planar)
    out = [None] * len(rec)
    for i, (_n, label, child_ids) in enumerate(rec):
        if not child_ids:
            out[i] = PlanarTree([label]) if planar else Tree([label])
        elif planar:
            out[i] = NoncommutativeForest(tuple(out[c] for c in child_ids)).join(root_color=label)
        else:
            out[i] = Forest(tuple(out[c] for c in child_ids)).join(root_color=label)
    return out


def _canonical_to_recursive_permutation_impl(d: int, max_order: int, planar: bool):
    try:
        import numpy as np
    except ImportError:
        raise ImportError("Permutation functions require numpy. Install with: pip install kauri[full]")
    _validate_num_colors(d)
    rec_kauri = _all_recursive_kauri_trees(d, max_order, planar)
    rec_lookup = {t: i for i, t in enumerate(rec_kauri)}
    canonical = (_colored_planar_tree_list_cached if planar else _colored_tree_list_cached)(max_order, d)
    perm = [rec_lookup[kt] for kt in canonical[1:]]
    return np.array(perm, dtype=np.int64)


def _recursive_to_canonical_permutation_impl(d: int, max_order: int, planar: bool):
    try:
        import numpy as np
    except ImportError:
        raise ImportError("Permutation functions require numpy. Install with: pip install kauri[full]")
    perm = (planar_canonical_to_recursive_permutation if planar else canonical_to_recursive_permutation)(d, max_order)
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm))
    return inv


@cache
def canonical_to_recursive_permutation(d: int, max_order: int):
    """
    Compute the permutation mapping canonical tree indices to recursive tree indices.

    ``perm[i] = j`` means the tree at canonical position ``i`` is at recursive
    position ``j``. Both are 0-indexed and exclude the empty tree.

    :param d: Number of colors (path dimension).
    :type d: int
    :param max_order: Maximum number of nodes.
    :type max_order: int
    :return: Permutation array of shape ``(num_trees,)``.
    :rtype: numpy.ndarray
    """
    return _canonical_to_recursive_permutation_impl(d, max_order, planar=False)


@cache
def recursive_to_canonical_permutation(d: int, max_order: int):
    """
    Compute the permutation mapping recursive tree indices to canonical tree indices.

    Inverse of :func:`canonical_to_recursive_permutation`.

    :param d: Number of colors (path dimension).
    :type d: int
    :param max_order: Maximum number of nodes.
    :type max_order: int
    :return: Inverse permutation array of shape ``(num_trees,)``.
    :rtype: numpy.ndarray
    """
    return _recursive_to_canonical_permutation_impl(d, max_order, planar=False)


@cache
def planar_canonical_to_recursive_permutation(d: int, max_order: int):
    """
    Compute the permutation mapping canonical planar-tree indices to recursive planar-tree indices.

    ``perm[i] = j`` means the planar tree at canonical position ``i`` is at recursive
    position ``j``. Both are 0-indexed and exclude the empty tree. The canonical
    ordering is the one emitted by :func:`colored_planar_trees_up_to_order`; the
    recursive ordering matches the bottom-up planar enumeration used internally
    by pySigLib.

    :param d: Number of colors (path dimension).
    :type d: int
    :param max_order: Maximum number of nodes.
    :type max_order: int
    :return: Permutation array of shape ``(num_trees,)``.
    :rtype: numpy.ndarray
    """
    return _canonical_to_recursive_permutation_impl(d, max_order, planar=True)


@cache
def planar_recursive_to_canonical_permutation(d: int, max_order: int):
    """
    Compute the permutation mapping recursive planar-tree indices to canonical planar-tree indices.

    Inverse of :func:`planar_canonical_to_recursive_permutation`.

    :param d: Number of colors (path dimension).
    :type d: int
    :param max_order: Maximum number of nodes.
    :type max_order: int
    :return: Inverse permutation array of shape ``(num_trees,)``.
    :rtype: numpy.ndarray
    """
    return _recursive_to_canonical_permutation_impl(d, max_order, planar=True)
