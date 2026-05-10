# Copyright 2026 Daniil Shmelev
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
"""Ordered-forest substitution for Lie--Butcher series.

The core operation implemented here is the ordered-forest coaction
``Delta_W`` from Lundervold--Munthe-Kaas: given a logarithmic linear map
``psi`` on ordered forests and a basis-aware outer character ``beta``,
``substitute(psi, beta)`` returns the substituted character
``psi star_W beta = (psi tensor beta) Delta_W``.

This is the LB-series analogue of ordinary B-series substitution used by
the reused-stage CF methods.
"""
from __future__ import annotations

from collections import Counter
from functools import lru_cache
from itertools import permutations, product

from .maps import Map
from .trees import (
    EMPTY_ORDERED_FOREST,
    ForestSum,
    OrderedForest,
    PlanarTree,
)
from .generic_algebra import mkw_apply, mkw_base_char_func
from .mkw.mkw import (
    _as_basis_aware_map,
    _basis_aware_func,
)


def _nonempty_trees(forest: OrderedForest) -> tuple[PlanarTree, ...]:
    return tuple(t for t in forest.tree_list if t.list_repr is not None)


def _forest_from_trees(trees: tuple[PlanarTree, ...]) -> OrderedForest:
    return OrderedForest(trees) if trees else EMPTY_ORDERED_FOREST


def _flatten_forest(forest: OrderedForest):
    """Return vertex records for a planar forest in preorder."""
    records = []
    root_ids = []

    def visit(tree: PlanarTree, parent, child_index):
        node_id = len(records)
        records.append(
            {
                "parent": parent,
                "child_index": child_index,
                "children": [],
            }
        )
        if parent is None:
            root_ids.append(node_id)
        else:
            records[parent]["children"].append(node_id)
        for i, child_repr in enumerate(tree.list_repr[:-1]):
            visit(PlanarTree(child_repr), node_id, i)

    for i, tree in enumerate(_nonempty_trees(forest)):
        visit(tree, None, i)
    return tuple(records), tuple(root_ids)


def _set_partitions(items: tuple[int, ...]):
    if not items:
        yield ()
        return

    first, rest = items[0], items[1:]
    for partition in _set_partitions(rest):
        yield (frozenset((first,)),) + partition
        for i, block in enumerate(partition):
            yield (
                partition[:i]
                + (frozenset((*block, first)),)
                + partition[i + 1 :]
            )


def _block_roots(block: frozenset[int], records) -> tuple[int, ...]:
    return tuple(v for v in block if records[v]["parent"] not in block)


def _sibling_list(parent, records, root_ids):
    return root_ids if parent is None else tuple(records[parent]["children"])


def _is_consecutive(values: list[int]) -> bool:
    return bool(values) and max(values) - min(values) + 1 == len(values)


def _is_admissible_block(block: frozenset[int], records, root_ids) -> bool:
    roots = _block_roots(block, records)
    parents = {records[root]["parent"] for root in roots}
    if len(parents) != 1:
        return False

    parent = next(iter(parents))
    siblings = _sibling_list(parent, records, root_ids)
    root_positions = [siblings.index(root) for root in roots]
    if not _is_consecutive(root_positions):
        return False

    for vertex in block:
        children = records[vertex]["children"]
        for index, child in enumerate(children):
            if child in block:
                if any(
                    right_child not in block
                    for right_child in children[index + 1 :]
                ):
                    return False
    return True


def _induced_forest(block: frozenset[int], records) -> OrderedForest:
    roots = sorted(_block_roots(block, records))

    def build_tree(vertex: int) -> PlanarTree:
        children = [
            build_tree(child).list_repr
            for child in records[vertex]["children"]
            if child in block
        ]
        return PlanarTree(tuple(children) + (0,))

    return _forest_from_trees(tuple(build_tree(root) for root in roots))


def _linear_extensions(items: tuple[int, ...], constraints: set[tuple[int, int]]):
    for candidate in permutations(items):
        positions = {item: i for i, item in enumerate(candidate)}
        if all(positions[left] < positions[right] for left, right in constraints):
            yield candidate


def _quotient_forests(
    partition: tuple[frozenset[int], ...],
    records,
    root_ids,
) -> Counter:
    block_of = {
        vertex: block_index
        for block_index, block in enumerate(partition)
        for vertex in block
    }
    parent_of: dict[int, int | None] = {}
    attachment_site: dict[int, int | None] = {}
    roots_by_block: dict[int, tuple[int, ...]] = {}

    for block_index, block in enumerate(partition):
        roots = _block_roots(block, records)
        roots_by_block[block_index] = roots
        parent = records[roots[0]]["parent"]
        attachment_site[block_index] = parent
        parent_of[block_index] = None if parent is None else block_of[parent]

    children_by_parent: dict[int | None, list[int]] = {None: []}
    for block_index, parent_index in parent_of.items():
        children_by_parent.setdefault(parent_index, [])
        children_by_parent.setdefault(block_index, [])
        if parent_index is None:
            children_by_parent[None].append(block_index)
        else:
            children_by_parent[parent_index].append(block_index)

    choices = []
    for parent_index, children in children_by_parent.items():
        child_tuple = tuple(children)
        constraints: set[tuple[int, int]] = set()
        for left in child_tuple:
            for right in child_tuple:
                if left == right:
                    continue
                if attachment_site[left] != attachment_site[right]:
                    continue
                site = attachment_site[left]
                siblings = _sibling_list(site, records, root_ids)
                left_pos = min(siblings.index(root) for root in roots_by_block[left])
                right_pos = min(siblings.index(root) for root in roots_by_block[right])
                if left_pos < right_pos:
                    constraints.add((left, right))
        choices.append(
            (
                parent_index,
                tuple(_linear_extensions(child_tuple, constraints)),
            )
        )

    out = Counter()
    for selected_orders in product(*(orders for _, orders in choices)):
        order_by_parent = {
            parent: order
            for (parent, _), order in zip(choices, selected_orders)
        }

        def build_tree(block_index: int) -> PlanarTree:
            children = [
                build_tree(child_index).list_repr
                for child_index in order_by_parent[block_index]
            ]
            return PlanarTree(tuple(children) + (0,))

        roots = tuple(build_tree(block_index) for block_index in order_by_parent[None])
        out[_forest_from_trees(roots)] += 1
    return out


@lru_cache(maxsize=None)
def delta_w_terms(forest: OrderedForest):
    """Return terms of the ordered-forest contraction coaction ``Delta_W``.

    Each term is ``(coeff, left_factors, right_forest)``, where
    ``left_factors`` is the symmetric product of admissible subforests.
    """
    forest = forest.simplify()
    trees = _nonempty_trees(forest)
    if not trees:
        return ((1, (), EMPTY_ORDERED_FOREST),)

    records, root_ids = _flatten_forest(forest)
    terms = []
    vertices = tuple(range(len(records)))
    for partition in _set_partitions(vertices):
        ordered_partition = tuple(sorted(partition, key=lambda block: min(block)))
        if not all(
            _is_admissible_block(block, records, root_ids)
            for block in ordered_partition
        ):
            continue
        left_factors = tuple(
            _induced_forest(block, records)
            for block in ordered_partition
        )
        for right_forest, coeff in _quotient_forests(
            ordered_partition, records, root_ids
        ).items():
            terms.append((coeff, left_factors, right_forest))
    return tuple(terms)


def substitute(logarithmic: Map, character: Map) -> Map:
    """Return the substituted character ``logarithmic star_W character``."""
    outer = _basis_aware_func(character)

    def _subst(x):
        if isinstance(x, ForestSum):
            return mkw_apply(x, _subst)
        forest = x.as_ordered_forest() if isinstance(x, PlanarTree) else x
        total = 0
        for coeff, left_factors, right_forest in delta_w_terms(forest):
            left_value = 1
            for factor in left_factors:
                left_value *= logarithmic(factor)
            total += coeff * left_value * outer(right_forest)
        return total

    return _as_basis_aware_map(_subst)


def frozen_exponential_character(weight) -> Map:
    """The pullback character of one frozen exponential ``exp(weight * F)``.

    On ordered trees this is the bullet-only character:

    - ``alpha(empty) = 1``,
    - ``alpha(bullet) = weight``,
    - ``alpha(t) = 0`` for every tree with more than one node.
    """

    return _as_basis_aware_map(
        mkw_base_char_func(
            lambda tree, coeff=weight: (
                1
                if tree.list_repr is None
                else (coeff if len(tree.list_repr) == 1 else 0)
            )
        )
    )
