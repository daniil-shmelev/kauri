"""
Truncated ordered-tree Hopf-algebra utilities for symbolic MKW-style EES verification.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import sympy

from ..maps import Map
from ..trees import (
    EMPTY_ORDERED_FOREST,
    EMPTY_PLANAR_TREE,
    OrderedForest,
    PlanarTree,
)
from ..generic_algebra import sign_factor


def counit_impl(tree: PlanarTree) -> sympy.core.basic.Basic:
    """Counit used by the truncated verifier."""
    return sympy.Integer(1) if tree.list_repr is None else sympy.Integer(0)


def _simplify_expanded(value: sympy.core.basic.Basic | int | float) -> sympy.core.basic.Basic:
    return sympy.simplify(sympy.expand(sympy.sympify(value)))


@dataclass(frozen=True)
class CoproductTerm:
    coeff: int
    left: OrderedForest
    right: PlanarTree


def _coproduct_helper(tree: PlanarTree) -> tuple[tuple[OrderedForest, PlanarTree], ...]:
    if tree.list_repr is None:
        return ((EMPTY_ORDERED_FOREST, EMPTY_PLANAR_TREE),)
    if len(tree.list_repr) == 1:
        return ((EMPTY_ORDERED_FOREST, tree), (tree.as_ordered_forest(), EMPTY_PLANAR_TREE))

    children: list[PlanarTree] = [PlanarTree(rep) for rep in tree.list_repr[:-1]]
    child_coproducts: list[tuple[tuple[OrderedForest, PlanarTree], ...]] = [
        _coproduct_helper(child) for child in children
    ]
    out_terms: list[tuple[OrderedForest, PlanarTree]] = [
        (OrderedForest((tree,)), EMPTY_PLANAR_TREE)
    ]
    for picks in product(*child_coproducts):
        right_repr_children: list[tuple] = []
        left_trees: list[PlanarTree] = []
        for left_forest, right_tree in picks:
            if right_tree.list_repr is not None:
                right_repr_children.append(right_tree.list_repr)
            left_trees.extend(left_forest.tree_list)
        right_repr_children.append(tree.list_repr[-1])
        out_terms.append(
            (
                OrderedForest(tuple(left_trees)).simplify(),
                PlanarTree(tuple(right_repr_children)),
            )
        )
    return tuple(out_terms)


def coproduct_terms(tree: PlanarTree) -> tuple[CoproductTerm, ...]:
    """Ordered-tree BCK-style coproduct terms, preserving sibling order."""
    return tuple(
        CoproductTerm(coeff=1, left=left, right=right)
        for left, right in _coproduct_helper(tree)
    )


counit = Map(counit_impl)


def planar_convolution(f: Map, g: Map) -> Map:
    """Function product of two maps using the planar BCK-style coproduct."""

    def conv(tree: PlanarTree):
        out = 0
        for term in coproduct_terms(tree):
            left_val = f(term.left)
            right_val = g(term.right)
            out = out + term.coeff * left_val * right_val
        return out

    return Map(conv)


def verify_mkw_ees(phi: Map, order: int) -> bool:
    from ..gentrees import planar_trees_up_to_order
    from ..trees import validate_order

    validate_order(order, allow_zero=False)
    phi_sign = Map(
        lambda tree: sympy.expand(
            sympy.sympify(sign_factor(tree)) * sympy.sympify(phi(tree))
        )
    )
    residual_map = planar_convolution(phi_sign, phi)
    return all(
        _simplify_expanded(sympy.sympify(residual_map(tree)) - sympy.sympify(counit_impl(tree)))
        == 0
        for tree in planar_trees_up_to_order(order)
    )
