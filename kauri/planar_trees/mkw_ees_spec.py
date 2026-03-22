"""
Specification helpers for truncated MKW-style EES verification.
"""

from __future__ import annotations

import sympy

from kauri.planar_trees.planar_basis import PlanarTree


def counit_planar(tree: PlanarTree) -> sympy.core.basic.Basic:
    """
    Counit used by the truncated verifier.
    """
    return sympy.Integer(1) if tree.list_repr is None else sympy.Integer(0)


def sign_for_tree(tree: PlanarTree) -> int:
    """
    Tree sign for involution: (-1)^|t|.
    """
    return 1 if tree.nodes() % 2 == 0 else -1
