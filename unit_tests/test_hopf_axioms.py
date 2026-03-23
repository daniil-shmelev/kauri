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
Tests for universal Hopf algebra identities across BCK, CEM, and GL.

References:
    - S. Montgomery, "Hopf Algebras and Their Actions on Rings", CBMS 82, AMS 1993.
    - C. Kassel, "Quantum Groups", Springer GTM 155, 1995.
    - D. Grinberg, V. Reiner, "Hopf Algebras in Combinatorics", arXiv:1409.8356.

Identities tested:
    1. ε ∘ S = ε
    2. S² = id  (for commutative or cocommutative Hopf algebras)
    3. m(S ⊗ id)Δ = η ∘ ε  (left antipode property)
    4. m(id ⊗ S)Δ = η ∘ ε  (right antipode property)
    5. f * ε = f = ε * f  (counit is convolution identity)
    6. (f * g) * h = f * (g * h)  (convolution associativity, from coassociativity)
    7. (Δ ⊗ id) ∘ Δ = (id ⊗ Δ) ∘ Δ  (coassociativity, direct check)
"""

import unittest
from collections import defaultdict
from kauri import Tree, Map, ident
import kauri.bck as bck
import kauri.cem as cem
import kauri.gl as gl

T = Tree

# BCK trees (empty tree is the unit)
bck_trees = [T(None), T([]), T([[]]), T([[],[]]), T([[[]]]),
             T([[],[],[]]), T([[],[[]]])]

# CEM and GL trees (single vertex is the unit)
cem_gl_trees = [T([]), T([[]]), T([[],[]]), T([[[]]]),
                T([[],[],[]]), T([[],[[]]])]


class TestHopfAxioms(unittest.TestCase):

    # ==================================================================
    # Identity 1: ε ∘ S = ε
    # ==================================================================

    def test_bck_counit_of_antipode(self):
        for t in bck_trees:
            self.assertEqual(bck.counit(t), bck.counit(bck.antipode(t)),
                             msg=f"ε(S({t})) ≠ ε({t}) in BCK")

    def test_cem_counit_of_antipode(self):
        for t in cem_gl_trees:
            self.assertEqual(cem.counit(t), cem.counit(cem.antipode(t)),
                             msg=f"ε(S({t})) ≠ ε({t}) in CEM")

    def test_gl_counit_of_antipode(self):
        for t in cem_gl_trees:
            self.assertEqual(gl.counit(t), gl.counit(gl.antipode(t)),
                             msg=f"ε(S({t})) ≠ ε({t}) in GL")

    # ==================================================================
    # Identity 2: S² = id
    # BCK is commutative, CEM is commutative, GL is cocommutative.
    # ==================================================================

    def test_bck_antipode_involution(self):
        for t in bck_trees:
            self.assertEqual(t, bck.antipode(bck.antipode(t)),
                             msg=f"S²({t}) ≠ {t} in BCK")

    def test_cem_antipode_involution(self):
        for t in cem_gl_trees:
            self.assertEqual(t, cem.antipode(cem.antipode(t)),
                             msg=f"S²({t}) ≠ {t} in CEM")

    def test_gl_antipode_involution(self):
        for t in cem_gl_trees:
            self.assertEqual(t, gl.antipode(gl.antipode(t)),
                             msg=f"S²({t}) ≠ {t} in GL")

    # ==================================================================
    # Identity 3: m(S ⊗ id)Δ = η ∘ ε  (left antipode property)
    # ==================================================================

    def test_bck_left_antipode(self):
        m = bck.map_product(bck.antipode, ident)
        for t in bck_trees:
            self.assertEqual(bck.counit(t), m(t),
                             msg=f"m(S⊗id)Δ({t}) ≠ ε({t}) in BCK")

    def test_cem_left_antipode(self):
        m = cem.map_product(cem.antipode, ident)
        for t in cem_gl_trees:
            expected = T([]) if cem.counit(t) == 1 else 0
            self.assertEqual(expected, m(t),
                             msg=f"m(S⊗id)Δ({t}) ≠ η(ε({t})) in CEM")

    def test_gl_left_antipode(self):
        for t in cem_gl_trees:
            cp = gl.coproduct(t)
            result = 0
            for c, lf, rf in cp:
                left = lf.tree_list[0]
                right = rf.tree_list[0]
                s_left = gl.antipode(left)
                result = result + c * gl.product(s_left, right)
            expected = T([]) if gl.counit(t) == 1 else 0
            self.assertEqual(expected, result,
                             msg=f"m(S⊗id)Δ({t}) ≠ η(ε({t})) in GL")

    # ==================================================================
    # Identity 4: m(id ⊗ S)Δ = η ∘ ε  (right antipode property)
    # ==================================================================

    def test_bck_right_antipode(self):
        m = bck.map_product(ident, bck.antipode)
        for t in bck_trees:
            self.assertEqual(bck.counit(t), m(t),
                             msg=f"m(id⊗S)Δ({t}) ≠ ε({t}) in BCK")

    def test_cem_right_antipode(self):
        m = cem.map_product(ident, cem.antipode)
        for t in cem_gl_trees:
            expected = T([]) if cem.counit(t) == 1 else 0
            self.assertEqual(expected, m(t),
                             msg=f"m(id⊗S)Δ({t}) ≠ η(ε({t})) in CEM")

    def test_gl_right_antipode(self):
        for t in cem_gl_trees:
            cp = gl.coproduct(t)
            result = 0
            for c, lf, rf in cp:
                left = lf.tree_list[0]
                right = rf.tree_list[0]
                s_right = gl.antipode(right)
                # Extend GL product linearly on right side
                for d, f in s_right.term_list:
                    tree_r = f.tree_list[0]
                    result = result + c * d * gl.product(left, tree_r)
            expected = T([]) if gl.counit(t) == 1 else 0
            self.assertEqual(expected, result,
                             msg=f"m(id⊗S)Δ({t}) ≠ η(ε({t})) in GL")

    # ==================================================================
    # Identity 5: f * ε = f = ε * f  (counit is convolution identity)
    # ==================================================================

    def test_bck_counit_conv_identity(self):
        f = Map(lambda t: t.nodes() + 1 if t.list_repr is not None else 1)
        f_eps = bck.map_product(f, bck.counit)
        eps_f = bck.map_product(bck.counit, f)
        for t in bck_trees:
            self.assertEqual(f(t), f_eps(t), msg=f"(f*ε)({t}) ≠ f({t}) in BCK")
            self.assertEqual(f(t), eps_f(t), msg=f"(ε*f)({t}) ≠ f({t}) in BCK")

    def test_cem_counit_conv_identity(self):
        f = Map(lambda t: t.nodes())
        f_eps = cem.map_product(f, cem.counit)
        eps_f = cem.map_product(cem.counit, f)
        for t in cem_gl_trees:
            self.assertEqual(f(t), f_eps(t), msg=f"(f*ε)({t}) ≠ f({t}) in CEM")
            self.assertEqual(f(t), eps_f(t), msg=f"(ε*f)({t}) ≠ f({t}) in CEM")

    def test_gl_counit_conv_identity(self):
        f = Map(lambda t: t.nodes())
        f_eps = gl.map_product(f, gl.counit)
        eps_f = gl.map_product(gl.counit, f)
        for t in cem_gl_trees:
            self.assertEqual(f(t), f_eps(t), msg=f"(f*ε)({t}) ≠ f({t}) in GL")
            self.assertEqual(f(t), eps_f(t), msg=f"(ε*f)({t}) ≠ f({t}) in GL")

    # ==================================================================
    # Identity 6: (f * g) * h = f * (g * h)  (convolution associativity)
    # Consequence of coassociativity of Δ.
    # ==================================================================

    def test_bck_conv_associativity(self):
        f = Map(lambda t: t.nodes() + 1 if t.list_repr is not None else 1)
        g = Map(lambda t: (-1)**t.nodes() if t.list_repr is not None else 1)
        h = Map(lambda t: t.nodes()**2 + 1 if t.list_repr is not None else 1)
        fg_h = bck.map_product(bck.map_product(f, g), h)
        f_gh = bck.map_product(f, bck.map_product(g, h))
        for t in bck_trees:
            self.assertEqual(fg_h(t), f_gh(t),
                             msg=f"(f*g)*h ≠ f*(g*h) at {t} in BCK")

    def test_cem_conv_associativity(self):
        f = Map(lambda t: t.nodes())
        g = Map(lambda t: (-1)**(t.nodes() - 1))
        h = Map(lambda t: t.nodes()**2)
        fg_h = cem.map_product(cem.map_product(f, g), h)
        f_gh = cem.map_product(f, cem.map_product(g, h))
        for t in cem_gl_trees:
            self.assertEqual(fg_h(t), f_gh(t),
                             msg=f"(f*g)*h ≠ f*(g*h) at {t} in CEM")

    def test_gl_conv_associativity(self):
        f = Map(lambda t: t.nodes())
        g = Map(lambda t: (-1)**(t.nodes() - 1))
        h = Map(lambda t: t.nodes()**2)
        fg_h = gl.map_product(gl.map_product(f, g), h)
        f_gh = gl.map_product(f, gl.map_product(g, h))
        for t in cem_gl_trees:
            self.assertEqual(fg_h(t), f_gh(t),
                             msg=f"(f*g)*h ≠ f*(g*h) at {t} in GL")

    # ==================================================================
    # Identity 7: (Δ ⊗ id) ∘ Δ = (id ⊗ Δ) ∘ Δ  (coassociativity)
    # Direct check on the GL coproduct (both tensor factors are trees).
    # ==================================================================

    def test_gl_coassociativity(self):
        for t in cem_gl_trees:
            cp = gl.coproduct(t)

            # (id ⊗ Δ) ∘ Δ(t)
            id_delta = defaultdict(int)
            for c1, lf, rf in cp:
                left = lf.tree_list[0]
                right = rf.tree_list[0]
                for c2, lf2, rf2 in gl.coproduct(right):
                    mid = lf2.tree_list[0]
                    right2 = rf2.tree_list[0]
                    key = (left.sorted_list_repr(),
                           mid.sorted_list_repr(),
                           right2.sorted_list_repr())
                    id_delta[key] += c1 * c2

            # (Δ ⊗ id) ∘ Δ(t)
            delta_id = defaultdict(int)
            for c1, lf, rf in cp:
                left = lf.tree_list[0]
                right = rf.tree_list[0]
                for c2, lf2, rf2 in gl.coproduct(left):
                    left2 = lf2.tree_list[0]
                    mid = rf2.tree_list[0]
                    key = (left2.sorted_list_repr(),
                           mid.sorted_list_repr(),
                           right.sorted_list_repr())
                    delta_id[key] += c1 * c2

            id_delta = {k: v for k, v in id_delta.items() if v != 0}
            delta_id = {k: v for k, v in delta_id.items() if v != 0}
            self.assertEqual(id_delta, delta_id,
                             msg=f"Coassociativity failed for {t} in GL")
