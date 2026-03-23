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

import unittest

from kauri.gentrees import planar_trees_of_order
from kauri.maps import Map
from kauri.mkw import coproduct_terms, verify_mkw_ees
from kauri.trees import EMPTY_PLANAR_TREE, OrderedForest, PlanarTree

# References:
# [F]   L. Foissy, "An introduction to Hopf algebras of trees",
#       Preprint, Universite de Reims.
#       https://www2.mathematik.hu-berlin.de/~kreimer/wp-content/uploads/Foissy.pdf
# [MKW] H. Munthe-Kaas, W. Wright,
#       "On the Hopf Algebraic Structure of Lie Group Integrators",
#       Found. Comput. Math. 8 (2008), pp. 227-257.
#       https://arxiv.org/abs/math/0603023


class MKWTests(unittest.TestCase):

    def test_coproduct_leaf(self):
        """Δ(●) = 1⊗● + ●⊗1  (2 terms)."""
        terms = coproduct_terms(PlanarTree([]))
        self.assertEqual(len(terms), 2)
        self.assertEqual(terms[0].left, EMPTY_PLANAR_TREE.as_ordered_forest())
        self.assertEqual(terms[0].right, PlanarTree([]))
        self.assertEqual(terms[1].left, PlanarTree([]).as_ordered_forest())
        self.assertEqual(terms[1].right, EMPTY_PLANAR_TREE)

    def test_coproduct_2_node(self):
        """Δ(|) has 3 terms: |⊗1, 1⊗|, ●⊗●.
        Reference: [F] §2.2.
        """
        terms = coproduct_terms(PlanarTree([[]]))
        self.assertEqual(len(terms), 3)

    def test_coproduct_fork_structure(self):
        """Δ(Y) has 5 terms with 2 distinct ●⊗| terms (one per child edge).
        Reference: [F] Examples 9, line 1.
        """
        terms = coproduct_terms(PlanarTree([[],[]]))
        self.assertEqual(len(terms), 5)

        inner = [t for t in terms
                 if t.right != EMPTY_PLANAR_TREE
                 and t.left != EMPTY_PLANAR_TREE.as_ordered_forest()]
        self.assertEqual(len(inner), 3)

        leaf = PlanarTree([])
        two_node = PlanarTree([[]])
        leaf_pair = OrderedForest((leaf, leaf))

        pairs = [(t.left, t.right) for t in inner]
        self.assertEqual(
            sum(1 for l, r in pairs if r == two_node), 2,
            "Expected two terms with right = PlanarTree([[]])"
        )
        self.assertEqual(
            sum(1 for l, r in pairs if l == leaf_pair and r == leaf), 1,
            "Expected one term with left = (●,●) and right = ●"
        )

    def test_coproduct_chain3_structure(self):
        """Δ(chain₃) has 4 terms.
        Reference: [F] Examples 9, line 2.
        """
        terms = coproduct_terms(PlanarTree([[[]]]))
        self.assertEqual(len(terms), 4)

    def test_order_4_coproduct_counts(self):
        """All 5 planar trees of order 4 have at least 2 coproduct terms.
        Reference: [F] §2.2.
        """
        trees_4 = list(planar_trees_of_order(4))
        self.assertEqual(len(trees_4), 5)
        for t in trees_4:
            terms = coproduct_terms(t)
            self.assertGreaterEqual(len(terms), 2)

    def test_catalan_counts(self):
        """dim H_PR(n) = C_n (Catalan number).
        Reference: [F] Proposition 2, OEIS A000108.
        """
        catalan = {1: 1, 2: 1, 3: 2, 4: 5, 5: 14}
        for n, c_n in catalan.items():
            self.assertEqual(len(list(planar_trees_of_order(n))), c_n,
                             msg=f"order {n}")

    def test_ees_counit(self):
        """The counit satisfies EES up to order 5.
        Reference: [MKW] §4.
        """
        counit = Map(lambda tree: 1 if tree == EMPTY_PLANAR_TREE else 0)
        self.assertTrue(verify_mkw_ees(counit, 5))

    def test_ees_constant_fails(self):
        """A constant map violates EES.
        Reference: [MKW] §4.
        """
        self.assertFalse(verify_mkw_ees(Map(lambda tree: 1), 3))
