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

import unittest

from kauri.trees import PlanarTree, ForestSum
from kauri.gentrees import planar_trees_of_order
from kauri.planar_oddeven import id_sqrt, minus, plus
import kauri.pbck as pbck


class PlanarIdSqrtTests(unittest.TestCase):

    def test_id_sqrt_bullet(self):
        bullet = PlanarTree([])
        result = id_sqrt(bullet)
        expected = bullet.as_forest_sum() * 0.5
        self.assertEqual(result, expected)

    def test_id_sqrt_squared_order_1(self):
        idsq = pbck.map_product(id_sqrt, id_sqrt)
        for t in planar_trees_of_order(1):
            self.assertEqual(idsq(t), t)

    def test_id_sqrt_squared_order_2(self):
        idsq = pbck.map_product(id_sqrt, id_sqrt)
        for t in planar_trees_of_order(2):
            self.assertEqual(idsq(t), t)

    def test_id_sqrt_squared_order_3(self):
        idsq = pbck.map_product(id_sqrt, id_sqrt)
        for t in planar_trees_of_order(3):
            self.assertEqual(idsq(t), t)

    def test_id_sqrt_squared_order_4(self):
        idsq = pbck.map_product(id_sqrt, id_sqrt)
        for t in planar_trees_of_order(4):
            self.assertEqual(idsq(t), t)


class PlanarPlusMinusTests(unittest.TestCase):

    def test_plus_times_minus_order_1(self):
        pm = pbck.map_product(plus, minus)
        for t in planar_trees_of_order(1):
            self.assertEqual(pm(t), t)

    def test_plus_times_minus_order_2(self):
        pm = pbck.map_product(plus, minus)
        for t in planar_trees_of_order(2):
            self.assertEqual(pm(t), t)

    def test_plus_times_minus_order_3(self):
        pm = pbck.map_product(plus, minus)
        for t in planar_trees_of_order(3):
            self.assertEqual(pm(t), t)

    def test_plus_times_minus_order_4(self):
        pm = pbck.map_product(plus, minus)
        for t in planar_trees_of_order(4):
            self.assertEqual(pm(t), t)

    def test_minus_bullet_value(self):
        bullet = PlanarTree([])
        result = minus(bullet)
        # bullet has 1 node (odd) => minus gets it
        self.assertEqual(result, bullet.as_forest_sum())

    def test_plus_bullet_value(self):
        bullet = PlanarTree([])
        result = plus(bullet)
        # bullet has 1 node (odd) => plus gets nothing
        self.assertEqual(result, ForestSum(()))

    def test_minus_chain2_value(self):
        chain2 = PlanarTree([[]])
        result = minus(chain2)
        # chain2 has 2 nodes (even) => minus gets nothing
        self.assertEqual(result, ForestSum(()))

    def test_plus_chain2_value(self):
        chain2 = PlanarTree([[]])
        result = plus(chain2)
        # chain2 has 2 nodes (even) => plus gets it
        self.assertEqual(result, chain2.as_forest_sum())

    def test_planar_sensitivity(self):
        """Different planar orderings give different decompositions."""
        left_heavy = PlanarTree([[[]], []])
        right_heavy = PlanarTree([[], [[]]])
        # These are different planar trees (same non-planar tree)
        self.assertNotEqual(left_heavy, right_heavy)
        # Their id_sqrt values should differ
        self.assertNotEqual(id_sqrt(left_heavy), id_sqrt(right_heavy))

    def test_counit_of_plus_extended(self):
        """counit(plus(t)) == counit(t) for all trees up to order 3."""
        for order in range(1, 4):
            for t in planar_trees_of_order(order):
                plus_t = plus(t)
                # counit on ForestSum: sum of coefficients * counit(forest)
                counit_val = sum(
                    c * (1 if all(ti.list_repr is None for ti in f) else 0)
                    for c, f in plus_t.term_list
                ) if plus_t.term_list else 0
                expected = 0  # counit(t) = 0 for non-empty trees
                self.assertEqual(counit_val, expected)
