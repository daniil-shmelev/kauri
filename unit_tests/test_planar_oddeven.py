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
from kauri.gentrees import trees_of_order, planar_trees_of_order
from kauri.planar_oddeven import id_sqrt, minus, plus
import kauri.nck as nck
import kauri.oddeven as oddeven


class PlanarIdSqrtTests(unittest.TestCase):

    def setUp(self):
        self.idsq = nck.map_product(id_sqrt, id_sqrt)

    def test_id_sqrt_bullet(self):
        bullet = PlanarTree([])
        result = id_sqrt(bullet)
        expected = bullet.as_forest_sum() * 0.5
        self.assertEqual(result, expected)

    def test_id_sqrt_squared_order_1(self):
        for t in planar_trees_of_order(1):
            self.assertEqual(self.idsq(t), t)

    def test_id_sqrt_squared_order_2(self):
        for t in planar_trees_of_order(2):
            self.assertEqual(self.idsq(t), t)

    def test_id_sqrt_squared_order_3(self):
        for t in planar_trees_of_order(3):
            self.assertEqual(self.idsq(t), t)

    def test_id_sqrt_squared_order_4(self):
        for t in planar_trees_of_order(4):
            self.assertEqual(self.idsq(t), t)


class PlanarPlusMinusTests(unittest.TestCase):

    def setUp(self):
        self.pm = nck.map_product(plus, minus)

    def test_plus_times_minus_order_1(self):
        for t in planar_trees_of_order(1):
            self.assertEqual(self.pm(t), t)

    def test_plus_times_minus_order_2(self):
        for t in planar_trees_of_order(2):
            self.assertEqual(self.pm(t), t)

    def test_plus_times_minus_order_3(self):
        for t in planar_trees_of_order(3):
            self.assertEqual(self.pm(t), t)

    def test_plus_times_minus_order_4(self):
        for t in planar_trees_of_order(4):
            self.assertEqual(self.pm(t), t)

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
        # Should match non-planar: 0.5 * bullet·bullet
        bullet = PlanarTree([])
        expected = (bullet * bullet).as_forest_sum() * 0.5
        self.assertEqual(result, expected)

    def test_plus_chain2_value(self):
        chain2 = PlanarTree([[]])
        result = plus(chain2)
        # Should match non-planar: chain2 - 0.5 * bullet·bullet
        bullet = PlanarTree([])
        expected = chain2.as_forest_sum() - (bullet * bullet).as_forest_sum() * 0.5
        self.assertEqual(result, expected)

    def test_planar_sensitivity(self):
        """Different planar orderings give different decompositions."""
        left_heavy = PlanarTree([[[]], []])
        right_heavy = PlanarTree([[], [[]]])
        # These are different planar trees (same non-planar tree)
        self.assertNotEqual(left_heavy, right_heavy)
        # Their minus values should differ
        self.assertNotEqual(minus(left_heavy), minus(right_heavy))

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


# ---- Helpers for canonical comparison across Tree / PlanarTree ----

def _canonical_tree(t):
    """Type-agnostic canonical representation: sorted nested tuple of children."""
    if t.list_repr is None:
        return ()  # empty tree
    children = t.unjoin()
    # Wrap in extra tuple to distinguish leaf ((),) from empty tree ()
    return (tuple(sorted(_canonical_tree(c) for c in children.tree_list)),)


def _canonical_forest(f):
    """Canonical sorted tuple of canonical tree representations."""
    return tuple(sorted(_canonical_tree(t) for t in f.tree_list))


def _fs_to_dict(fs):
    """Convert a ForestSum to {canonical_forest: coefficient}."""
    d = {}
    for c, f in fs.term_list:
        key = _canonical_forest(f)
        d[key] = d.get(key, 0) + c
    # Drop near-zero entries
    return {k: v for k, v in d.items() if abs(v) > 1e-12}


class PlanarMatchesNonPlanarTests(unittest.TestCase):
    """Up to order 3, planar and non-planar trees are in bijection.
    The odd-even decomposition should agree on these trees."""

    def _check_order(self, order):
        np_trees = sorted(trees_of_order(order))
        p_trees = sorted(planar_trees_of_order(order))
        # Up to order 3 the counts match (1-to-1 bijection)
        self.assertEqual(len(np_trees), len(p_trees))
        for np_t, p_t in zip(np_trees, p_trees):
            # Verify the trees correspond to the same unordered tree
            self.assertEqual(
                _canonical_tree(np_t), _canonical_tree(p_t),
                f"tree mismatch at order {order}")
            # minus values should match
            p_minus = _fs_to_dict(minus(p_t))
            np_minus = _fs_to_dict(oddeven.minus(np_t))
            self.assertEqual(
                p_minus, np_minus,
                f"minus mismatch at order {order}: {np_t}")
            # plus values should match
            p_plus = _fs_to_dict(plus(p_t))
            np_plus = _fs_to_dict(oddeven.plus(np_t))
            self.assertEqual(
                p_plus, np_plus,
                f"plus mismatch at order {order}: {np_t}")

    def test_matches_order_1(self):
        self._check_order(1)

    def test_matches_order_2(self):
        self._check_order(2)

    def test_matches_order_3(self):
        self._check_order(3)
