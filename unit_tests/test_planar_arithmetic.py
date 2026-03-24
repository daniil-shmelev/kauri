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

from kauri.trees import (PlanarTree, NoncommutativeForest, ForestSum,
                          EMPTY_ORDERED_FOREST, EMPTY_PLANAR_TREE)


class PlanarTreeArithmeticTests(unittest.TestCase):

    def setUp(self):
        self.bullet = PlanarTree([])
        self.chain2 = PlanarTree([[]])
        self.cherry = PlanarTree([[], []])
        self.left_heavy = PlanarTree([[[]], []])
        self.right_heavy = PlanarTree([[], [[]]])

    # -- Multiplication --

    def test_scalar_mul(self):
        result = 2 * self.bullet
        self.assertIsInstance(result, ForestSum)

    def test_scalar_rmul(self):
        result = self.bullet * 3
        self.assertIsInstance(result, ForestSum)
        self.assertEqual(result, 3 * self.bullet)

    def test_tree_mul_produces_ncf(self):
        result = self.bullet * self.chain2
        self.assertIsInstance(result, NoncommutativeForest)

    def test_tree_mul_noncommutative(self):
        left = self.bullet * self.chain2
        right = self.chain2 * self.bullet
        self.assertNotEqual(left, right)

    def test_tree_mul_ncf(self):
        ncf = NoncommutativeForest((self.chain2,))
        result = self.bullet * ncf
        self.assertEqual(result, NoncommutativeForest((self.bullet, self.chain2)))

    def test_tree_rmul_ncf(self):
        ncf = NoncommutativeForest((self.chain2,))
        result = ncf * self.bullet
        self.assertEqual(result, NoncommutativeForest((self.chain2, self.bullet)))

    def test_tree_mul_forestsum(self):
        fs = self.chain2.as_forest_sum()
        result = self.bullet * fs
        self.assertIsInstance(result, ForestSum)

    # -- Power --

    def test_pow_zero(self):
        self.assertEqual(self.bullet ** 0, EMPTY_ORDERED_FOREST)

    def test_pow_one(self):
        result = self.bullet ** 1
        self.assertIsInstance(result, NoncommutativeForest)

    def test_pow_three(self):
        result = self.bullet ** 3
        expected = NoncommutativeForest((self.bullet, self.bullet, self.bullet))
        self.assertEqual(result, expected)

    # -- Addition / Subtraction --

    def test_add_trees(self):
        result = self.bullet + self.chain2
        self.assertIsInstance(result, ForestSum)

    def test_add_scalar(self):
        result = self.bullet + 5
        self.assertIsInstance(result, ForestSum)

    def test_sub(self):
        result = self.bullet - self.chain2
        expected = self.bullet + (-self.chain2)
        self.assertEqual(result, expected)

    def test_neg(self):
        result = -self.bullet
        self.assertIsInstance(result, ForestSum)
        self.assertEqual(self.bullet + result, ForestSum(()))

    # -- Sign --

    def test_sign_odd_nodes(self):
        # bullet has 1 node (odd) => sign flips
        self.assertEqual(self.bullet.sign(), -self.bullet)

    def test_sign_even_nodes(self):
        # chain2 has 2 nodes (even) => sign preserves
        self.assertEqual(self.chain2.sign(), self.chain2.as_forest_sum())

    # -- Equality --

    def test_eq_same(self):
        self.assertEqual(self.bullet, PlanarTree([]))

    def test_eq_different(self):
        self.assertNotEqual(self.bullet, self.chain2)

    def test_eq_ncf(self):
        ncf = NoncommutativeForest((self.bullet,))
        self.assertEqual(self.bullet, ncf)

    def test_eq_forestsum(self):
        fs = self.bullet.as_forest_sum()
        self.assertEqual(self.bullet, fs)

    # -- as_forest_sum / as_ordered_forest --

    def test_as_forest_sum(self):
        fs = self.bullet.as_forest_sum()
        self.assertIsInstance(fs, ForestSum)
        self.assertEqual(len(fs.term_list), 1)

    def test_as_ordered_forest(self):
        of = self.bullet.as_ordered_forest()
        self.assertIsInstance(of, NoncommutativeForest)


class NoncommutativeForestArithmeticTests(unittest.TestCase):

    def setUp(self):
        self.bullet = PlanarTree([])
        self.chain2 = PlanarTree([[]])
        self.ncf = NoncommutativeForest((self.bullet, self.chain2))

    # -- Power --

    def test_pow_zero(self):
        self.assertEqual(self.ncf ** 0, EMPTY_ORDERED_FOREST)

    def test_pow_two(self):
        result = self.ncf ** 2
        expected = NoncommutativeForest(
            (self.bullet, self.chain2, self.bullet, self.chain2))
        self.assertEqual(result, expected)

    # -- Addition / Subtraction --

    def test_add_tree(self):
        result = self.ncf + self.bullet
        self.assertIsInstance(result, ForestSum)

    def test_add_ncf(self):
        result = self.ncf + self.ncf
        self.assertIsInstance(result, ForestSum)

    def test_sub(self):
        result = self.ncf - self.ncf
        self.assertEqual(result, ForestSum(()))

    def test_neg(self):
        result = -self.ncf
        self.assertIsInstance(result, ForestSum)
        self.assertEqual(self.ncf + result, ForestSum(()))

    # -- Equality --

    def test_eq_ncf(self):
        other = NoncommutativeForest((self.bullet, self.chain2))
        self.assertEqual(self.ncf, other)

    def test_eq_different_order(self):
        other = NoncommutativeForest((self.chain2, self.bullet))
        self.assertNotEqual(self.ncf, other)

    def test_eq_tree(self):
        single = NoncommutativeForest((self.bullet,))
        self.assertEqual(single, self.bullet)

    def test_eq_forestsum(self):
        fs = self.ncf.as_forest_sum()
        self.assertEqual(self.ncf, fs)

    # -- Sign --

    def test_sign(self):
        # 3 nodes total (odd) => sign flips
        self.assertEqual(self.ncf.sign(), -self.ncf)

    # -- Join (B+) --

    def test_join(self):
        result = self.ncf.join()
        expected = PlanarTree([[], [[]]])
        self.assertEqual(result, expected)

    def test_join_roundtrip(self):
        t = PlanarTree([[], [[]]])
        children = [PlanarTree(rep) for rep in t.list_repr[:-1]]
        ncf = NoncommutativeForest(tuple(children))
        self.assertEqual(ncf.join(), t)

    # -- as_forest_sum --

    def test_as_forest_sum(self):
        fs = self.ncf.as_forest_sum()
        self.assertIsInstance(fs, ForestSum)
        self.assertEqual(fs, self.ncf)


class ForestSumInteropTests(unittest.TestCase):

    def setUp(self):
        self.bullet = PlanarTree([])
        self.chain2 = PlanarTree([[]])
        self.fs = self.bullet.as_forest_sum()

    def test_forestsum_mul_planartree(self):
        result = self.fs * self.chain2
        self.assertIsInstance(result, ForestSum)

    def test_planartree_mul_forestsum(self):
        result = self.chain2 * self.fs
        self.assertIsInstance(result, ForestSum)

    def test_forestsum_mul_preserves_order(self):
        # bullet_fs * chain2 should have bullet before chain2
        result = self.fs * self.chain2
        # chain2 * bullet_fs should have chain2 before bullet
        result2 = self.chain2 * self.fs
        self.assertNotEqual(result, result2)

    def test_forestsum_add_planartree(self):
        result = self.fs + self.chain2
        self.assertIsInstance(result, ForestSum)

    def test_forestsum_eq_planartree(self):
        self.assertEqual(self.fs, self.bullet)

    def test_forestsum_eq_ncf(self):
        ncf = NoncommutativeForest((self.bullet,))
        self.assertEqual(self.fs, ncf)

    def test_order_sensitivity(self):
        """Noncommutativity: different orderings give different results."""
        left = PlanarTree([[], [[]]]) * PlanarTree([])
        right = PlanarTree([]) * PlanarTree([[], [[]]])
        self.assertNotEqual(left, right)
