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
from kauri import *
from kauri import Tree as T

trees = [T(None),
         T([]),
         T([[]]),
         T([[],[]]),
         T([[[]]]),
         T([[],[],[]]),
         T([[],[[]]]),
         T([[[],[]]]),
         T([[[[]]]])]

class TreeTests(unittest.TestCase):

    def test_repr(self):
        self.assertEqual(repr(T([[[]], []])), '[[[]], []]')
        self.assertEqual(repr(T([[[]], []]).as_forest()), '[[[]], []]')
        self.assertEqual(repr(T([[[]], []]).as_forest_sum()), ' 1 * [[[]], []]')
        self.assertEqual(repr(T(None)), "∅")
        self.assertEqual(repr(Forest([])), "∅")
        self.assertEqual(repr(ForestSum([])), "0")

    def test_conversion(self):
        for t in trees:
            self.assertEqual(repr(t), repr(t.as_forest()), repr(t) + " " + repr(t.as_forest()))
            self.assertEqual(" 1 * " + repr(t), repr(t.as_forest_sum()), repr(t) + " " + repr(t.as_forest_sum()))

    def test_add(self):
        t1 = T([]) + T([[],[]])
        t2 = ForestSum((
            (1, Forest([T([])])),
            (1, Forest([T([[],[]])]))
        ))
        self.assertEqual(t2, t1)

        t1 = T([]) - 2 * T([[], []])
        t2 = ForestSum((
            (1, Forest([T([])])),
            (-2, Forest([T([[], []])]))
        ))
        self.assertEqual(t2, t1)

        t1 = 1 + T([[], []])
        t2 = ForestSum((
            (1, Forest([T(None)])),
            (1, Forest([T([[], []])]))
        ))
        self.assertEqual(t2, t1)

        t1 = T([[], []]) + 2
        t2 = ForestSum((
            (1, Forest([T([[], []])])),
            (2, Forest([T(None)]))
        ))
        self.assertEqual(t2, t1)

        t1 = T([[], []]) + Forest([T([]), T([[]])])
        t2 = ForestSum((
            (1, Forest([T([[], []])])),
            (1, Forest([T([]), T([[]])]))
        ))
        self.assertEqual(t2, t1)

    def test_mul(self):
        t1 = T([]) * T([[], []])
        t2 = Forest([T([]), T([[],[]])])
        self.assertEqual(t2, t1)

        t1 = T([]) * T(None) * T(None)
        t2 = T([])
        self.assertEqual(t1, t2)

        t1 = T([[]]) * Forest([T([]), T([[],[]])])
        t2 = Forest([T([[]]), T([]), T([[],[]])])
        self.assertEqual(t1, t2)

        t1 = (T([]) - T([[]])) * T([])
        t2 = T([]) * T([]) - T([[]]) * T([])
        self.assertEqual(t1,t2)

    def test_pow_tree(self):
        t1 = T([]) ** 3
        t2 = T([]) * T([]) * T([])
        self.assertEqual(t1, t2)

        t1 = T(None) ** 3
        self.assertEqual(1, t1)

        t1 = T([]) ** 0
        self.assertEqual(1, t1)

        with self.assertRaises(ValueError):
            t1 ** -1
        with self.assertRaises(TypeError):
            t1 ** 1.5

    def test_pow_forest(self):
        t1 = Forest([T([]), T(None)]) ** 3
        t2 = T([]) * T([]) * T([])
        self.assertEqual(t1, t2)

        t1 = T(None).as_forest() ** 3
        self.assertEqual(1, t1)

        t1 = T([]).as_forest() ** 0
        self.assertEqual(1, t1)

        with self.assertRaises(ValueError):
            t1 ** -1
        with self.assertRaises(TypeError):
            t1 ** 1.5

    def test_pow_forest_sum(self):
        t1 = (T([]) - 2 * T([[]])) ** 2
        t2 = T([]) * T([]) - 4 * T([]) * T([[]]) + 4 * T([[]]) * T([[]])
        self.assertEqual(t1, t2)

        t1 = T([]).as_forest_sum() ** 0
        self.assertEqual(1, t1)

        with self.assertRaises(ValueError):
            t1 ** -1
        with self.assertRaises(TypeError):
            t1 ** 1.5

    def test_equality(self):
        self.assertEqual(T([[],[[]]]), T([[[]],[]]))
        self.assertEqual(T([[], [[]]]).as_forest(), T([[[]], []]))
        self.assertEqual(T([[],[[],[]]]), T([[[],[]], []]))
        self.assertEqual(T([[[]],[],[]]), T([[],[[]],[]]))
        self.assertEqual(T([[[]], [], []]), T([[], [], [[]]]))
        self.assertEqual(Forest([T([[]]), T([]), T([[],[]])]), Forest([T([]), T([[],[]]), T([[]])]))
        self.assertEqual(T([]) - T([[]]), -T([[]]) + T([]))
        self.assertEqual(ForestSum([(1, T([[]]).as_forest()), (1, T([]).as_forest()), (-1, T([]).as_forest())]), T([[]]).as_forest_sum())

    def test_equality_2(self):
        self.assertEqual(
            2*T([[]]) * T([]),
            2*T([]) * T([[]])
        )

        self.assertEqual(
            -T([]) * T([]) * T([]) + 2 * T([[]]) * T([]) - T([[], []]),
            - T([[], []]) + 2 * T([]) * T([[]]) -T([]) * T([]) * T([])
        )

    def test_hash(self):
        self.assertEqual(hash(T([[],[[]]])), hash(T([[[]],[]])))
        self.assertEqual(hash(T([[],[[],[]]])), hash(T([[[],[]], []])))
        self.assertEqual(hash(T([[[]],[],[]])), hash(T([[],[[]],[]])))
        self.assertEqual(hash(T([[[]], [], []])), hash(T([[], [], [[]]])))
        self.assertEqual(hash(Forest([T([[]]), T([]), T([[],[]])])), hash(Forest([T([]), T([[],[]]), T([[]])])))
        self.assertEqual(hash(T([]) - T([[]])), hash(-T([[]]) + T([])))
        self.assertEqual(hash(T([[]]) + T([]) - T([])), hash(T([[]]).as_forest_sum()))

    def test_mixed_arithmetic(self):
        t0 = T(None)
        t1 = T([])
        t2 = T([[]])
        t3 = T([[], []])

        f1 = Forest([t1, t2])
        f2 = Forest([t3])

        self.assertTrue(f1 == (t1 * t2))

        s1 = ForestSum((  (2,f1), (-1, f2)  ))  # 2*f1 - f2
        s2 = s1 + 5 * t0

        self.assertTrue(s1 == (2 * f1 - t3), repr(s1) + ", " + repr(2 * f1 - t3))
        self.assertTrue(s2 == (2 * f1 - t3 + 5), repr(s2) + ", " + repr(2 * f1 - t3 + 5))

    def test_nodes(self):
        nums = [0,1,2,3,3,4,4,4,4]
        for t, n in zip(trees, nums):
            self.assertEqual(n, t.nodes(), repr(t) + " T")
            self.assertEqual(n, t.as_forest().nodes(), repr(t) + " Forest")
            self.assertEqual(n, t.as_forest_sum().nodes(), repr(t) + " Forest")

    def test_height(self):
        nums = [0,1,2,2,3,2,3,3,4]
        for t, n in zip(trees, nums):
            self.assertEqual(n, t.height(), repr(t))

    def test_sigma(self):
        vals = [1, 1, 1, 1, 2, 1, 2, 1, 6, 1, 2, 1, 1, 6, 2, 2, 2, 24]
        i = 0
        for t in trees_up_to_order(5):
            self.assertEqual(vals[i], t.sigma(), "i = " + str(i))
            i += 1

    def test_alpha(self):
        vals = [1, 1, 2, 6, 24, 120, 720]
        for n in range(1, 8):
            alpha_sum = 0
            for t in trees_of_order(n):
                alpha_sum += t.alpha()
            self.assertEqual(vals[n-1], alpha_sum)

    def test_beta(self):
        vals = [1,2,9,64,625,7776,117649]
        for n in range(1, 8):
            beta_sum = 0
            for t in trees_of_order(n):
                beta_sum += t.beta()
            self.assertEqual(vals[n-1], beta_sum)

    def test_factorial(self):
        factorials = [1,1,2,3,6,4,8,12,24]
        for t, f in zip(trees, factorials):
            self.assertEqual(f, t.factorial(), repr(t) + " T")
            self.assertEqual(f, t.as_forest().factorial(), repr(t) + " Forest")
            self.assertEqual(f, t.as_forest_sum().factorial(), repr(t) + " ForestSum")

    def test_empty_forest(self):
        self.assertEqual(Forest([]), Forest([Tree(None)]))

    def test_totally_ordered(self):
        last_tree = Tree(None)
        for current_tree in trees_up_to_order(5):
            if current_tree == Tree(None):
                continue
            self.assertTrue(last_tree < current_tree, repr(last_tree) + " and " + repr(current_tree))
            self.assertFalse(last_tree > current_tree, repr(last_tree) + " and " + repr(current_tree))
            self.assertTrue(current_tree == current_tree, repr(current_tree))
            last_tree = current_tree