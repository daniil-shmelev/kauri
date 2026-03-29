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

from kauri.gentrees import planar_trees_of_order, planar_trees_up_to_order
from kauri.trees import OrderedForest, PlanarTree


class PlanarTreeTests(unittest.TestCase):
    def test_ordering_distinguishes_planar_trees(self):
        left_heavy = PlanarTree([[[]], []])
        right_heavy = PlanarTree([[], [[]]])

        self.assertNotEqual(left_heavy, right_heavy)
        self.assertEqual(left_heavy.to_nonplanar_tree(), right_heavy.to_nonplanar_tree())

    def test_planar_tree_counts(self):
        expected_counts = {0: 1, 1: 1, 2: 1, 3: 2, 4: 5}
        for order, count in expected_counts.items():
            self.assertEqual(len(list(planar_trees_of_order(order))), count)

        self.assertEqual(len(list(planar_trees_up_to_order(4))), sum(expected_counts.values()))

    def test_ordered_forest_rmul(self):
        t1 = PlanarTree([])
        t2 = PlanarTree([[]])
        forest = OrderedForest((t2,))

        # t1 * forest should prepend: (t1, t2)
        left = t1 * forest
        # forest * t1 should append: (t2, t1)
        right = forest * t1

        self.assertNotEqual(left, right)
        self.assertEqual(left, OrderedForest((t1, t2)))
        self.assertEqual(right, OrderedForest((t2, t1)))
