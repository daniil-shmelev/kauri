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
from kauri import TensorProductSum
from kauri import Tree as T
from kauri.trees import PlanarTree, NoncommutativeForest, OrderedForest, EMPTY_PLANAR_TREE, EMPTY_ORDERED_FOREST

trees = [T(None),
         T([]),
         T([[]]),
         T([[],[]]),
         T([[[]]]),
         T([[],[],[]]),
         T([[],[[]]]),
         T([[[],[]]]),
         T([[[[]]]])]

class TensorProductSumTests(unittest.TestCase):
    def test_tensor(self):
        t1 = T([]) @ T([[]]) + T([]) * T([]) @ T([[],[]])
        t2 = TensorProductSum([(1, T([]), T([[]])), (1, T([]) * T([]), T([[],[]]))])
        self.assertEqual(t1, t2)

    def test_planar_tensor(self):
        """TensorProductSum accepts PlanarTree and NoncommutativeForest."""
        PT = PlanarTree
        tp = TensorProductSum([
            (1, EMPTY_ORDERED_FOREST, PT([])),
            (1, PT([]).as_ordered_forest(), EMPTY_PLANAR_TREE),
        ])
        self.assertEqual(len(tp.term_list), 2)
        # Stored as forests, not bare trees
        for c, f1, f2 in tp.term_list:
            self.assertIsInstance(f1, NoncommutativeForest)
            self.assertIsInstance(f2, NoncommutativeForest)

    def test_planar_tensor_simplify(self):
        """Simplify merges equal planar tensor terms."""
        PT = PlanarTree
        tp = TensorProductSum([
            (1, PT([]).as_ordered_forest(), PT([[]]).as_ordered_forest()),
            (2, PT([]).as_ordered_forest(), PT([[]]).as_ordered_forest()),
        ])
        simplified = tp.simplify()
        self.assertEqual(len(simplified.term_list), 1)
        self.assertEqual(simplified.term_list[0][0], 3)

    def test_planar_tensor_order_sensitive(self):
        """Planar tensor products preserve sibling order."""
        PT = PlanarTree
        f1 = OrderedForest((PT([]), PT([[]])))
        f2 = OrderedForest((PT([[]]), PT([])))
        tp1 = TensorProductSum([(1, f1, EMPTY_ORDERED_FOREST)])
        tp2 = TensorProductSum([(1, f2, EMPTY_ORDERED_FOREST)])
        self.assertNotEqual(tp1, tp2)