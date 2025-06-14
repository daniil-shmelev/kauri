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

class LabelledTests(unittest.TestCase):

    def test_init(self):
        t = Tree([[],[]])
        self.assertEqual(t.list_repr, ((0,), (0,),0) )
        t = Tree([[1],0])
        self.assertEqual(t.list_repr, ((1,),0) )

        with self.assertRaises(TypeError):
            Tree(0)
        with self.assertRaises(ValueError):
            t = Tree([[],[],'s'])
        with self.assertRaises(ValueError):
            t = Tree([0, [],[1]])
        with self.assertRaises(ValueError):
            t = Tree([0,0])

    def test_repr(self):
        self.assertEqual(repr(T([[[1],2], [1],2])), '[[[1], 2], [1], 2]')
        self.assertEqual(repr(T([[[1],2], [1],2]).as_forest()), '[[[1], 2], [1], 2]')
        self.assertEqual(repr(T([[[1], 2], [1], 2]).as_forest_sum()), ' 1 * [[[1], 2], [1], 2]')

    def test_equality(self):
        t1 = Tree([[[3], [2], 1], [2]])
        t2 = Tree([[2], [[2], [3], 1]])
        self.assertEqual(t1, t2)

    def test_colors(self):
        self.assertEqual(0, T(None).colors())
        self.assertEqual(1, T([]).colors())
        self.assertEqual(1, T([0]).colors())
        self.assertEqual(10, T([9]).colors())
        self.assertEqual(6, T([[[5],2],1]).colors())

    def test_totally_ordered(self):
        self.assertTrue(T([0]) < T([1]))
        self.assertTrue(T([[[3],1],[1]]) < T([[[6],0],[0]]))
        self.assertTrue(T([[[1], 0], [0],0]) > T([[[0], 1], [1],1]))
        self.assertTrue(T([[[1], 0], [0],0]) < T([[[1], 1], [1],1]))
