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

trees = [T([]),
         T([[]]),
         T([[],[]]),
         T([[[]]]),
         T([[],[],[]]),
         T([[],[[]]]),
         T([[[],[]]]),
         T([[[[]]]])]

id_sqrt_vals = [
    (1/2) * T([]),
    (1/2) * T([[]]) - (1/8) * T([]),
    (1/2) * T([[],[]]) - (1/4) * T([[]]),
    (1/2) * T([[[]]]) - (1/4) * T([[]]) + (1/16) * T([]),
    (1/2) * T([[],[],[]]) - (3/8) * T([[],[]]) + (1/64) * T([]),
    (1/2) * T([[],[[]]]) - (1/8) * T([[]]) * T([[]]) - (1/8) * T([[[]]]) - (1/8) * T([[],[]]) + (1/16) * T([[]]) + (1/128) * T([]),
    (1/2) * T([[[],[]]]) - (1/4) * T([[[]]]) - (1/8) * T([[],[]]) + (1/8) * T([[]]) - (1/64) * T([]),
    (1/2) * T([[[[]]]]) - (1/8) * T([[]]) * T([[]]) - (1/4) * T([[[]]]) + (3/16) * T([[]]) - (5/128) * T([])
]

minus_vals = [
    T([]),
    (1/2) * T([]),
    T([[],[]]),
    T([[[]]]) - T([[]]) + (1/2) * T([]),
    (3/2) * T([[],[]]) - (1/4) * T([]),
    (1/2) * T([[[]]]) + (1/2) * T([[],[]]) - (1/2) * T([[]]) + (1/8) * T([]),
    T([[[]]]) + (1/2) * T([[],[]]) - T([[]]) + (1/4) * T([]),
    T([[[]]]) - T([[]]) + (3/8) * T([])
]

plus_vals = [
    0,
    T([[]]) - (1/2) * T([]),
    0,
    0,
    T([[],[],[]]) - (3/2) * T([[],[]]) + (1/4) * T([]),
    T([[],[[]]]) - (1/2) * T([[[]]]) - (1/2) * T([[],[]]) + (1/8) * T([]),
    T([[[],[]]]) - T([[[]]]) - (1/2) * T([[],[]]) + T([[]]) - (1/4) * T([]),
    T([[[[]]]]) - T([[[]]]) + (1/2) * T([[]]) - (1/8) * T([])
]

class BCKTests(unittest.TestCase):

    def test_id_sqrt(self):
        for t, s in zip(trees, id_sqrt_vals):
            self.assertEqual(id_sqrt(t).singleton_reduced(), s)

    def test_minus(self):
        for t, s in zip(trees, minus_vals):
            self.assertEqual(minus(t).singleton_reduced(), s)

    def test_plus(self):
        for t, s in zip(trees, plus_vals):
            self.assertEqual(plus(t).singleton_reduced(), s)

    def test_id_sqrt_2(self):
        m = id_sqrt ** 2
        for t in trees_up_to_order(5):
            self.assertEqual(t, m(t))

    def test_plus_minus(self):
        m = plus * minus
        for t in trees_up_to_order(5):
            self.assertEqual(t, m(t))

    def test_exact_weights(self):
        for t in trees_up_to_order(5):
            self.assertAlmostEqual(exact_weights(t), exact_weights(minus(t)))
            self.assertAlmostEqual(bck.counit(t), exact_weights(plus(t)))