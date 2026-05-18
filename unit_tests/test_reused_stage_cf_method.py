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

import sympy

import kauri


R = sympy.Rational
PT = kauri.PlanarTree


def graft(*children):
    return PT([child.list_repr for child in children])


LEAF = PT([])
C2 = graft(LEAF)
C3 = graft(C2)
C4 = graft(C3)
C5 = graft(C4)


def cfees25_method():
    return kauri.ReusedStageCFMethod(
        a=[R(-7, 15), R(-35, 32)],
        b=[R(1, 3), R(15, 16), R(2, 5)],
        name="CFEES(2,5;1/10)",
    )


def cfees27_method():
    sqrt2 = sympy.sqrt(2)
    return kauri.ReusedStageCFMethod(
        a=[
            (-7 + 4 * sqrt2) / 3,
            -(4 + 5 * sqrt2) / 12,
            3 * (-31 + 8 * sqrt2) / 49,
        ],
        b=[
            (2 - sqrt2) / 3,
            (4 + sqrt2) / 8,
            3 * (3 - sqrt2) / 7,
            (9 - 4 * sqrt2) / 14,
        ],
        name="CFEES(2,7)",
    )


def cfees25_table_values():
    x = R(1, 10)
    two_leaves = graft(LEAF, LEAF)
    three_leaves = graft(LEAF, LEAF, LEAF)
    four_leaves = graft(LEAF, LEAF, LEAF, LEAF)

    return {
        kauri.EMPTY_PLANAR_TREE: R(1),
        LEAF: R(1),
        C2: R(1, 2),
        two_leaves: (2 * x - 5) / (32 * (x - 1)),
        C3: R(1, 8),
        three_leaves: -(2 * x + 7) / (192 * (x - 1)),
        graft(LEAF, C2): -(x + 2) / (32 * (x - 1)),
        graft(C2, LEAF): R(1, 32),
        graft(two_leaves): -(2 * x + 1) / (64 * (x - 1)),
        C4: R(0),
        four_leaves: (8 * x**3 + 24 * x**2 + 36 * x - 41)
        / (6144 * (x - 1) ** 3),
        graft(LEAF, LEAF, C2): (4 * x**2 + 10 * x + 13)
        / (768 * (x - 1) ** 2),
        graft(LEAF, C2, LEAF): -(4 * x + 5) / (384 * (x - 1)),
        graft(LEAF, two_leaves): ((2 * x + 1) * (x + 2))
        / (256 * (x - 1) ** 2),
        graft(LEAF, C3): R(0),
        graft(C2, LEAF, LEAF): R(1, 192),
        graft(C2, C2): -R(1, 64) / (2 * x - 1),
        graft(two_leaves, LEAF): -(2 * x + 1) / (256 * (x - 1)),
        graft(three_leaves): (2 * x + 1) ** 2 / (768 * (x - 1) ** 2),
        graft(graft(LEAF, C2)): R(0),
        graft(C3, LEAF): R(0),
        graft(graft(C2, LEAF)): R(0),
        graft(graft(two_leaves)): R(0),
        C5: R(0),
    }


class ReusedStageCFMethodTests(unittest.TestCase):
    def test_reused_stage_class_is_exported(self):
        self.assertTrue(hasattr(kauri, "ReusedStageCFMethod"))
        self.assertIs(kauri.ReusedStageCFMethod, kauri.cf.ReusedStageCFMethod)

    def test_projected_rk_matches_cfees25_tableau(self):
        rk = cfees25_method().projected_rk()
        expected_a = [
            [R(0), R(0), R(0)],
            [R(1, 3), R(0), R(0)],
            [-R(5, 48), R(15, 16), R(0)],
        ]
        expected_b = [R(1, 10), R(1, 2), R(2, 5)]
        self.assertEqual(rk.a, expected_a)
        self.assertEqual(rk.b, expected_b)

    def test_cfees25_character_matches_published_table_at_x_one_tenth(self):
        alpha = cfees25_method().lb_character()
        for tree, expected in cfees25_table_values().items():
            self.assertEqual(
                sympy.simplify(alpha(tree) - expected),
                0,
                msg=f"tree {tree.list_repr}",
            )

    def test_cfees25_reported_orders(self):
        method = cfees25_method()
        self.assertEqual(method.planar_order(limit=3), 2)
        self.assertEqual(method.planar_antisymmetric_order(limit=6), 5)

    def test_cfees27_canonical_low_order_smoke(self):
        method = cfees27_method()
        alpha = method.lb_character()
        self.assertEqual(sympy.simplify(alpha(LEAF) - 1), 0)
        self.assertEqual(sympy.simplify(alpha(C2) - R(1, 2)), 0)

        defect = method.symmetry_defect_map()
        for order in (1, 2):
            for tree in kauri.planar_trees_of_order(order):
                self.assertEqual(
                    sympy.simplify(defect(tree)),
                    0,
                    msg=f"order {order}, tree {tree.list_repr}",
                )


if __name__ == "__main__":
    unittest.main()
