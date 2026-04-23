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
"""Reference tests for :func:`kauri.mkw.mkw.forest_coproduct_impl` against
Table 5 of Munthe-Kaas & Wright (2008), "On the Hopf Algebraic Structure
of Lie Group Integrators", arxiv math/0603023.

TeX source: ``testing_scratch/munthe-kaas06oth.tex`` lines 1583-1648.

Tree macro decoding (``\\a...b`` is bracket notation — one ``a`` opens a
node, matching ``b`` closes it; content between the outer pair is the
children as a concatenation of sub-macros):

    \\one      = empty unit
    \\ab       = bullet         = PlanarTree([])
    \\aabb     = chain2         = PlanarTree([[]])
    \\aaabbb   = chain3         = PlanarTree([[[]]])
    \\aababb   = cherry         = PlanarTree([[], []])
    \\aaaabbbb = chain4         = PlanarTree([[[[]]]])
    \\aaababbb = B+(cherry)     = PlanarTree([[[], []]])
    \\aabaabbb = B+(b, chain2)  = PlanarTree([[], [[]]])     (t43)
    \\aaabbabb = B+(chain2, b)  = PlanarTree([[[]], []])     (t44)
    \\aabababb = corolla3       = PlanarTree([[], [], []])

Forest macros are concatenations of tree macros.
"""
import unittest

from kauri.trees import PlanarTree, OrderedForest
from kauri.mkw.mkw import forest_coproduct_impl


# --- Tree shorthands ----------------------------------------------------

PT = PlanarTree
OF = OrderedForest

b = PT([])                    # bullet
c2 = PT([[]])                 # chain2
c3 = PT([[[]]])               # chain3
ch = PT([[], []])             # cherry
c4 = PT([[[[]]]])             # chain4
bch = PT([[[], []]])          # B+(cherry)
t43 = PT([[], [[]]])          # B+(b, c2)
t44 = PT([[[]], []])          # B+(c2, b)
cor3 = PT([[], [], []])       # corolla3


def _decode(val):
    """Accept either a tree or a tuple of trees; return OrderedForest."""
    if isinstance(val, PlanarTree):
        return OF((val,))
    return OF(tuple(val))


def _dict_form(cp):
    """Convert a TensorProductSum to {(left_key, right_key): coef}."""
    d = {}
    for c, l, r in cp.term_list:
        lk = tuple(t.list_repr for t in l.tree_list if t.list_repr is not None)
        rk = tuple(t.list_repr for t in r.tree_list if t.list_repr is not None)
        d[(lk, rk)] = d.get((lk, rk), 0) + c
    return {k: v for k, v in d.items() if v != 0}


def _expected_dict(terms):
    """Build expected dict from a list of (coef, left, right) where each
    side is either a tree (interpreted as single-tree forest) or a tuple
    of trees (interpreted as multi-tree forest).  () represents the empty
    forest."""
    d = {}
    for c, lft, rgt in terms:
        lk = tuple(
            (t.list_repr for t in (lft if isinstance(lft, tuple) else (lft,)))
        ) if lft != () else ()
        rk = tuple(
            (t.list_repr for t in (rgt if isinstance(rgt, tuple) else (rgt,)))
        ) if rgt != () else ()
        # Filter out Nones (empty trees)
        lk = tuple(x for x in lk if x is not None)
        rk = tuple(x for x in rk if x is not None)
        d[(lk, rk)] = d.get((lk, rk), 0) + c
    return {k: v for k, v in d.items() if v != 0}


# --- Reference data extracted verbatim from Table 5 ---------------------
# Each entry: (forest, [(coeff, left, right), ...]).
# left/right is either a tree, a tuple of trees, or () for the empty forest.

REFERENCES = [
    # --- order 1 ---
    # \ab : \ab⊗1 + 1⊗\ab
    (OF((b,)),
        [(1, b, ()), (1, (), b)]),
    # --- order 2 ---
    # \aabb : \aabb⊗1 + \ab⊗\ab + 1⊗\aabb
    (OF((c2,)),
        [(1, c2, ()), (1, b, b), (1, (), c2)]),
    # \ab\ab : \ab\ab⊗1 + \ab⊗\ab + 1⊗\ab\ab
    (OF((b, b)),
        [(1, (b, b), ()), (1, b, b), (1, (), (b, b))]),
    # --- order 3 ---
    # \aaabbb (chain3) : chain3⊗1 + b⊗chain2 + chain2⊗b + 1⊗chain3
    (OF((c3,)),
        [(1, c3, ()), (1, b, c2), (1, c2, b), (1, (), c3)]),
    # \aababb (cherry) : cherry⊗1 + (b,b)⊗b + b⊗chain2 + 1⊗cherry
    (OF((ch,)),
        [(1, ch, ()), (1, (b, b), b), (1, b, c2), (1, (), ch)]),
    # \ab\aabb : (b,c2)⊗1 + 2(b,b)⊗b + b⊗chain2 + b⊗(b,b) + 1⊗(b,c2)
    (OF((b, c2)),
        [(1, (b, c2), ()), (2, (b, b), b), (1, b, c2),
         (1, b, (b, b)), (1, (), (b, c2))]),
    # \aabb\ab : (chain2,b)⊗1 + chain2⊗b + b⊗(b,b) + 1⊗(chain2,b)
    (OF((c2, b)),
        [(1, (c2, b), ()), (1, c2, b), (1, b, (b, b)), (1, (), (c2, b))]),
    # \ab\ab\ab : (b,b,b)⊗1 + (b,b)⊗b + b⊗(b,b) + 1⊗(b,b,b)
    (OF((b, b, b)),
        [(1, (b, b, b), ()), (1, (b, b), b), (1, b, (b, b)),
         (1, (), (b, b, b))]),
    # --- order 4: the named trees ---
    # \aaaabbbb (chain4) : chain4⊗1 + chain3⊗b + chain2⊗chain2 + b⊗chain3 + 1⊗chain4
    (OF((c4,)),
        [(1, c4, ()), (1, c3, b), (1, c2, c2), (1, b, c3), (1, (), c4)]),
    # \aaababbb = B+(cherry) : bch⊗1 + cherry⊗b + (b,b)⊗chain2 + b⊗chain3 + 1⊗bch
    (OF((bch,)),
        [(1, bch, ()), (1, ch, b), (1, (b, b), c2),
         (1, b, c3), (1, (), bch)]),
    # \aabaabbb = t43 : t43⊗1 + (b,chain2)⊗b + 2(b,b)⊗chain2 + b⊗chain3 + b⊗cherry + 1⊗t43
    (OF((t43,)),
        [(1, t43, ()), (1, (b, c2), b), (2, (b, b), c2),
         (1, b, c3), (1, b, ch), (1, (), t43)]),
    # \aaabbabb = t44 : t44⊗1 + (chain2,b)⊗b + chain2⊗chain2 + b⊗cherry + 1⊗t44
    (OF((t44,)),
        [(1, t44, ()), (1, (c2, b), b), (1, c2, c2),
         (1, b, ch), (1, (), t44)]),
    # \aabababb = corolla3 : corolla3⊗1 + (b,b,b)⊗b + (b,b)⊗chain2 + b⊗cherry + 1⊗corolla3
    (OF((cor3,)),
        [(1, cor3, ()), (1, (b, b, b), b), (1, (b, b), c2),
         (1, b, ch), (1, (), cor3)]),
    # --- order 4: the multi-tree forests ---
    # \ab\aaabbb = (b, chain3) : (b,c3)⊗1 + (b,chain2)⊗b + (chain2,b)⊗b + chain2⊗(b,b)
    #   + 2(b,b)⊗chain2 + b⊗(b,chain2) + b⊗chain3 + 1⊗(b,c3)
    (OF((b, c3)),
        [(1, (b, c3), ()),
         (1, (b, c2), b),
         (1, (c2, b), b),
         (1, c2, (b, b)),
         (2, (b, b), c2),
         (1, b, (b, c2)),
         (1, b, c3),
         (1, (), (b, c3))]),
    # \aaabbb\ab = (chain3, b) : (c3,b)⊗1 + chain3⊗b + chain2⊗(b,b) + b⊗(chain2,b) + 1⊗(c3,b)
    (OF((c3, b)),
        [(1, (c3, b), ()),
         (1, c3, b),
         (1, c2, (b, b)),
         (1, b, (c2, b)),
         (1, (), (c3, b))]),
    # \ab\aababb = (b, cherry) : (b,ch)⊗1 + 3(b,b,b)⊗b + (b,b)⊗(b,b) + 2(b,b)⊗chain2
    #   + b⊗(b,chain2) + b⊗cherry + 1⊗(b,cherry)
    (OF((b, ch)),
        [(1, (b, ch), ()),
         (3, (b, b, b), b),
         (1, (b, b), (b, b)),
         (2, (b, b), c2),
         (1, b, (b, c2)),
         (1, b, ch),
         (1, (), (b, ch))]),
    # \aababb\ab = (cherry, b) : (ch,b)⊗1 + cherry⊗b + (b,b)⊗(b,b) + b⊗(chain2,b) + 1⊗(ch,b)
    (OF((ch, b)),
        [(1, (ch, b), ()),
         (1, ch, b),
         (1, (b, b), (b, b)),
         (1, b, (c2, b)),
         (1, (), (ch, b))]),
    # \aabb\aabb = (chain2, chain2) : (c2,c2)⊗1 + (chain2,b)⊗b + (b,chain2)⊗b
    #   + chain2⊗chain2 + 2(b,b)⊗(b,b) + b⊗(b,chain2) + b⊗(chain2,b) + 1⊗(c2,c2)
    (OF((c2, c2)),
        [(1, (c2, c2), ()),
         (1, (c2, b), b),
         (1, (b, c2), b),
         (1, c2, c2),
         (2, (b, b), (b, b)),
         (1, b, (b, c2)),
         (1, b, (c2, b)),
         (1, (), (c2, c2))]),
    # \ab\ab\aabb = (b,b,chain2) : (b,b,c2)⊗1 + 3(b,b,b)⊗b + 2(b,b)⊗(b,b)
    #   + (b,b)⊗chain2 + b⊗(b,b,b) + b⊗(b,chain2) + 1⊗(b,b,c2)
    (OF((b, b, c2)),
        [(1, (b, b, c2), ()),
         (3, (b, b, b), b),
         (2, (b, b), (b, b)),
         (1, (b, b), c2),
         (1, b, (b, b, b)),
         (1, b, (b, c2)),
         (1, (), (b, b, c2))]),
    # \ab\aabb\ab = (b,chain2,b) : (b,c2,b)⊗1 + (b,chain2)⊗b + 2(b,b)⊗(b,b)
    #   + b⊗(b,b,b) + b⊗(chain2,b) + 1⊗(b,c2,b)
    (OF((b, c2, b)),
        [(1, (b, c2, b), ()),
         (1, (b, c2), b),
         (2, (b, b), (b, b)),
         (1, b, (b, b, b)),
         (1, b, (c2, b)),
         (1, (), (b, c2, b))]),
    # \aabb\ab\ab = (chain2,b,b) : (c2,b,b)⊗1 + (chain2,b)⊗b + chain2⊗(b,b)
    #   + b⊗(b,b,b) + 1⊗(c2,b,b)
    (OF((c2, b, b)),
        [(1, (c2, b, b), ()),
         (1, (c2, b), b),
         (1, c2, (b, b)),
         (1, b, (b, b, b)),
         (1, (), (c2, b, b))]),
    # \ab\ab\ab\ab = (b,b,b,b) : (b,b,b,b)⊗1 + (b,b,b)⊗b + (b,b)⊗(b,b)
    #   + b⊗(b,b,b) + 1⊗(b,b,b,b)
    (OF((b, b, b, b)),
        [(1, (b, b, b, b), ()),
         (1, (b, b, b), b),
         (1, (b, b), (b, b)),
         (1, b, (b, b, b)),
         (1, (), (b, b, b, b))]),
]


class ForestCoproductReferenceTests(unittest.TestCase):
    """Every entry of Table 5 (Munthe-Kaas & Wright 2008, page ~37,
    forests of order 1-4) must match kauri's computed forest coproduct
    exactly, including signs and multinomial coefficients."""

    def _check(self, forest, expected_terms):
        got = _dict_form(forest_coproduct_impl(forest))
        expected = _expected_dict(expected_terms)
        self.assertEqual(got, expected,
            msg=f"Delta_forest mismatch for {forest.tree_list}")


def _make_test(forest, expected):
    def test(self):
        self._check(forest, expected)
    return test


for idx, (forest, expected) in enumerate(REFERENCES):
    # Build a human-readable name
    trees = [t for t in forest.tree_list if t.list_repr is not None]
    name_parts = []
    for t in trees:
        name_parts.append(str(t.list_repr).replace(', ', '_').replace(' ', ''))
    name = "test_" + ("__".join(name_parts) if name_parts else "empty")
    # Avoid invalid chars in test name
    name = name.replace('(', '').replace(')', '').replace(',', '').replace("'", "")
    name = name[:80]
    # Append index to avoid collisions
    name = f"{name}_{idx:02d}"
    setattr(ForestCoproductReferenceTests, name, _make_test(forest, expected))


if __name__ == "__main__":
    unittest.main()
