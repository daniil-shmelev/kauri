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

"""
Back-end for the BCK module
"""
from functools import cache
from ..trees import (Tree, TensorProductSum,
                     EMPTY_TREE, EMPTY_FOREST, EMPTY_FOREST_SUM)
from ..generic_algebra import _forest_apply

def _counit(t):
    # Return 1 if t is the empty tree, otherwise 0
    return 1 if t.list_repr is None else 0

@cache
def _antipode(t):
    if t.list_repr is None:
        return EMPTY_FOREST_SUM # Antipode of empty tree is the empty tree
    if t.list_repr == tuple():
        return -t # Antipode of singleton is the negative singleton

    cp = _coproduct(t)
    out = -t.as_forest_sum() # First term, -t
    for c, branches, subtree_ in cp: # Remaining terms
        subtree = subtree_[0] # Convert from Forest to Tree
        if subtree.equals(t) or subtree.equals(EMPTY_TREE):
            continue # We've already included the -t term at the start, so move on
        out = out - c * _forest_apply(branches, _antipode) * subtree

    return out.simplify()

@cache
def _coproduct(t):
    # This follows the recursive definition of https://arxiv.org/pdf/hep-th/9808042
    # using B_- and B_+
    if t == Tree(None):
        return TensorProductSum(( (1, EMPTY_FOREST, EMPTY_FOREST), )) # Tree(None) @ Tree(None)
    if len(t.list_repr) == 1:
        return TensorProductSum(( (1, EMPTY_FOREST, t.as_forest()), (1, t.as_forest(), EMPTY_FOREST) )) # Tree(None) @ t + t @ Tree(None)

    root_color = t.list_repr[-1]
    branches = t.unjoin()

    cp_prod = 1
    for subtree in branches:
        cp = _coproduct(subtree)
        cp_prod = cp_prod * cp

    # Return t \otimes \emptyset + (id \otimes B_+)[\Delta(B_-(t))]
    out = t @ Tree(None) + TensorProductSum(tuple((c, f1, f2.join(root_color)) for c, f1, f2 in cp_prod))
    return out.simplify()
