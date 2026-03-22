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
This module provides instances of ``kauri.Map`` related to the odd-even
decomposition applied to the BCK Hopf algebra :cite:`shmelev2025ees, aguiar2006combinatorial`.
"""

from .trees import Tree
from .bck import antipode
from .generic_algebra import _apply
from .maps import Map, ident, sign
from functools import cache

@cache
def _id_sqrt(self): #Id^{1/2}
    if self.equals(Tree(None)):
        return Tree(None) * 1
    if self.equals(Tree([])):
        return Tree([]) * 0.5
    else:
        out = (ident ** 2)(self) - 2 * self
        out = _apply(out, _id_sqrt)
        out = (self - out) * 0.5
        out = out.simplify()
        return out

id_sqrt = Map(_id_sqrt)
id_sqrt.__doc__ = """
The square root of the identity map, :math:`\\mathrm{Id}^{1/2}`. The unique
multiplicative map such that :math:`\\mathrm{Id}^{1/2} \\cdot \\mathrm{Id}^{1/2} = \\mathrm{Id}`
:cite:`shmelev2025ees`.
"""
minus = ((sign & antipode) * ident) & id_sqrt
minus.__doc__ = """
The minus operation, defined by :cite:`shmelev2025ees`

.. math::
    
    \\tau^- = \\mu \\circ (\\overline{S} \\otimes \\mathrm{Id}) \\circ \\Delta \\circ \\mathrm{Id}^{1/2}(\\tau)

where :math:`\\overline{S}(\\tau) := (-1)^{|\\tau|}S(\\tau)`.
"""
plus = ident * (minus & antipode)
plus.__doc__ = """
The plus operation, defined by :cite:`shmelev2025ees`

.. math::

    \\tau^- = \\mu \\circ (\\mathrm{Id} \\otimes (\\cdot)^- \\circ S) \\circ \\Delta(\\tau)

"""