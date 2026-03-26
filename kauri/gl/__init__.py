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
The ``kauri.gl`` sub-package implements the Grossman-Larson (GL) :cite:`grossman1989hopf` Hopf algebra
:math:`(H, \\Delta_{GL}, \\cdot_{GL}, \\varepsilon_{GL}, \\bullet, S_{GL})`, defined as follows.

- :math:`H` is the set of all non-planar rooted trees.
- The unit is the single-vertex tree :math:`\\bullet`.
- The counit map is defined by :math:`\\varepsilon_{GL}(\\bullet) = 1`,
  :math:`\\varepsilon_{GL}(t) = 0` for all :math:`|t| > 1`.
- The coproduct :math:`\\Delta_{GL}` is cocommutative, and splits the children of the root
  into all possible subsets: for :math:`t = B_+(t_1, \\ldots, t_k)`,

  .. math::

      \\Delta_{GL}(t) = \\sum_{S \\subseteq \\{1,\\ldots,k\\}} B_+(t_i : i \\in S) \\otimes B_+(t_j : j \\notin S)

- The product :math:`\\cdot_{GL}` (grafting) sums over all ways of attaching the children of
  the right tree to vertices of the left tree. The product is noncommutative.
- The antipode :math:`S_{GL}(\\bullet) = \\bullet` and

  .. math::

      S_{GL}(t) = -t - \\sum_{\\substack{S \\subset \\{1,\\ldots,k\\} \\\\ S \\neq \\emptyset,\\, S \\neq \\{1,\\ldots,k\\}}} S_{GL}(B_+(t_i : i \\in S)) \\cdot_{GL} B_+(t_j : j \\notin S)
"""

from .gl import antipode, counit, coproduct, product, map_power, map_product

__all__ = ['coproduct', 'counit', 'antipode', 'product', 'map_product', 'map_power']
