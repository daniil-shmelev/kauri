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
The ``kauri.pgl`` sub-package implements the planar (ordered) Grossman-Larson
Hopf algebra
:math:`(H, \\Delta_{PGL}, \\cdot_{PGL}, \\varepsilon_{PGL}, \\bullet, S_{PGL})`,
defined as follows.

- :math:`H` is the set of all planar (ordered) rooted trees, where sibling order matters.
- The unit is the single-vertex tree :math:`\\bullet`.
- The counit map is defined by :math:`\\varepsilon_{PGL}(\\bullet) = 1`,
  :math:`\\varepsilon_{PGL}(t) = 0` for all :math:`|t| > 1`.
- The coproduct :math:`\\Delta_{PGL}` splits the children of the root into all
  possible subsets, preserving sibling order on both sides:
  for :math:`t = B_+(t_1, \\ldots, t_k)`,

  .. math::

      \\Delta_{PGL}(t) = \\sum_{S \\subseteq \\{1,\\ldots,k\\}}
          B_+(t_i : i \\in S) \\otimes B_+(t_j : j \\notin S)

- The product :math:`\\cdot_{PGL}` (planar grafting) sums over all ways of attaching
  the children of the right tree to vertices of the left tree at all possible
  insertion positions among existing children.
- The antipode :math:`S_{PGL}` is defined recursively using the planar grafting product.

.. note::

    Unlike the non-planar GL algebra, the planar GL algebra is not cocommutative,
    so the antipode is **not** an involution (:math:`S^2 \\neq \\mathrm{id}` in general).
"""

from .pgl import antipode, counit, coproduct, product, map_power, map_product
