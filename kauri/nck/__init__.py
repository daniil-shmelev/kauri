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
The ``kauri.nck`` sub-package implements the noncommutative Connes-Kreimer (NCK)
Hopf algebra :cite:`foissy2002les`
:math:`(H, \\Delta_{NCK}, \\mu, \\varepsilon_{NCK}, \\emptyset, S_{NCK})`, defined as follows.

- :math:`H` is the set of all planar (ordered) rooted trees, where sibling order matters.
- The unit :math:`\\emptyset` is the empty ordered forest.
- The counit map is defined by :math:`\\varepsilon_{NCK}(\\emptyset) = 1`,
  :math:`\\varepsilon_{NCK}(t) = 0` for all :math:`\\emptyset \\neq t \\in H`.
- Multiplication :math:`\\mu : H \\otimes H \\to H` is defined as the
  noncommutative (ordered) concatenation of forests.
- Comultiplication :math:`\\Delta_{NCK} : H \\to H \\otimes H` is defined as

  .. math::

      \\Delta_{NCK}(t) = t \\otimes \\emptyset + \\emptyset \\otimes t + \\sum_{s \\subset t} [t \\setminus s] \\otimes s

  where the sum runs over all proper rooted subtrees :math:`s` of :math:`t`, and :math:`[t \\setminus s]`
  is the ordered forest remaining after erasing :math:`s` from :math:`t`, preserving sibling order.
- The antipode :math:`S_{NCK}` is defined by :math:`S_{NCK}(\\bullet) = -\\bullet` and

  .. math::

      S_{NCK}(t) = -t - \\sum_{s \\subset t} S_{NCK}([t \\setminus s]) \\, s.

.. note::

    Unlike the non-planar BCK algebra, the NCK algebra is neither
    commutative nor cocommutative, so the antipode is **not** an involution
    (:math:`S^2 \\neq \\mathrm{id}` in general).
"""

from .nck import antipode, counit, coproduct, map_power, map_product

__all__ = ['coproduct', 'counit', 'antipode', 'map_product', 'map_power']
