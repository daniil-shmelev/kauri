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
The ``kauri.bck`` sub-package implements the Butcher-Connes-Kreimer (BCK) :cite:`connes1999hopf` Hopf algebra
:math:`(H, \\Delta_{BCK}, \\mu, \\varepsilon_{BCK}, \\emptyset, S_{BCK})`, defined as follows.

- :math:`H` is the set of all non-planar rooted trees.
- The unit :math:`\\emptyset` is the empty forest.
- The counit map is defined by :math:`\\varepsilon_{BCK}(\\emptyset) = 1`,
  :math:`\\varepsilon_{BCK}(t) = 0` for all :math:`\\emptyset \\neq t \\in H`.
- Multiplication :math:`\\mu : H \\otimes H \\to H` is defined as the
  commutative juxtaposition of two forests.
- Comultiplication :math:`\\Delta : H \\to H \\otimes H` is defined as

  .. math::

      \\Delta_{BCK}(t) = t \\otimes \\emptyset + \\emptyset \\otimes t + \\sum_{s \\subset t} [t \\setminus s] \\otimes s

  where the sum runs over all proper rooted subtrees :math:`s` of :math:`t`, and :math:`[t \\setminus s]`
  is the forest of all trees remaining after erasing :math:`s` from :math:`t`.
- The antipode :math:`S_{BCK}` is defined by :math:`S_{BCK}(\\bullet) = -\\bullet` and

  .. math::

      S_{BCK}(t) = -t - \\sum_{s \\subset t} (-1)^{n(t \\setminus s)} S_{BCK}([t \\setminus s]) s,

  where :math:`n(t \\setminus s)` is the number of trees in the forest :math:`[t \\setminus s]`.
"""

from .bck import antipode, counit, coproduct, map_power, map_product
