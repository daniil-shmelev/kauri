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
The ``kauri.pbck`` sub-package implements the planar (ordered) Butcher-Connes-Kreimer
Hopf algebra :cite:`munthe2008hopf`
:math:`(H, \\Delta, \\mu, \\varepsilon, \\emptyset, S)`, defined as follows.

- :math:`H` is the set of all planar (ordered) rooted trees, where sibling order matters.
- The unit :math:`\\emptyset` is the empty ordered forest.
- The counit map is defined by :math:`\\varepsilon(\\emptyset) = 1`,
  :math:`\\varepsilon(t) = 0` for all :math:`\\emptyset \\neq t \\in H`.
- Multiplication :math:`\\mu : H \\otimes H \\to H` is the
  noncommutative (ordered) concatenation of forests.
- Comultiplication :math:`\\Delta : H \\to H \\otimes H` is defined recursively using
  admissible cuts, preserving sibling order:

  .. math::

      \\Delta(t) = t \\otimes \\emptyset + (\\mathrm{id} \\otimes B_+) \\Delta(B_-(t))

- The antipode :math:`S` is defined recursively:

  .. math::

      S(t) = -t - \\sum_{\\text{proper}} S(\\text{branches}) \\cdot \\text{subtree}

.. note::

    Unlike the non-planar BCK algebra, the planar BCK algebra is neither
    commutative nor cocommutative, so the antipode is **not** an involution
    (:math:`S^2 \\neq \\mathrm{id}` in general).
"""

from .pbck import antipode, counit, coproduct, map_power, map_product
