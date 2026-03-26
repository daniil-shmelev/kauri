"""
The ``kauri.mkw`` sub-package implements the Munthe-Kaas--Wright (MKW)
Hopf algebra :cite:`munthe2008hopf`
:math:`(H, \\Delta_{MKW}, \\shuffle, \\varepsilon_{MKW}, \\emptyset, S_{MKW})`,
defined as follows.

- :math:`H` is the set of all planar (ordered) rooted trees, where sibling order matters.
- The unit :math:`\\emptyset` is the empty ordered forest.
- The counit map is defined by :math:`\\varepsilon_{MKW}(\\emptyset) = 1`,
  :math:`\\varepsilon_{MKW}(t) = 0` for all :math:`\\emptyset \\neq t \\in H`.
- Multiplication :math:`\\shuffle : H \\otimes H \\to H` is the commutative
  shuffle product of ordered forests.
- Comultiplication :math:`\\Delta_{MKW} : H \\to H \\otimes H` is defined
  recursively by :math:`\\Delta_{MKW}(B_+(t_1, \\ldots, t_k)) = B_+(t_1, \\ldots, t_k)
  \\otimes \\emptyset + (\\mathrm{id} \\otimes B_+)(\\tilde{\\Delta}(t_1) \\bar{\\shuffle}
  \\cdots \\bar{\\shuffle} \\tilde{\\Delta}(t_k))`, where
  :math:`\\tilde{\\Delta} = \\Delta - \\mathrm{id} \\otimes \\emptyset` is the reduced
  coproduct and :math:`\\bar{\\shuffle}` shuffles left tensor factors while concatenating
  right factors.
- The antipode :math:`S_{MKW}` is a **homomorphism** (since the algebra is commutative),
  unlike the NCK antipode which is an anti-homomorphism.

.. note::

    The MKW algebra is **commutative** but **not cocommutative**.  It is the
    correct Hopf algebra for the analysis of Lie--Butcher series and Lie group
    integrators :cite:`munthe2008hopf`.  The closely related noncommutative
    Connes--Kreimer algebra (:mod:`kauri.nck`) uses concatenation rather than
    shuffle as its product.
"""

from .mkw import antipode, counit, coproduct, shuffle_product, map_product, map_power

__all__ = ['coproduct', 'counit', 'antipode', 'shuffle_product', 'map_product', 'map_power']
