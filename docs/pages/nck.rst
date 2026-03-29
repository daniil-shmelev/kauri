NCK Hopf Algebra
========================

.. automodule:: kauri.nck

.. rubric:: Example: NCK coproduct

.. kauri-exec::

   import kauri as kr
   import kauri.nck as nck
   t = kr.PlanarTree([[],[[]]])
   kr.display(nck.coproduct(t))

.. autodata:: counit
.. autodata:: antipode

.. autofunction:: coproduct
.. autofunction:: kauri.nck.map_product
.. autofunction:: kauri.nck.map_power
