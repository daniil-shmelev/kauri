CEM Hopf Algebra
========================

.. automodule:: kauri.cem

.. rubric:: Example: CEM coproduct

.. kauri-exec::

   import kauri as kr
   import kauri.cem as cem
   t = kr.Tree([[],[[]]])
   kr.display(cem.coproduct(t))

.. autodata:: counit
.. autodata:: antipode

.. autofunction:: coproduct
.. autofunction:: kauri.cem.map_product
.. autofunction:: kauri.cem.map_power

.. include:: ../refs.rst