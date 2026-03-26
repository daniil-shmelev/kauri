Planar BCK Hopf Algebra
========================

.. automodule:: kauri.pbck

.. rubric:: Example: Planar BCK coproduct

.. kauri-exec::

   import kauri as kr
   import kauri.pbck as pbck
   t = kr.PlanarTree([[],[[]]])
   kr.display(pbck.coproduct(t))

.. autodata:: counit
.. autodata:: antipode

.. autofunction:: coproduct
.. autofunction:: kauri.pbck.map_product
.. autofunction:: kauri.pbck.map_power

