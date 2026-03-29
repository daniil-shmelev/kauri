Planar Grossman--Larson Hopf Algebra
======================================

.. automodule:: kauri.pgl

.. rubric:: Example: Planar GL coproduct

.. kauri-exec::

   import kauri as kr
   import kauri.pgl as pgl
   t = kr.PlanarTree([[],[[]]])
   kr.display(pgl.coproduct(t))

.. rubric:: Example: Planar GL grafting product

.. kauri-exec::

   import kauri as kr
   import kauri.pgl as pgl
   t1 = kr.PlanarTree([[],[]])
   t2 = kr.PlanarTree([[]])
   kr.display(pgl.product(t1, t2))

.. autodata:: counit
.. autodata:: antipode

.. autofunction:: coproduct
.. autofunction:: product
.. autofunction:: kauri.pgl.map_product
.. autofunction:: kauri.pgl.map_power

