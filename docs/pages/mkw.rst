MKW Hopf Algebra
========================

.. automodule:: kauri.mkw

.. rubric:: Example: MKW coproduct

.. kauri-exec::

   import kauri as kr
   import kauri.mkw as mkw
   t = kr.PlanarTree([[],[]])
   kr.display(mkw.coproduct(t))

.. rubric:: Example: Shuffle product

.. kauri-exec::

   import kauri as kr
   import kauri.mkw as mkw
   t1 = kr.PlanarTree([])
   t2 = kr.PlanarTree([[]])
   kr.display(mkw.shuffle_product(t1, t2))

.. autodata:: counit
.. autodata:: antipode

.. autofunction:: coproduct
.. autofunction:: shuffle_product
.. autofunction:: kauri.mkw.map_product
.. autofunction:: kauri.mkw.map_power
