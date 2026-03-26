Planar Odd-Even Decomposition
==============================

.. automodule:: kauri.planar_oddeven

.. rubric:: Example: Square root of the identity map

.. kauri-exec::

   import kauri as kr
   import kauri.planar_oddeven as planar_oddeven
   for t in kr.planar_trees_of_order(4):
       print(repr(t), '->', planar_oddeven.id_sqrt(t))

.. autodata:: id_sqrt
.. autodata:: minus
.. autodata:: plus

