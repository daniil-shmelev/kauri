Odd-Even Decomposition
========================

.. automodule:: kauri.oddeven

.. rubric:: Example: Square root of the identity map

.. kauri-exec::

   import kauri as kr
   import kauri.oddeven as oddeven
   for t in kr.trees_of_order(4):
       print(repr(t), '->', oddeven.id_sqrt(t))

.. autodata:: id_sqrt
.. autodata:: minus
.. autodata:: plus

.. include:: ../refs.rst