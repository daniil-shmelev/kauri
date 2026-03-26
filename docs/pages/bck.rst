BCK Hopf Algebra
========================

.. automodule:: kauri.bck

.. rubric:: Example: BCK coproduct

.. kauri-exec::

   import kauri as kr
   import kauri.bck as bck
   t = kr.Tree([[],[[]]])
   kr.display(bck.coproduct(t))

.. rubric:: Example: BCK antipode

.. kauri-exec::

   import kauri as kr
   import kauri.bck as bck
   t = kr.Tree([[],[[]]])
   kr.display(bck.antipode(t))

.. autodata:: counit
.. autodata:: antipode

.. autofunction:: coproduct
.. autofunction:: kauri.bck.map_product
.. autofunction:: kauri.bck.map_power

.. include:: ../refs.rst