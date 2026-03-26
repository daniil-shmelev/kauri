Grossman--Larson Hopf Algebra
==============================

.. automodule:: kauri.gl

.. rubric:: Example: GL coproduct

.. kauri-exec::

   import kauri as kr
   import kauri.gl as gl
   t = kr.Tree([[],[[]]])
   kr.display(gl.coproduct(t))

.. rubric:: Example: GL grafting product

.. kauri-exec::

   import kauri as kr
   import kauri.gl as gl
   t1 = kr.Tree([[],[]])
   t2 = kr.Tree([[]])
   kr.display(gl.product(t1, t2))

.. autodata:: counit
.. autodata:: antipode

.. autofunction:: coproduct
.. autofunction:: product
.. autofunction:: kauri.gl.map_product
.. autofunction:: kauri.gl.map_power

.. include:: ../refs.rst
