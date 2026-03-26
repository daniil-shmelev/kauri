Display
========================

.. automodule:: kauri.display

.. rubric:: Displaying a tree

.. kauri-exec::

   import kauri as kr
   t = kr.Tree([[],[[]]])
   kr.display(t)

.. rubric:: Displaying a colored tree

.. kauri-exec::

   import kauri as kr
   t = kr.Tree([[[3],2],[1],0])
   kr.display(t)

.. rubric:: Displaying a ForestSum

.. kauri-exec::

   import kauri as kr
   import kauri.bck as bck
   t = kr.Tree([[],[]])
   kr.display(bck.antipode(t))

.. autofunction:: kauri.display.display