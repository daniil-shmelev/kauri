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

.. rubric:: Displaying multiple objects side by side

Multiple arguments are rendered in a single image, analogous to
``print(a, b, c)``:

.. kauri-exec::

   import kauri as kr
   t1 = kr.Tree([])
   t2 = kr.Tree([[]])
   t3 = kr.Tree([[], [[]]])
   kr.display(t1, t2, t3)

.. autofunction:: kauri.display.display