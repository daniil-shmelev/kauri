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

.. rubric:: Displaying a forest

.. kauri-exec::

   import kauri as kr
   f = kr.CommutativeForest((kr.Tree([]), kr.Tree([[]]), kr.Tree([])))
   kr.display(f)

.. rubric:: Displaying a ForestSum

.. kauri-exec::

   import kauri as kr
   import kauri.bck as bck
   t = kr.Tree([[],[]])
   kr.display(bck.antipode(t))

.. rubric:: Displaying a TensorProductSum

.. kauri-exec::

   import kauri as kr
   import kauri.bck as bck
   t = kr.Tree([[],[]])
   kr.display(bck.coproduct(t))

.. rubric:: Displaying a planar tree and ordered forest

.. kauri-exec::

   import kauri as kr
   t = kr.PlanarTree([[],[[]]])
   f = kr.NoncommutativeForest((kr.PlanarTree([[]]), kr.PlanarTree([])))
   kr.display(t, f)

.. rubric:: Displaying multiple objects side by side

Multiple arguments are rendered in a single image, analogous to
``print(a, b, c)``.  Strings are rendered as text labels:

.. kauri-exec::

   import kauri as kr
   t = kr.Tree([[], [[]]])
   kr.display(t, "\u2192", t.unjoin())

.. autofunction:: kauri.display.display