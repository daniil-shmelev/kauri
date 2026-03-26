Getting Started
========================

This page walks through the basics of kauri: creating trees, inspecting their
properties, computing coproducts, and checking Runge--Kutta order conditions.

Creating and displaying a tree
-------------------------------

A rooted tree is built from nested lists. The empty list ``[]`` represents a
leaf (single vertex), and nesting adds branches.

.. kauri-exec::

   import kauri as kr

   t = kr.Tree([[], [[]]])
   kr.display(t)

Tree properties
-------------------------------

Every tree carries combinatorial data used in B-series and order condition
theory.

.. kauri-exec::

   import kauri as kr

   t = kr.Tree([[], [[]]])
   print("nodes  :", t.nodes())
   print("factorial:", t.factorial())
   print("sigma  :", t.sigma())
   print("density :", t.density())
   print("alpha  :", t.alpha())

BCK coproduct
-------------------------------

The Butcher--Connes--Kreimer coproduct decomposes a tree into admissible cuts.

.. kauri-exec::

   import kauri as kr
   import kauri.bck as bck

   t = kr.Tree([[], [[]]])
   kr.display(bck.coproduct(t))

Runge--Kutta methods
-------------------------------

Kauri ships with several predefined Runge--Kutta methods. You can query their
order and inspect elementary weights.

.. kauri-exec::

   import kauri as kr
   from kauri.rk_methods import rk4

   print("RK4 order:", rk4.order())
   ew = rk4.elementary_weights_map()
   for t in kr.trees_of_order(4):
       print(repr(t), "weight:", ew(t))

Enumerating trees
-------------------------------

Kauri can generate all rooted trees of a given order.

.. kauri-exec::

   import kauri as kr

   for t in kr.trees_of_order(4):
       kr.display(t)

Forest arithmetic
-------------------------------

Multiplying trees produces forests, and adding them produces formal linear
combinations (forest sums).

.. kauri-exec::

   import kauri as kr

   t1 = kr.Tree([])
   t2 = kr.Tree([[]])
   forest = t1 * t2
   print("Forest:", repr(forest))

   forest_sum = t1 + t2 - 2 * kr.Tree([[], []])
   print("ForestSum:", repr(forest_sum))

Planar trees
-------------------------------

Planar trees preserve sibling order. Two trees that are equal in the
non-planar setting can be distinct when planar.

.. kauri-exec::

   import kauri as kr

   t1 = kr.PlanarTree([[], [[]]])
   t2 = kr.PlanarTree([[[]], []])
   print("t1 == t2:", t1 == t2)
   kr.display(t1)
   kr.display(t2)
