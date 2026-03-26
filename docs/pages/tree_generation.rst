Tree Generation
========================

.. automodule:: kauri.gentrees

Unlabelled Trees
----------------

.. rubric:: All non-planar trees of order 4

.. kauri-exec::

   import kauri as kr
   for t in kr.trees_of_order(4):
       kr.display(t)

.. autofunction:: kauri.gentrees.trees_of_order
.. autofunction:: kauri.gentrees.trees_up_to_order

Colored Trees
-------------

.. autofunction:: kauri.gentrees.colored_trees_of_order
.. autofunction:: kauri.gentrees.colored_trees_up_to_order

Planar Trees
------------

.. rubric:: All planar trees of order 4

.. kauri-exec::

   import kauri as kr
   for t in kr.planar_trees_of_order(4):
       kr.display(t)

.. autofunction:: kauri.gentrees.planar_trees_of_order
.. autofunction:: kauri.gentrees.planar_trees_up_to_order

Colored Planar Trees
--------------------

.. autofunction:: kauri.gentrees.colored_planar_trees_of_order
.. autofunction:: kauri.gentrees.colored_planar_trees_up_to_order

.. include:: ../refs.rst