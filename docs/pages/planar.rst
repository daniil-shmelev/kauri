Planar (Ordered) Trees
========================

Planar rooted trees preserve the left-to-right ordering of children, so
``PlanarTree([[], [[]]])`` and ``PlanarTree([[[]],[]])`` are distinct trees.

For the core types (:class:`~kauri.trees.PlanarTree`, :class:`~kauri.trees.NoncommutativeForest`,
:class:`~kauri.trees.ForestSum`, :class:`~kauri.trees.TensorProductSum`),
see :doc:`tree_objects`.

For planar tree generation, see :doc:`tree_generation`.

Planar trees are also used in :doc:`pbck` and :doc:`pgl`.

.. toctree::
   :titlesonly:

   planar_oddeven
