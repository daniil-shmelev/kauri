Non-Planar Trees
========================

Non-planar (unordered) rooted trees treat children as a multiset — sibling order
does not matter, so ``Tree([[], [[]]])`` and ``Tree([[[]],[]])`` represent the
same tree.

For the core types (:class:`~kauri.trees.Tree`, :class:`~kauri.trees.CommutativeForest`,
:class:`~kauri.trees.ForestSum`, :class:`~kauri.trees.TensorProductSum`),
see :doc:`tree_objects`.

For tree generation, see :doc:`tree_generation`.

Non-planar trees are also used in :doc:`maps`, :doc:`rk`, :doc:`bseries`,
and the :doc:`hopf_algebras` (specifically :doc:`bck`, :doc:`cem`, and :doc:`gl`).

.. toctree::
   :titlesonly:

   oddeven
