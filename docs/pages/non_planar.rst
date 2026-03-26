Non-Planar Trees
========================

Non-planar (unordered) rooted trees treat children as a multiset — sibling order
does not matter, so ``Tree([[], [[]]])`` and ``Tree([[[]],[]])`` represent the
same tree.

The following pages cover non-planar tree types and generation:

.. toctree::
   :titlesonly:

   non_planar/tree_objects
   non_planar/tree_generation

Non-planar trees are also used in :doc:`maps`, :doc:`rk`, :doc:`bseries`,
and the :doc:`hopf_algebras` (specifically :doc:`bck`, :doc:`cem`, and :doc:`gl`).