Tree Objects
========================

.. automodule:: kauri.trees

.. note::
         The classes `Tree`, `Forest` and `ForestSum` are immutable and hashable.
         The hash is generated in such a way that two elements of the same class which are equivalent
         (e.g. two different orderings of the same tree) will have the same hash. 
         However, this is not the case across classes. For example, for a Tree t, `hash(t)`, `hash(t.as_forest())`
         and `hash(t.as_forest_sum())` are different.

.. toctree::
   :titlesonly:

   tree_objects/tree
   tree_objects/forest
   tree_objects/forestsum