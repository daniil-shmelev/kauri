RootedTrees
========================

.. note::
         The classes `Tree`, `Forest` and `ForestSum` are immutable and hashable.
         The hash is generated in such a way that two elements of the same class which are equivalent
         (e.g. two different orderings of the same tree) will have the same hash. 
         However, this is not the case across classes. For example, for a Tree t, `hash(t)`, `hash(t.as_forest())`
         and `hash(t.as_forest_sum())` are different.


Tree
========================
.. autoclass:: rootedtrees.trees.Tree
   :members:
   :special-members: __mul__, __pow__, __add__, __eq__, __next__

Forest
========================

.. autoclass:: rootedtrees.trees.Forest
   :members:
   :special-members: __mul__, __pow__, __add__, __eq__

ForestSum
========================

.. autoclass:: rootedtrees.trees.ForestSum
   :members:
   :special-members: __mul__, __pow__, __add__, __eq__

Maps
========================

.. autoclass:: rootedtrees.maps.Map
   :members:
   :special-members: __mul__, __pow__, __add__, __matmul__

Tree Generation
========================

.. autofunction:: rootedtrees.gentrees.trees_of_order
.. autofunction:: rootedtrees.gentrees.trees_up_to_order

Runge--Kutta Schemes
========================

.. autofunction:: rootedtrees.rk.RK_symbolic_weight
.. autofunction:: rootedtrees.rk.RK_order_cond

.. autoclass:: rootedtrees.rk.RK
   :members:
   :special-members: __mul__, __pow__, __add__