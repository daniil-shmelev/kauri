Kauri
========================

.. note::
         The classes `Tree`, `Forest` and `ForestSum` are immutable and hashable.
         The hash is generated in such a way that two elements of the same class which are equivalent
         (e.g. two different orderings of the same tree) will have the same hash. 
         However, this is not the case across classes. For example, for a Tree t, `hash(t)`, `hash(t.as_forest())`
         and `hash(t.as_forest_sum())` are different.


Tree
========================
.. autoclass:: kauri.trees.Tree
   :members:
   :special-members: __mul__, __pow__, __add__, __eq__, __next__

Forest
========================

.. autoclass:: kauri.trees.Forest
   :members:
   :special-members: __mul__, __pow__, __add__, __eq__

ForestSum
========================

.. autoclass:: kauri.trees.ForestSum
   :members:
   :special-members: __mul__, __pow__, __add__, __eq__

Maps
========================

.. autoclass:: kauri.maps.Map
   :members:
   :special-members: __mul__, __pow__, __add__, __matmul__

Tree Generation
========================

.. autofunction:: kauri.gentrees.trees_of_order
.. autofunction:: kauri.gentrees.trees_up_to_order

Runge--Kutta Schemes
========================

.. autofunction:: kauri.rk.RK_symbolic_weight
.. autofunction:: kauri.rk.RK_order_cond

.. autoclass:: kauri.rk.RK
   :members:
   :special-members: __mul__, __pow__, __add__


References
========================

.. bibliography::