RootedTrees
========================


Tree
========================
.. autoclass:: rootedtrees.trees.Tree
   :members:
   :special-members: __mul__, __pow__, __add__, __eq__

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

.. autofunction:: rootedtrees.gentrees.next_tree
.. autofunction:: rootedtrees.gentrees.trees_of_order
.. autofunction:: rootedtrees.gentrees.trees_up_to_order

Runge--Kutta Schemes
========================

.. autofunction:: rootedtrees.rk.RK_symbolic_weight

.. autoclass:: rootedtrees.rk.RK
   :members:
   :special-members: __mul__, __pow__, __add__