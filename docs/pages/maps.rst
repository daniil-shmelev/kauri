Maps
========================

.. automodule:: kauri.maps

.. autoclass:: kauri.maps.Map
   :members:
   :special-members: __mul__, __pow__, __add__, __matmul__, __and__

Batch evaluation
----------------

Use :meth:`Map.evaluate_many` to evaluate one map on several inputs while
preserving input order.

.. code-block:: python

    from kauri import Map, trees_of_order

    weights = Map(lambda t: 1 / t.factorial())
    inputs = tuple(trees_of_order(5))

    serial_values = weights.evaluate_many(inputs)
    threaded_values = weights.evaluate_many(inputs, workers=4)


Instances
------------

We provide a few common instances of the :class:`Map` class for convenience.

.. autodata:: ident
    :no-value:
.. autodata:: sign
    :no-value:
.. autodata:: exact_weights
    :no-value:
.. autodata:: omega
    :no-value:
