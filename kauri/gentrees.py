from .trees import *
from .utils import _level_sequence_to_list_repr, _next_layout

def trees_up_to_order(n):
    """
    Yields the trees up to and including order :math:`n`, ordered by the lexicographic order.

    :param n: Maximum order
    :type n: int
    :yields: The next tree in lexicographic order, as long as the order of the tree does not exceed :math:`n`.
    :rtype: Tree

    Example usage::

            for t in trees_up_to_order(4):
                display(t.antipode())
    """
    t = Tree(None)
    while t.nodes() <= n:
        yield t
        t = next(t)

def trees_of_order(n):
    """
    Yields the trees of order :math:`n`, ordered by the lexicographic order.

    :param n: Order
    :type n: int
    :yields: The next tree in lexicographic order, as long as the order of the tree is :math:`n`.
    :rtype: Tree

    Example usage::

            for t in trees_of_order(4):
                display(t.antipode())
    """
    t = Tree(_level_sequence_to_list_repr([i for i in range(n)]))
    while t.nodes() == n:
        yield t
        t = next(t)