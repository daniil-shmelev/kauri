"""
Functions for generating rooted trees in lexicographic order, based on the algorithms of :cite:`beyer1980constant`.
"""
from typing import Generator

from .trees import Tree
from .utils import _level_sequence_to_list_repr

def trees_up_to_order(order : int) -> Generator[Tree, None, None]:
    """
    Yields the trees up to and including order :math:`n`, ordered by the lexicographic order.

    :param order: Maximum order
    :type order: int
    :yields: The next tree in lexicographic order, as long as the
        order of the tree does not exceed :math:`n`.
    :rtype: Tree

    Example usage::

            import kauri as kr
            import kauri.bck as bck

            for t in kr.trees_up_to_order(4):
                display(bck.antipode(t))
    """
    if not isinstance(order, int):
        raise TypeError("order must be an int, not " + str(type(order)))
    if order < 0:
        raise ValueError("order must be positive")

    t = Tree(None)
    while t.nodes() <= order:
        yield t
        t = next(t)

def trees_of_order(order : int) -> Generator[Tree, None, None]:
    """
    Yields the trees of order :math:`n`, ordered by the lexicographic order.

    :param order: Order
    :type order: int
    :yields: The next tree in lexicographic order, as long as the order of the tree is :math:`n`.
    :rtype: Tree

    Example usage::

            import kauri as kr
            import kauri.bck as bck

            for t in kr.trees_of_order(4):
                display(bck.antipode(t))
    """
    if not isinstance(order, int):
        raise TypeError("order must be an int, not " + str(type(order)))
    if order < 0:
        raise ValueError("order must be positive")

    t = Tree(_level_sequence_to_list_repr(list(range(order))))
    while t.nodes() == order:
        yield t
        t = next(t)
