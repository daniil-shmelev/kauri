from .trees import *
from .utils import _level_sequence_to_list_repr

def _next_layout(layout):
    p = len(layout) - 1
    while layout[p] == 1:
        p -= 1

    if p == 0:
        n = len(layout)
        return list(range(n + 1))

    q = p - 1
    while layout[q] != layout[p] - 1:
        q -= 1
    result = list(layout)
    for i in range(p, len(result)):
        result[i] = result[i - p + q]
    return result


def next_tree(t):
    """Given a tree t, generates the next tree with respect to the lexicographic order."""
    if t == Tree(None):
        return Tree([])

    layout = t.level_sequence()
    next = _next_layout(layout)
    return Tree(_level_sequence_to_list_repr(next))

def trees_up_to_order(n):
    """A generator function for the trees up to and including order n, ordered by the lexicographic order."""
    t = Tree(None)
    while t.nodes() <= n:
        yield t
        t = next_tree(t)

def trees_of_order(n):
    """A generator function for the trees of order n, ordered by the lexicographic order."""
    t = Tree(_level_sequence_to_list_repr([i for i in range(n)]))
    while t.nodes() == n:
        yield t
        t = next_tree(t)