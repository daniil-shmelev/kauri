from .trees import *

def next_layout(layout):
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
    if t == Tree(None):
        return Tree([])

    layout = t.layout()
    next = next_layout(layout)
    return Tree(layoutToListRepr_(next))

def trees_up_to_order(n):
    t = Tree(None)
    while t.numNodes() <= n:
        yield t
        t = next_tree(t)

def trees_of_order(n):
    t = Tree(layoutToListRepr_([i for i in range(n)]))
    while t.numNodes() == n:
        yield t
        t = next_tree(t)