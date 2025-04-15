import math
from functools import cache

def _to_tuple(obj):
    if isinstance(obj, list):
        return tuple(_to_tuple(el) for el in obj)
    return obj

def _to_list(obj):
    if isinstance(obj, tuple):
        return list(_to_list(el) for el in obj)
    return obj

@cache
def _nodes(rep):
    if rep is None:
        return 0
    elif rep == tuple():
        return 1
    else:
        return 1 + sum(_nodes(r) for r in rep)

@cache
def _height(rep):
    if rep is None:
        return 0
    elif rep == tuple():
        return 1
    else:
        return 1 + max(_height(r) for r in rep)

@cache
def _factorial(rep):
    if rep is None:
        return 1, 0
    if rep == tuple():
        return 1, 1
    else:
        f = 1
        n = 1
        for r in rep:
            res = _factorial(r)
            f *= res[0]
            n += res[1]
        f *= n
        return f, n

@cache
def _sigma(rep):
    if rep is None or rep == tuple():
        return 1
    rep_dict = {}
    unique_rep = []
    for r in rep:
        if r in rep_dict.keys():
            rep_dict[r] += 1
        else:
            rep_dict[r] = 1
            unique_rep.append(r)

    out = 1
    for r in unique_rep:
        k = rep_dict[r]
        out *= math.factorial(k) * (_sigma(r) ** k)
    return out

@cache
def _sorted_list_repr(rep):
    if rep is None:
        return None
    elif rep == tuple():
        return tuple()
    else:
        return tuple(sorted(map(_sorted_list_repr, rep), reverse=True))

@cache
def _list_repr_to_level_sequence(rep):
    if rep is None:
        return []

    layout = [0]
    for r in rep:
        lay = _list_repr_to_level_sequence(r)
        layout += [i+1 for i in lay]
    return layout

def _level_sequence_to_list_repr(levelSeq):
    if len(levelSeq) == 0:
        return None
    branch_layouts = _branch_level_sequences(levelSeq)
    rep = tuple(_level_sequence_to_list_repr(lay) for lay in branch_layouts)
    return rep

def _branch_level_sequences(levelSeq):
    branch_layouts = []
    for i in levelSeq[1:]:
        if i == 1:
            branch_layouts.append([0])
        else:
            branch_layouts[-1].append(i - 1)
    return branch_layouts

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

# ##############################################
# ##############################################
#
# def _is_tree_like(obj):
#     return isinstance(obj, Tree) or isinstance(obj, Forest) or
#
# def _mul(obj1, obj2, applyReduction = True):
#     if not (isinstance):
#         if isinstance(obj2, int) or isinstance(obj2, float):
#             return obj1 * obj2
#         else:
#             return obj2.__mul__(obj1, applyReduction)
#     else:
#         return obj1.__mul__(obj2, applyReduction)
#
# def _add(obj1, obj2, applyReduction = True):
#     if isinstance(obj1, int) or isinstance(obj1, float):
#         if isinstance(obj2, int) or isinstance(obj2, float):
#             return obj1 + obj2
#         else:
#             return obj2.__add__(obj1, applyReduction)
#     else:
#         return obj1.__add__(obj2, applyReduction)
#
# def _sub(obj1, obj2, applyReduction = True):
#     if isinstance(obj1, int) or isinstance(obj1, float):
#         if isinstance(obj2, int) or isinstance(obj2, float):
#             return obj1 - obj2
#         else:
#             return obj2.__sub__(obj1, applyReduction)
#     else:
#         return obj1.__sub__(obj2, applyReduction)