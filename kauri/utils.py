"""
Back-end utility functions
"""
import math
from functools import cache
import sympy as sp

def _to_tuple(obj):
    if isinstance(obj, list):
        return tuple(_to_tuple(el) for el in obj)
    return obj

def _to_list(obj):
    if isinstance(obj, tuple):
        return list(_to_list(el) for el in obj)
    return obj

def _check_valid(obj):
    if obj == tuple() or obj == []:
        return True
    for el in obj[:-1]:
        if not isinstance(el, (tuple, list)) or not _check_valid(el):
            return False

    if not isinstance(obj[-1], (int, tuple, list)):
        return False

    if isinstance(obj[-1], int) and obj[-1] < 0:
        return False
    return True

def _to_labelled_tuple(obj):
    if obj == tuple() or obj == []:
        return (0,)
    out = tuple(_to_labelled_tuple(el) for el in obj[:-1])
    if isinstance(obj[-1], int):
        out += (obj[-1],)
    else:
        out += (_to_labelled_tuple(obj[-1]), 0)
    return out

def _to_unlabelled_tuple(obj):
    return tuple(_to_unlabelled_tuple(el) for el in obj[:-1])

def _get_max_color(obj):
    if obj is None:
        return 0
    out = 0
    for el in obj[:-1]:
        out = max(out, _get_max_color(el))
    out = max(out, obj[-1])
    return out

@cache
def _nodes(rep):
    if rep is None:
        return 0
    if rep == tuple():
        return 1
    return 1 + sum(_nodes(r) for r in rep)

@cache
def _height(rep):
    if rep is None:
        return 0
    if rep == tuple():
        return 1
    return 1 + max(_height(r) for r in rep)

@cache
def _factorial(rep):
    if rep is None:
        return 1, 0
    if rep == tuple():
        return 1, 1

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
        if r in rep_dict:
            rep_dict[r] += 1
        else:
            rep_dict[r] = 1
            unique_rep.append(r)

    out = 1
    for r in unique_rep:
        k = rep_dict[r]
        out *= math.factorial(k) * (_sigma(r) ** k)
    return out

class LabelledReprComparison:
    def __init__(self, rep):
        self.rep = rep
        self.unlabelled_rep = _to_unlabelled_tuple(rep)

    def __lt__(self, other):
        if self.unlabelled_rep == other.unlabelled_rep:
            return self.rep < other.rep
        return self.unlabelled_rep < other.unlabelled_rep

@cache
def _sorted_list_repr(rep):
    if rep is None:
        return None
    if len(rep) == 1:
        return rep

    *children, label = rep
    sorted_children = tuple(sorted((_sorted_list_repr(child) for child in children), key = LabelledReprComparison, reverse=True))
    return sorted_children + (label,)

@cache
def _list_repr_to_level_sequence(rep):
    if rep is None:
        return []

    layout = [0]
    for r in rep:
        lay = _list_repr_to_level_sequence(r)
        layout += [i+1 for i in lay]
    return layout

def _level_sequence_to_list_repr(level_seq):
    if len(level_seq) == 0:
        return None
    branch_layouts = _branch_level_sequences(level_seq)
    rep = tuple(_level_sequence_to_list_repr(lay) for lay in branch_layouts)
    return rep

def _branch_level_sequences(level_seq):
    branch_layouts = []
    for i in level_seq[1:]:
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

def _rationalise(c, tol = 1e-10):
    return str(sp.nsimplify(c, tolerance=tol, rational = True))

def _str(c, rationalise = False, tol = 1e-10):
    if rationalise:
        return _rationalise(c, tol)
    return str(c)
