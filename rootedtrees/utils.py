def _nodes(rep):
    if rep is None:
        return 0
    elif rep == []:
        return 1
    else:
        return 1 + sum(_nodes(r) for r in rep)

def _factorial(rep):
    if rep == None:
        return 1, 0
    if rep == []:
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

def _sorted_list_repr(rep):
    if rep is None:
        return None
    elif rep == []:
        return []
    else:
        return sorted([_sorted_list_repr(r) for r in rep], reverse = True)

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
    rep = [_level_sequence_to_list_repr(lay) for lay in branch_layouts]
    return rep

def _branch_level_sequences(levelSeq):
    branch_layouts = []
    for i in levelSeq[1:]:
        if i == 1:
            branch_layouts.append([0])
        else:
            branch_layouts[-1].append(i - 1)
    return branch_layouts

##############################################
##############################################


def _mul(obj1, obj2, applyReduction = True):
    if isinstance(obj1, int) or isinstance(obj1, float):
        if isinstance(obj2, int) or isinstance(obj2, float):
            return obj1 * obj2
        else:
            return obj2.__mul__(obj1, applyReduction)
    else:
        return obj1.__mul__(obj2, applyReduction)

def _add(obj1, obj2, applyReduction = True):
    if isinstance(obj1, int) or isinstance(obj1, float):
        if isinstance(obj2, int) or isinstance(obj2, float):
            return obj1 + obj2
        else:
            return obj2.__add__(obj1, applyReduction)
    else:
        return obj1.__add__(obj2, applyReduction)

def _sub(obj1, obj2, applyReduction = True):
    if isinstance(obj1, int) or isinstance(obj1, float):
        if isinstance(obj2, int) or isinstance(obj2, float):
            return obj1 - obj2
        else:
            return obj2.__sub__(obj1, applyReduction)
    else:
        return obj1.__sub__(obj2, applyReduction)