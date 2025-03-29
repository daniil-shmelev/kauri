def numNodes_(rep):
    if rep is None:
        return 0
    elif rep == []:
        return 1
    else:
        return 1 + sum(numNodes_(r) for r in rep)

def factorial_(rep):
    if rep == None:
        return 1, 0
    if rep == []:
        return 1, 1
    else:
        f = 1
        n = 1
        for r in rep:
            res = factorial_(r)
            f *= res[0]
            n += res[1]
        f *= n
        return f, n

def sortedListRepr_(rep):
    if rep is None:
        return None
    elif rep == []:
        return []
    else:
        return sorted([sortedListRepr_(r) for r in rep], reverse = True)

def listReprToLayout_(rep):
    if rep is None:
        return []

    layout = [0]
    for r in rep:
        lay = listReprToLayout_(r)
        layout += [i+1 for i in lay]
    return layout

def layoutToListRepr_(layout):
    if len(layout) == 0:
        return None

    branch_layouts = []
    for i in layout[1:]:
        if i == 1:
            branch_layouts.append([0])
        else:
            branch_layouts[-1].append(i-1)

    rep = [layoutToListRepr_(lay) for lay in branch_layouts]
    return rep

##############################################
##############################################


def mul_(obj1, obj2, applyReduction = True):
    if isinstance(obj1, int) or isinstance(obj1, float):
        if isinstance(obj2, int) or isinstance(obj2, float):
            return obj1 * obj2
        else:
            return obj2.__mul__(obj1, applyReduction)
    else:
        return obj1.__mul__(obj2, applyReduction)

def add_(obj1, obj2, applyReduction = True):
    if isinstance(obj1, int) or isinstance(obj1, float):
        if isinstance(obj2, int) or isinstance(obj2, float):
            return obj1 + obj2
        else:
            return obj2.__add__(obj1, applyReduction)
    else:
        return obj1.__add__(obj2, applyReduction)

def sub_(obj1, obj2, applyReduction = True):
    if isinstance(obj1, int) or isinstance(obj1, float):
        if isinstance(obj2, int) or isinstance(obj2, float):
            return obj1 - obj2
        else:
            return obj2.__sub__(obj1, applyReduction)
    else:
        return obj1.__sub__(obj2, applyReduction)