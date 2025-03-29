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
        return sorted([sortedListRepr_(r) for r in rep])