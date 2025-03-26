import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (15,1)
from trees import *

scale = 0.2


def display_tree_(t, x):
    if t == Tree(None):
        return
    display_subtree(t, x, 0)


def display_subtree(t, x, y):
    plt.scatter([x], [y], c='k', marker='o', linewidth=scale / 2)

    subtrees = t.unjoin()
    numSubtrees = subtrees.len()

    for i, subtr in enumerate(t.unjoin().treeList):
        display_subtree(subtr, x + scale * (i - 0.5 * (numSubtrees - 1)), y + 1)
        plt.plot([x, x + scale * (i - 0.5 * (numSubtrees - 1))], [y, y + 1], color="black")


def tree_width(t, x):
    if t == Tree(None):
        return 0, 0, 0
    height, xleft, xright = subtree_width(t, x, 0, 0, 0, 0)
    return height + 1, -xleft - scale / 2, xright + scale / 2


def subtree_width(t, x, y, h, xleft, xright):
    subtrees = t.unjoin()
    numSubtrees = subtrees.len()

    new_h = h
    xleft = min(xleft, x + scale * (numSubtrees - 0.5 * (numSubtrees - 1)))
    xright = max(xright, x + scale * (0.5 * (numSubtrees - 1)))

    for i, subtr in enumerate(t.unjoin().treeList):
        new_h, xleft_, xright_ = subtree_width(subtr, x + scale * (i - 0.5 * (numSubtrees - 1)), y + 1, h + 1, xleft,
                                               xright)
    return new_h, xleft, xright


def display(fs):
    if not isinstance(fs, ForestSum):
        fs = fs.asForestSum()    
    if fs == ForestSum([]):
        plt.text(0, 0, str(0))
        h = 1
    else:
        x, y = 0, 0
        h = 0

        for f, c, i in zip(fs.forestList, fs.coeffList, range(len(fs.coeffList))):
            if i > 0:
                c = abs(c)
            plt.text(x, y, str(c))
            x += (len(str(c)) + 1) * scale / 2
            for t in f.treeList:
                h1, xl, xr = tree_width(t, 0)
                x -= xl
                display_tree_(t, x)
                x += xr
                h = max(h, h1)
            x += scale / 2
            if i < len(fs.forestList) - 1:
                plt.text(x, y, "+" if fs.coeffList[i + 1] > 0 else "-")
                x += scale

    plt.xlim(- 1, 15)
    plt.ylim(-0.5, h + 0.5)
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.tight_layout()
    plt.show()