import matplotlib.pyplot as plt
from .trees import *
from .utils import branch_layouts_

#use a layout paired with coords
def get_node_coords(layout, x = 0, y = 0, scale = 0.2):
    branch_gap = scale / 2

    if layout == []:
        return [], 0
    elif layout == [0]:
        return [(x,y)], branch_gap

    coords = [(x,y)]
    branch_layouts = branch_layouts_(layout)

    branch_coords = []
    branch_widths = []
    for branch in branch_layouts:
        c, w = get_node_coords(branch, x, y + 1, scale)
        branch_coords.append(c)
        branch_widths.append(w)

    width = sum(branch_widths) + (len(branch_widths) - 1) * branch_gap #leave gaps between branches

    pos = - width / 2
    for i in range(len(branch_coords)):
        branch_coords[i] = [(c[0] + pos + branch_widths[i] / 2, c[1]) for c in branch_coords[i]]
        pos += branch_widths[i] + branch_gap

    for c in branch_coords:
        coords += c

    return coords, width

def display_tree_(layout, coords, scale = 0.2):
    branch_gap = scale / 2

    if layout == []:
        return

    xroot, yroot = coords[0]

    plt.scatter([xroot], [yroot], c='k', marker='o', linewidth= scale / 2)

    branch_layouts = []
    branch_coords = []
    for idx, i in enumerate(layout[1:]):
        if i == 1:
            branch_layouts.append([0])
            branch_coords.append([coords[idx+1]])
        else:
            branch_layouts[-1].append(i - 1)
            branch_coords[-1].append(coords[idx+1])

    for lay, c in zip(branch_layouts, branch_coords):
        plt.plot([xroot, c[0][0]], [yroot, c[0][1]], color = "black")
        display_tree_(lay, c, scale)

def display(fs, scale = 0.2, figsize = (15,1),  fileName = None):
    tree_gap = scale / 4
    coeff_gap = scale / 2

    plt.figure(figsize = figsize)
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
            x += (len(str(c)) + 1) * coeff_gap
            for t in f.treeList:
                c, w = get_node_coords(t.layout(), x, 0, scale)
                c = [(c_[0] + w / 2, c_[1]) for c_ in c]
                display_tree_(t.layout(), c, scale)
                x += w + tree_gap
                if len(c) > 0:
                    h_ = max(c_[1] for c_ in c)
                    h = max(h, h_)
            x += coeff_gap / 2
            if i < len(fs.forestList) - 1:
                plt.text(x, y, "+" if fs.coeffList[i + 1] > 0 else "-")
                x += coeff_gap*2

    plt.xlim(- 1, 15)
    plt.ylim(-0.5, h + 0.5)
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.tight_layout()

    if fileName is not None:
        plt.savefig(fileName + ".png")

    plt.show()