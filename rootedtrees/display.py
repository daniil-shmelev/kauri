import matplotlib.pyplot as plt
from .trees import *
from .utils import _branch_level_sequences

#use a layout paired with coords
def _get_node_coords(layout, x = 0, y = 0, scale = 0.2):
    branch_gap = scale / 2

    if layout == []:
        return [], 0
    elif layout == [0]:
        return [(x,y)], branch_gap

    coords = [(x,y)]
    branch_layouts = _branch_level_sequences(layout)

    branch_coords = []
    branch_widths = []
    for branch in branch_layouts:
        c, w = _get_node_coords(branch, x, y + 1, scale)
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

def _display_tree(layout, coords, scale = 0.2):
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
        _display_tree(lay, c, scale)

def display(forest_sum, scale = 0.2, fig_size = (15, 1), file_name = None):
    """
    Plots a forest sum.

    :param forest_sum: Forest sum to plot
    :type forest_sum: ForestSum
    :param scale: scale of the plot (default = 0.2)
    :type scale: float
    :param fig_size: figure size (default = (15,1))
    :type fig_size: tuple
    :param file_name: If file_name is not None, will save the plot as a png file with the name file_name (default = None).
    :type file_name: string
    """
    tree_gap = scale / 4
    coeff_gap = scale / 2

    plt.figure(figsize = fig_size)
    if not isinstance(forest_sum, ForestSum):
        if isinstance(forest_sum, int) or isinstance(forest_sum, float):
            forest_sum = Tree(None) * forest_sum
        else:
            forest_sum = forest_sum.as_forest_sum()
    if forest_sum == ForestSum([]):
        plt.text(0, 0, str(0))
        h = 1
    else:
        x, y = 0, 0
        h = 0

        for f, c, i in zip(forest_sum.forest_list, forest_sum.coeff_list, range(len(forest_sum.coeff_list))):
            if i > 0:
                c = abs(c)
            plt.text(x, y, str(c))
            x += (len(str(c)) + 1) * coeff_gap
            for t in f.tree_list:
                c, w = _get_node_coords(t.level_sequence(), x, 0, scale)
                c = [(c_[0] + w / 2, c_[1]) for c_ in c]
                _display_tree(t.level_sequence(), c, scale)
                x += w + tree_gap
                if len(c) > 0:
                    h_ = max(c_[1] for c_ in c)
                    h = max(h, h_)
            x += coeff_gap / 2
            if i < len(forest_sum.forest_list) - 1:
                plt.text(x, y, "+" if forest_sum.coeff_list[i + 1] > 0 else "-")
                x += coeff_gap*2

    plt.xlim(- 1, 15)
    plt.ylim(-0.5, h + 0.5)
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.tight_layout()

    if file_name is not None:
        plt.savefig(file_name + ".png")

    plt.show()