"""
The display function plots objects of type :class:`Tree`, :class:`Forest`, :class:`ForestSum`
or :class:`TensorProductSum`.
"""
from typing import Union

import matplotlib.pyplot as plt
import plotly.graph_objects as go

from .trees import Tree, ForestSum, Forest, TensorProductSum
from .utils import _branch_level_sequences, _str

EMPTY_FONTSIZE = 10
TENSOR_FONTSIZE = 14

COLORS = ['black',
          'firebrick',
          'mediumblue',
          'forestgreen',
          'rebeccapurple',
          'darkorange',
          'grey',
          'dodgerblue',
          'deeppink']

def _get_node_coords(layout, x=0, y=0, scale=0.2):
    gap = scale / 2
    if layout == []:
        return [], 0
    if layout == [0]:
        return [(x, y)], gap

    coords = [(x, y)]
    branch_layouts = _branch_level_sequences(layout)

    branch_coords = []
    branch_widths = []
    for branch in branch_layouts:
        c, w = _get_node_coords(branch, x, y + 1, scale)
        branch_coords.append(c)
        branch_widths.append(w)

    width = sum(branch_widths) + (len(branch_widths) - 1) * gap
    pos = - width / 2
    for i in range(len(branch_coords)):
        branch_coords[i] = [(c[0] + pos + branch_widths[i] / 2, c[1]) for c in branch_coords[i]]
        pos += branch_widths[i] + gap

    for c in branch_coords:
        coords += c

    return coords, width

###############################################################
#Plotly
###############################################################

def _get_tree_traces(layout, coords, scale=0.2):
    traces = []
    if layout == []:
        return traces

    xroot, yroot = coords[0]

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
        # Add edge line
        traces.append(go.Scatter(
            x=[xroot, c[0][0]],
            y=[yroot, c[0][1]],
            mode='lines',
            line={"color" : 'black'},
            showlegend=False,
            hoverinfo='skip'
        ))
        traces.extend(_get_tree_traces(lay, c, scale))

    return traces

def _display_plotly_forest(f, x, y, h, scale, traces, gap, empty = False):

    if empty and f == Tree(None):
        traces.append(go.Scatter(
            x=[x], y=[y], text=["\u2205"], mode='text',
            showlegend=False
        ))
        x += 2*gap
        return x, h, traces

    for t in f.tree_list:
        level_seq = t.level_sequence()
        color_seq = t.color_sequence()
        c_, w = _get_node_coords(level_seq, x, 0, scale)
        c_ = [(cx + w / 2, cy) for cx, cy in c_]

        # Edges
        traces.extend(_get_tree_traces(level_seq, c_, scale))

        # Nodes
        traces.append(go.Scatter(
            x=[p[0] for p in c_],
            y=[p[1] for p in c_],
            mode='markers',
            marker={'color' : [COLORS[i] for i in color_seq]},
            showlegend=False,
            hoverinfo='skip'
        ))

        x += w + gap
        if len(c_) > 0:
            h_ = max(cy for _, cy in c_)
            h = max(h, h_)

    return x, h, traces

def _display_plotly(forest_sum,
                    scale=0.7,
                    fig_size=(1500, 50),
                    file_name=None,
                    rationalise = True):
    gap = scale / 2
    traces = []

    if not isinstance(forest_sum, ForestSum):
        if isinstance(forest_sum, (int, float)):
            forest_sum = Tree(None) * forest_sum
        else:
            forest_sum = forest_sum.as_forest_sum()

    x, y = 0, 0
    h = 0

    for i, (c, f) in enumerate(forest_sum.term_list):
        if i > 0:
            c = abs(c)

        # Add coefficient as text
        traces.append(go.Scatter(
            x=[x], y=[y], text=[_str(c, rationalise)], mode='text',
            showlegend=False
        ))

        x += (len(_str(c, rationalise)) + 1) * gap

        x, h, traces = _display_plotly_forest(f, x, y, h, scale, traces, gap)
        x += gap / 2

        if i < len(forest_sum.term_list) - 1:
            op = "+" if forest_sum.term_list[i + 1][0] > 0 else "-"
            traces.append(go.Scatter(
                x=[x], y=[y], text=[op], mode='text',
                showlegend=False
            ))
            x += gap * 2

    fig = go.Figure(traces)
    extra_padding = 1 if h == 1 else 0
    fig.update_layout(template="simple_white")
    fig.update_layout(
        width=fig_size[0],
        height=fig_size[1],
        xaxis={"showgrid" : False,
               "zeroline" : False,
               "visible" : False,
               "range" : [-10, 100]},
        yaxis={"showgrid" : False,
               "zeroline" : False,
               "visible" : False,
               "range" : [-0.5, h + extra_padding + 0.5]},
        margin={"l": 0, "r": 0, "t": 0, "b": 0}
    )

    if file_name:
        fig.write_image(file_name + ".png")

    fig.show(config={
        "displayModeBar": False,
        "staticPlot": True
    })

def _display_tensor_plotly(tensor_sum,
                    scale=0.7,
                    fig_size=(1500, 50),
                    file_name=None,
                    rationalise = True):
    gap = scale / 2
    traces = []

    x, y = 0, 0
    h = 0

    for i, (c, f1, f2) in enumerate(tensor_sum.term_list):
        if i > 0:
            c = abs(c)

        # Add coefficient as text
        traces.append(go.Scatter(
            x=[x], y=[y], text=[_str(c, rationalise)], mode='text',
            showlegend=False
        ))

        x += (len(_str(c, rationalise)) + 1) * gap

        x, h, traces = _display_plotly_forest(f1, x, y, h, scale, traces, gap, True)
        x += 1.5 * gap
        traces.append(go.Scatter(
            x=[x], y=[y], text=["\u2297"], mode='text',
            showlegend=False
        ))
        x += 2.5 * gap
        x, h, traces = _display_plotly_forest(f2, x, y, h, scale, traces, gap, True)
        x += 1.5 * gap

        if i < len(tensor_sum.term_list) - 1:
            op = "+" if tensor_sum.term_list[i + 1][0] > 0 else "-"
            traces.append(go.Scatter(
                x=[x], y=[y], text=[op], mode='text',
                showlegend=False
            ))
            x += gap * 2

    fig = go.Figure(traces)
    extra_padding = 1 if h == 1 else 0
    fig.update_layout(template="simple_white")
    fig.update_layout(
        width=fig_size[0],
        height=fig_size[1],
        xaxis={"showgrid" : False,
               "zeroline" : False,
               "visible" : False,
               "range" : [-10, 100]},
        yaxis={"showgrid" : False,
               "zeroline" : False,
               "visible" : False,
               "range" : [-0.5, h + extra_padding + 0.5]},
        margin={"l": 0, "r": 0, "t": 0, "b": 0}
    )

    if file_name:
        fig.write_image(file_name + ".png")

    fig.show(config={
        "displayModeBar": False,
        "staticPlot": True
    })



###############################################################
#Matplotlib
###############################################################

def _display_tree(layout, color_sequence, coords, scale = 0.2):

    if layout == []:
        return

    xroot, yroot = coords[0]

    plt.scatter([xroot], [yroot], marker='o', linewidth= scale / 2, color = COLORS[color_sequence[0]], zorder = 1)

    branch_layouts = []
    branch_coords = []
    branch_colors = []
    for idx, i in enumerate(layout[1:]):
        if i == 1:
            branch_layouts.append([0])
            branch_coords.append([coords[idx+1]])
            branch_colors.append([color_sequence[idx + 1]])
        else:
            branch_layouts[-1].append(i - 1)
            branch_coords[-1].append(coords[idx+1])
            branch_colors[-1].append(color_sequence[idx + 1])

    for lay, c, cols in zip(branch_layouts, branch_coords, branch_colors):
        plt.plot([xroot, c[0][0]], [yroot, c[0][1]], color = 'black', zorder = -1)
        _display_tree(lay, cols, c, scale)

def _display_forest(f, x, y, scale, tree_gap, h, empty = False):

    if empty and f == Tree(None):
        plt.text(x, y, "\u2205", fontsize=EMPTY_FONTSIZE)
        x += 4 * tree_gap
        return x, h

    for t in f.tree_list:
        c, w = _get_node_coords(t.level_sequence(), x, 0, scale)
        c = [(c_[0] + w / 2, c_[1]) for c_ in c]
        _display_tree(t.level_sequence(), t.color_sequence(), c, scale)
        x += w + tree_gap
        if len(c) > 0:
            h_ = max(c_[1] for c_ in c)
            h = max(h, h_)
    return x, h

def _display_plt(forest_sum,
                 scale = 0.2,
                 fig_size = (15, 1),
                 file_name = None,
                 rationalise = True):
    tree_gap = scale / 4
    coeff_gap = scale / 2

    plt.figure(figsize = fig_size)
    if not isinstance(forest_sum, ForestSum):
        if isinstance(forest_sum, (int, float)):
            forest_sum = Tree(None) * forest_sum
        else:
            forest_sum = forest_sum.as_forest_sum()
    if forest_sum == ForestSum([]):
        plt.text(0, 0, str(0))
        h = 1
    else:
        x, y = 0, 0
        h = 0

        for i, (c, f) in enumerate(forest_sum.term_list):
            if i > 0:
                c = abs(c)
            plt.text(x, y, _str(c, rationalise))
            x += (len(_str(c, rationalise)) + 1) * coeff_gap

            x, h = _display_forest(f, x, y, scale, tree_gap, h)

            x += coeff_gap / 2
            if i < len(forest_sum.term_list) - 1:
                plt.text(x, y, "+" if forest_sum.term_list[i + 1][0] > 0 else "-")
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

def _display_tensor_plt(tensor_sum,
                 scale = 0.2,
                 fig_size = (15, 1),
                 file_name = None,
                 rationalise = True):
    tree_gap = scale / 4
    coeff_gap = scale / 2

    plt.figure(figsize=fig_size)
    x, y = 0, 0
    h = 0

    for i, (c, f1, f2) in enumerate(tensor_sum.term_list):
        if i > 0:
            c = abs(c)
        plt.text(x, y, _str(c, rationalise))
        x += (len(_str(c, rationalise)) + 1) * coeff_gap

        x, h = _display_forest(f1, x, y, scale, tree_gap, h, True)
        x += 0.5 * coeff_gap
        plt.text(x, y, "\u2297", fontsize=TENSOR_FONTSIZE)
        x += 2.5 * coeff_gap
        x, h = _display_forest(f2, x, y, scale, tree_gap, h, True)

        x += coeff_gap / 2
        if i < len(tensor_sum.term_list) - 1:
            plt.text(x, y, "+" if tensor_sum.term_list[i + 1][0] > 0 else "-")
            x += coeff_gap * 2

    plt.xlim(- 1, 15)
    plt.ylim(-0.5, h + 0.5)
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.tight_layout()

    if file_name is not None:
        plt.savefig(file_name + ".png")

    plt.show()


###############################################################
#Display
###############################################################

def display(obj : Union[Tree, Forest, ForestSum, TensorProductSum],
            *,
            scale : float = None,
            fig_size : tuple = None,
            file_name : str = None,
            use_plt : bool = True,
            rationalise : bool = False) -> None:
    """
    Plots a Tree, Forest, ForestSum or TensorProductSum.

    :param obj: Object to plot
    :type obj: Tree | Forest | ForestSum | TensorProductSum
    :param scale: scale of the plot (default = 0.2 if use_plt is True otherwise 0.7)
    :type scale: float
    :param fig_size: figure size (default = (15,1) if use_plt is True otherwise (1500,50))
    :type fig_size: tuple
    :param file_name: If file_name is not None, will save the plot as a png file with the
        name file_name (default = None).
    :type file_name: string
    :param use_plt: If True uses matplotlib (default), otherwise uses Plotly.
        Plotly is quicker, but results in larger file sizes when used in notebooks.
    :type use_plt: bool
    """
    if not isinstance(obj, (Tree, Forest, ForestSum, TensorProductSum)):
        raise TypeError("Cannot display object of type " + str(type(obj)) + ". Object must be Tree, Forest, ForestSum or TensorProductSum.")

    if obj.colors() > 9:
        raise ValueError("Cannot display labelled trees with over 10 different colors.")

    if use_plt:
        if scale is None:
            scale = 0.2
        if fig_size is None:
            fig_size = (15,1)

        if isinstance(obj, TensorProductSum):
            _display_tensor_plt(obj, scale, fig_size, file_name, rationalise)
        else:
            _display_plt(obj, scale, fig_size, file_name, rationalise)
    else:
        if scale is None:
            scale = 0.7
        if fig_size is None:
            fig_size = (1500, 50)

        if isinstance(obj, TensorProductSum):
            _display_tensor_plotly(obj, scale, fig_size, file_name, rationalise)
        else:
            _display_plotly(obj, scale, fig_size, file_name, rationalise)
