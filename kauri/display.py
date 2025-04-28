"""
Functions for plotting Tree, Forest, ForestSum and TensorProductSum objects.
"""
#TODO add plotting of TensorSum to display
#TODO: simplify
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from .trees import Tree, ForestSum
from .utils import _branch_level_sequences, _str

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

        for t in f.tree_list:
            level_seq = t.level_sequence()
            c_, w = _get_node_coords(level_seq, x, 0, scale)
            c_ = [(cx + w / 2, cy) for cx, cy in c_]

            # Nodes
            traces.append(go.Scatter(
                x=[p[0] for p in c_],
                y=[p[1] for p in c_],
                mode='markers',
                marker={"color" : 'black', "size" : 6},
                showlegend=False,
                hoverinfo='skip'
            ))

            # Edges
            traces.extend(_get_tree_traces(level_seq, c_, scale))

            x += w + gap
            if len(c_) > 0:
                h_ = max(cy for _, cy in c_)
                h = max(h, h_)
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



###############################################################
#Matplotlib
###############################################################

def _display_tree(layout, coords, scale = 0.2):

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
            for t in f.tree_list:
                c, w = _get_node_coords(t.level_sequence(), x, 0, scale)
                c = [(c_[0] + w / 2, c_[1]) for c_ in c]
                _display_tree(t.level_sequence(), c, scale)
                x += w + tree_gap
                if len(c) > 0:
                    h_ = max(c_[1] for c_ in c)
                    h = max(h, h_)
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


###############################################################
#Display
###############################################################

def display(forest_sum : ForestSum, *, #TODO: change to Tree, Forest, ForestSum or TensorProductSum
            scale : float = None,
            fig_size : tuple = None,
            file_name : str = None,
            use_plt : bool = True,
            rationalise : bool = False) -> None:
    """ #TODO
    Plots a forest sum.

    :param forest_sum: Forest sum to plot
    :type forest_sum: ForestSum
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
    if use_plt:
        if scale is None:
            scale = 0.2
        if fig_size is None:
            fig_size = (15,1)
        _display_plt(forest_sum, scale, fig_size, file_name, rationalise)
    else:
        if scale is None:
            scale = 0.7
        if fig_size is None:
            fig_size = (1500, 50)
        _display_plotly(forest_sum, scale, fig_size, file_name, rationalise)
