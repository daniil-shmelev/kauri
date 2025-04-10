import plotly.graph_objects as go
from .trees import *
from .utils import _branch_level_sequences


def _get_node_coords(layout, x=0, y=0, scale=0.2):
    branch_gap = scale / 2
    if layout == []:
        return [], 0
    elif layout == [0]:
        return [(x, y)], branch_gap

    coords = [(x, y)]
    branch_layouts = _branch_level_sequences(layout)

    branch_coords = []
    branch_widths = []
    for branch in branch_layouts:
        c, w = _get_node_coords(branch, x, y + 1, scale)
        branch_coords.append(c)
        branch_widths.append(w)

    width = sum(branch_widths) + (len(branch_widths) - 1) * branch_gap
    pos = - width / 2
    for i in range(len(branch_coords)):
        branch_coords[i] = [(c[0] + pos + branch_widths[i] / 2, c[1]) for c in branch_coords[i]]
        pos += branch_widths[i] + branch_gap

    for c in branch_coords:
        coords += c

    return coords, width


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
            line=dict(color='black'),
            showlegend=False,
            hoverinfo='skip'
        ))
        traces.extend(_get_tree_traces(lay, c, scale))

    return traces


def display(forest_sum, scale=0.7, fig_size=(1500, 50), file_name=None):
    tree_gap = scale / 2
    coeff_gap = scale / 2

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
            x=[x], y=[y], text=[str(c)], mode='text',
            showlegend=False
        ))

        x += (len(str(c)) + 1) * coeff_gap

        for t in f.tree_list:
            level_seq = t.level_sequence()
            c_, w = _get_node_coords(level_seq, x, 0, scale)
            c_ = [(cx + w / 2, cy) for cx, cy in c_]

            # Nodes
            traces.append(go.Scatter(
                x=[p[0] for p in c_],
                y=[p[1] for p in c_],
                mode='markers',
                marker=dict(color='black', size=6),
                showlegend=False,
                hoverinfo='skip'
            ))

            # Edges
            traces.extend(_get_tree_traces(level_seq, c_, scale))

            x += w + tree_gap
            if len(c_) > 0:
                h_ = max(cy for _, cy in c_)
                h = max(h, h_)
        x += coeff_gap / 2

        if i < len(forest_sum.term_list) - 1:
            op = "+" if forest_sum.term_list[i + 1][0] > 0 else "-"
            traces.append(go.Scatter(
                x=[x], y=[y], text=[op], mode='text',
                showlegend=False
            ))
            x += coeff_gap * 2

    fig = go.Figure(traces)
    extra_padding = 1 if h == 1 else 0
    fig.update_layout(template="simple_white")
    fig.update_layout(
        width=fig_size[0],
        height=fig_size[1],
        xaxis=dict(showgrid=False, zeroline=False, visible=False, range=[-10, 100]),
        yaxis=dict(showgrid=False, zeroline=False, visible=False, range=[-0.5, h + extra_padding + 0.5]),
        margin=dict(l=0, r=0, t=0, b=0)
    )

    if file_name:
        fig.write_image(file_name + ".png")

    fig.show(config={
        "displayModeBar": False,
        "staticPlot": True
    })
