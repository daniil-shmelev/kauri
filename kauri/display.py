# Copyright 2025 Daniil Shmelev
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

"""
SVG rendering for Tree, Forest, ForestSum and TensorProductSum objects.
"""
import warnings
from typing import Union

from .trees import (Tree, ForestSum, Forest, TensorProductSum,
                    PlanarTree, NoncommutativeForest, _is_scalar)
from .utils import _branch_level_sequences, _str

# ── Configuration constants ──────────────────────────────────────────────
NODE_RADIUS = 3.5
EDGE_WIDTH = 1.2
LEVEL_SPACING = 20
SIBLING_GAP = 10
TREE_GAP = 12
TERM_GAP = 18
COEFF_GAP = 5
TENSOR_GAP = 10
FONT_SIZE = 11
PADDING = 8
CHAR_WIDTH_FACTOR = 0.6   # estimated character width as fraction of font size
NODE_STROKE_WIDTH = 0.8

COLORS = ['black',
          'firebrick',
          'mediumblue',
          'forestgreen',
          'rebeccapurple',
          'darkorange',
          'grey',
          'dodgerblue',
          'deeppink']


# ── Helpers ──────────────────────────────────────────────────────────────

def _shift_items_x(items, offset):
    """Shift all x-coordinates in render items by offset."""
    shifted = []
    for item in items:
        kind = item[0]
        if kind == 'node':
            shifted.append(('node', item[1] + offset, item[2], item[3]))
        elif kind == 'edge':
            shifted.append(('edge', item[1] + offset, item[2],
                            item[3] + offset, item[4]))
        else:
            shifted.append(item)
    return shifted


# ── Layer 1: Layout ──────────────────────────────────────────────────────

def _layout_tree(level_seq, color_seq, x_center, y_base, scale):
    """Recursively compute layout items for a single tree.

    Returns (items, width, height) where items are render primitives and
    y grows *upward* (root at bottom).
    """
    gap = SIBLING_GAP * scale
    level_sp = LEVEL_SPACING * scale

    if level_seq == []:
        return [], 0, 0
    if level_seq == [0]:
        return [('node', x_center, y_base, color_seq[0])], gap, level_sp

    items = []
    branches = _branch_level_sequences(level_seq)

    # Split color_seq into per-branch sequences
    branch_colors = []
    idx = 1
    for branch in branches:
        branch_colors.append(color_seq[idx:idx + len(branch)])
        idx += len(branch)

    # Layout each branch
    branch_items = []
    branch_widths = []
    branch_heights = []
    for branch, bcols in zip(branches, branch_colors):
        b_items, b_w, b_h = _layout_tree(branch, bcols, 0, y_base + level_sp, scale)
        branch_items.append(b_items)
        branch_widths.append(b_w)
        branch_heights.append(b_h)

    total_width = sum(branch_widths) + (len(branch_widths) - 1) * gap
    total_width = max(total_width, gap)

    # Position branches left-to-right centred on x_center
    pos = x_center - total_width / 2
    for i in range(len(branch_items)):
        offset_x = pos + branch_widths[i] / 2
        items.extend(_shift_items_x(branch_items[i], offset_x))
        # Edge from root to branch root (branch root is always at local
        # (0, y_base+level_sp), shifted by offset_x)
        if branch_items[i]:
            items.append(('edge', x_center, y_base,
                          offset_x, y_base + level_sp))
        pos += branch_widths[i] + gap

    # Add root node
    items.append(('node', x_center, y_base, color_seq[0]))

    max_height = level_sp
    if branch_heights:
        max_height = max(branch_heights) + level_sp

    return items, total_width, max_height


def _layout_forest(forest, x_start, y_base, scale, show_empty=False):
    """Lay out a forest (sequence of trees) left-to-right.

    Returns (items, width, height).
    """
    gap = TREE_GAP * scale
    cw = CHAR_WIDTH_FACTOR

    if show_empty and forest == Forest((Tree(None),)):
        fs = FONT_SIZE * scale
        items = [('text', x_start + fs * cw / 2, y_base, '\u2205', fs)]
        return items, fs * cw, LEVEL_SPACING * scale

    items = []
    x = x_start
    max_height = 0

    for t in forest.tree_list:
        level_seq = t.level_sequence()
        color_seq = t.color_sequence()
        if level_seq == []:
            continue
        t_items, t_w, t_h = _layout_tree(level_seq, color_seq, 0, y_base, scale)
        items.extend(_shift_items_x(t_items, x + t_w / 2))
        x += t_w + gap
        max_height = max(max_height, t_h)

    width = max(x - x_start - gap, 0) if items else 0
    if max_height == 0:
        max_height = LEVEL_SPACING * scale
    return items, width, max_height


def _format_coeff(c, is_first, rationalise):
    """Format coefficient for display, suppressing trivial '1' coefficients."""
    abs_c = abs(c)
    if abs_c == 1:
        if is_first:
            return '' if c >= 0 else '\u2212'
        else:
            return ''  # sign handled by +/- operator
    else:
        return _str(abs_c, rationalise)


def _layout_coeff_op(items, x, c, is_first, scale, rationalise):
    """Lay out the operator (+/−) and coefficient for a single term.

    Returns the new x position after emitting any text items.
    """
    term_gap = TERM_GAP * scale
    coeff_gap = COEFF_GAP * scale
    fs = FONT_SIZE * scale
    cw = CHAR_WIDTH_FACTOR

    if not is_first:
        op = '+' if c >= 0 else '\u2212'
        items.append(('text', x + term_gap / 2, 0, op, fs))
        x += term_gap

    coeff_str = _format_coeff(c, is_first, rationalise)
    if coeff_str:
        items.append(('text', x + len(coeff_str) * fs * cw / 2, 0, coeff_str, fs))
        x += len(coeff_str) * fs * cw + coeff_gap

    return x


def _layout_forest_sum(forest_sum, scale, rationalise=False):
    """Lay out a ForestSum left-to-right.

    Returns (items, total_width, total_height).
    """
    if not isinstance(forest_sum, ForestSum):
        if _is_scalar(forest_sum):
            forest_sum = Tree(None) * forest_sum
        else:
            forest_sum = forest_sum.as_forest_sum()

    if len(forest_sum.term_list) == 0:
        fs = FONT_SIZE * scale
        items = [('text', 0, 0, '0', fs)]
        return items, fs * 0.5, fs

    items = []
    x = 0
    max_height = 0
    coeff_gap = COEFF_GAP * scale

    for i, (c, f) in enumerate(forest_sum.term_list):
        x = _layout_coeff_op(items, x, c, i == 0, scale, rationalise)

        f_items, f_w, f_h = _layout_forest(f, x, 0, scale)
        items.extend(f_items)
        x += f_w + coeff_gap / 2
        max_height = max(max_height, f_h)

    if max_height == 0:
        max_height = LEVEL_SPACING * scale

    return items, x, max_height


def _layout_tensor_sum(tensor_sum, scale, rationalise=False):
    """Lay out a TensorProductSum left-to-right.

    Returns (items, total_width, total_height).
    """
    items = []
    x = 0
    max_height = 0
    coeff_gap = COEFF_GAP * scale
    tensor_gap = TENSOR_GAP * scale
    fs = FONT_SIZE * scale
    cw = CHAR_WIDTH_FACTOR

    for i, (c, f1, f2) in enumerate(tensor_sum.term_list):
        x = _layout_coeff_op(items, x, c, i == 0, scale, rationalise)

        # Left forest
        f1_items, f1_w, f1_h = _layout_forest(f1, x, 0, scale, show_empty=True)
        items.extend(f1_items)
        x += f1_w + tensor_gap / 2

        # Tensor symbol
        items.append(('text', x + fs * cw / 2, 0, '\u2297', fs))
        x += fs * cw + tensor_gap / 2

        # Right forest
        f2_items, f2_w, f2_h = _layout_forest(f2, x, 0, scale, show_empty=True)
        items.extend(f2_items)
        x += f2_w + coeff_gap / 2

        max_height = max(max_height, f1_h, f2_h)

    if max_height == 0:
        max_height = LEVEL_SPACING * scale

    return items, x, max_height


# ── Layer 2: SVG Rendering ───────────────────────────────────────────────

def _render_svg(items, width, height, scale):
    """Convert render items into an SVG string.

    Layout uses y-up coordinates (root at bottom).  SVG uses y-down,
    so we flip: svg_y = height - layout_y.
    """
    pad = PADDING * scale
    svg_w = width + 2 * pad
    svg_h = height + 2 * pad
    r = NODE_RADIUS * scale
    ew = EDGE_WIDTH * scale
    nsw = NODE_STROKE_WIDTH

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{svg_w:.1f}" height="{svg_h:.1f}" '
        f'viewBox="0 0 {svg_w:.1f} {svg_h:.1f}">'
    ]

    # Render order: edges (behind), then nodes, then text (on top)
    _ORDER = {'edge': 0, 'node': 1, 'text': 2}
    for item in sorted(items, key=lambda it: _ORDER[it[0]]):
        kind = item[0]
        if kind == 'edge':
            _, x1, y1, x2, y2 = item
            parts.append(
                f'<line x1="{x1 + pad:.1f}" y1="{height - y1 + pad:.1f}" '
                f'x2="{x2 + pad:.1f}" y2="{height - y2 + pad:.1f}" '
                f'stroke="black" stroke-width="{ew:.1f}" '
                f'stroke-linecap="round" />'
            )
        elif kind == 'node':
            _, x, y, color_idx = item
            sx, sy = x + pad, height - y + pad
            color = COLORS[color_idx] if color_idx < len(COLORS) else 'black'
            if color_idx > 0:
                parts.append(
                    f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="{r:.1f}" '
                    f'fill="{color}" stroke="black" stroke-width="{nsw}" />'
                )
            else:
                parts.append(
                    f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="{r:.1f}" '
                    f'fill="black" />'
                )
        else:
            _, x, y, text, font_size = item
            parts.append(
                f'<text x="{x + pad:.1f}" y="{height - y + pad:.1f}" '
                f'font-size="{font_size:.1f}" font-family="sans-serif" '
                f'text-anchor="middle" dominant-baseline="central">'
                f'{text}</text>'
            )

    parts.append('</svg>')
    return '\n'.join(parts)


# ── Layer 3: Orchestration ───────────────────────────────────────────────

def _to_svg(obj, scale=1.0, rationalise=False):
    """Generate an SVG string for a tree-algebra object."""
    if isinstance(obj, TensorProductSum):
        items, w, h = _layout_tensor_sum(obj, scale, rationalise)
    elif isinstance(obj, ForestSum):
        items, w, h = _layout_forest_sum(obj, scale, rationalise)
    elif isinstance(obj, (Forest, NoncommutativeForest)):
        items, w, h = _layout_forest(obj, 0, 0, scale)
    elif isinstance(obj, (Tree, PlanarTree)):
        if obj.list_repr is None:
            fs = FONT_SIZE * scale
            cw = CHAR_WIDTH_FACTOR
            items = [('text', fs * cw / 2, fs * cw / 2, '\u2205', fs)]
            w, h = fs * cw, fs * cw
        else:
            level_seq = obj.level_sequence()
            color_seq = obj.color_sequence()
            items, w, h = _layout_tree(level_seq, color_seq, 0, 0, scale)
            items = _shift_items_x(items, w / 2)
    else:
        raise TypeError("Cannot display object of type " + str(type(obj)))

    return _render_svg(items, w, h, scale)


# ── Jupyter detection ────────────────────────────────────────────────────

def _in_jupyter():
    try:
        from IPython import get_ipython
        ip = get_ipython()
        return ip is not None and 'IPKernelApp' in ip.config
    except (ImportError, AttributeError):
        return False


# ── Public API ───────────────────────────────────────────────────────────

def display(obj: Union[Tree, Forest, ForestSum, TensorProductSum,
                       PlanarTree, NoncommutativeForest],
            *,
            scale: float = 1.0,
            fig_size: tuple = None,
            file_name: str = None,
            use_plt: bool = None,
            rationalise: bool = False) -> None:
    """
    Display a Tree, Forest, ForestSum, TensorProductSum, PlanarTree, or NoncommutativeForest.

    In Jupyter, renders inline SVG.

    :param obj: Object to display
    :param scale: Scale factor for SVG output (default 1.0)
    :param file_name: If provided, saves SVG to ``file_name.svg``
    :param rationalise: If True, rationalise float coefficients
    """
    if not isinstance(obj, (Tree, Forest, ForestSum, TensorProductSum,
                            PlanarTree, NoncommutativeForest)):
        raise TypeError("Cannot display object of type " + str(type(obj))
                        + ". Object must be Tree, Forest, ForestSum, TensorProductSum,"
                        + " PlanarTree, or NoncommutativeForest.")

    if isinstance(obj, ForestSum) and len(obj.term_list) == 0:
        pass  # zero — no colors to check
    elif isinstance(obj, TensorProductSum) and (obj.term_list is None or len(obj.term_list) == 0):
        pass
    elif obj.colors() > 9:
        raise ValueError("Cannot display labelled trees with over 10 different colors.")

    if use_plt is not None:
        warnings.warn("use_plt is deprecated and ignored. Output is now SVG.",
                       DeprecationWarning, stacklevel=2)

    if fig_size is not None:
        warnings.warn("fig_size is deprecated and ignored. SVG auto-sizes.",
                       DeprecationWarning, stacklevel=2)

    svg = _to_svg(obj, scale=scale, rationalise=rationalise)

    if file_name is not None:
        with open(file_name + '.svg', 'w', encoding='utf-8') as fh:
            fh.write(svg)

    if _in_jupyter():
        from IPython.display import display as ipy_display
        ipy_display({'image/svg+xml': svg}, raw=True)
