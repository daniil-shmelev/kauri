"""
Sphinx extension: ``kauri-exec`` directive.

Executes a Python code block at build time, captures ``print()`` output
and ``kauri.display()`` SVG output, and renders both inline in the docs.

Options
-------
:hide-code:    Show only the output (no code block).
:hide-output:  Show only the code block (no output).
"""

import io
import sys
import contextlib

from docutils import nodes
from docutils.parsers.rst import directives
from sphinx.util.docutils import SphinxDirective


class KauriExecDirective(SphinxDirective):
    has_content = True
    option_spec = {
        'hide-code': directives.flag,
        'hide-output': directives.flag,
    }

    def run(self):
        code = '\n'.join(self.content)
        result_nodes = []

        # ── code block ──────────────────────────────────────────────
        if 'hide-code' not in self.options:
            literal = nodes.literal_block(code, code)
            literal['language'] = 'python'
            literal['linenos'] = False
            result_nodes.append(literal)

        if 'hide-output' in self.options:
            return result_nodes

        # ── execute ─────────────────────────────────────────────────
        captured_svgs = []

        from kauri.display import _to_svg

        def _capturing_display(obj, *, scale=1.0, rationalise=False, **kw):
            svg = _to_svg(obj, scale=scale, rationalise=rationalise)
            captured_svgs.append(svg)

        import kauri
        # kauri.display is the *function* (re-exported in __init__).
        # Access the *module* through sys.modules.
        _disp_mod = sys.modules['kauri.display']

        orig_pkg_display = kauri.display          # the function on the package
        orig_mod_display = _disp_mod.display      # the function on the module
        kauri.display = _capturing_display
        _disp_mod.display = _capturing_display

        ns = {'kauri': kauri, 'kr': kauri}
        stdout_buf = io.StringIO()

        try:
            with contextlib.redirect_stdout(stdout_buf):
                exec(code, ns)  # noqa: S102
        except Exception as exc:
            error = self.state_machine.reporter.error(
                f'kauri-exec failed:\n{exc}',
                nodes.literal_block(code, code),
                line=self.lineno,
            )
            return [error]
        finally:
            kauri.display = orig_pkg_display
            _disp_mod.display = orig_mod_display

        # ── stdout ──────────────────────────────────────────────────
        stdout_text = stdout_buf.getvalue()
        if stdout_text.strip():
            result_nodes.append(
                nodes.literal_block(stdout_text, stdout_text)
            )

        # ── SVGs ────────────────────────────────────────────────────
        for svg in captured_svgs:
            result_nodes.append(
                nodes.raw('', f'<div style="margin-bottom:1.5em;">{svg}</div>',
                           format='html')
            )

        return result_nodes


def setup(app):
    app.add_directive('kauri-exec', KauriExecDirective)
    return {'version': '0.1', 'parallel_read_safe': False}
