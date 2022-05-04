from abc import abstractmethod
from typing import List, Iterator, Union
from docutils import nodes
from docutils.statemachine import ViewList, string2lines
from docutils.parsers.rst import Directive, directives

from conversion import transpile_py_to_r

def setup(app):
    app.add_directive('pharmpy-execute', PharmpyExecute)
    app.add_directive('pharmpy-code', PharmpyCode)

    return {
        'version': '0.1',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }

def csv_option(s):
    return [p.strip() for p in s.split(",")] if s else []

class RecursiveDirective(Directive):

    def _convert_lines_to_nodes(self, lines: List[str]) -> List[nodes.Node]:
        """Turn an RST string into a node that can be used in the document.

           See https://github.com/sphinx-doc/sphinx/issues/8039
        """

        node = nodes.Element()
        node.document = self.state.document
        self.state.nested_parse(
            ViewList(
                string2lines('\n'.join(lines)),
                source='[SnippetDirective]',
            ),
            self.content_offset,
            node,
        )

        return node.children

class PharmpyAbstractCodeDirective(RecursiveDirective):

    option_spec = {
        'linenos': directives.flag,
        'lineno-start': directives.nonnegative_int,
        'emphasize-lines': directives.unchanged_required,
    }

    def run(self):
        return self._nodes()

    def _nodes(self):
        lines = self._lines()
        return self._convert_lines_to_nodes(lines)

    @abstractmethod
    def _lines(self) -> List[str]:
        """Return lines for this directive"""

    def _input(self):
        return [
            '.. tabs::',
            *_indent(3, [
                '',
                '.. code-tab:: py',
                *_indent(3, self._code_option_lines()),
                '',
                *_indent(3, self.content),
                '',
                '.. code-tab:: r R',
                *_indent(3, self._code_option_lines()),
                '',
                *_indent(3, transpile_py_to_r(self.content)),
            ]),
        ]

    def _code_option_lines(self):
        if 'emphasize-lines' in self.options:
            yield f':emphasize-lines:{self.options.get("emphasize-lines")}'
        if 'linenos' in self.options:
            yield ':linenos:'
        if 'lineno-start' in self.options:
            yield f':lineno-start:{self.options.get("lineno-start")}'


class PharmpyExecute(PharmpyAbstractCodeDirective):
    required_arguments = 0
    optional_arguments = 0
    final_argument_whitespace = True
    has_content = True

    option_spec = {
        **PharmpyAbstractCodeDirective.option_spec,
        'hide-code': directives.flag,
        'hide-output': directives.flag,
        'code-below': directives.flag,
        'raises': csv_option,
        'stderr': directives.flag,
    }

    def _lines(self) -> List[str]:
        return [
            f'.. container:: pharmpy-snippet{"" if "hide-output" in self.options else " with-output"}',
            '',
            *_indent(3, self._input_output_lines())
        ]

    def _input_output_lines(self):
        # NOTE self._output should always be returned here, even when
        # `hide-output` is set, otherwise the code will not be executed.
        if 'hide-code' in self.options:
            return self._output()

        if 'code-below' in self.options:
            return [
                *self._output(),
                '',
                *self._input(),
            ]

        return [
            *self._input(),
            '',
            *self._output(),
        ]


    def _output(self):
        return [
            '.. jupyter-execute::',
            *_indent(3, [
                *self._jupyter_option_lines(),
                '',
                *self.content
            ]),
        ]

    def _jupyter_option_lines(self):
        yield ':hide-code:'
        if 'hide-output' in self.options:
            yield ':hide-output:'
        if 'raise' in self.options:
            yield f':raises:{",".join(self.options.get("raises"))}'
        if 'stderr' in self.options:
            yield ':stderr:'


class PharmpyCode(PharmpyAbstractCodeDirective):
    required_arguments = 0
    optional_arguments = 0
    final_argument_whitespace = True
    has_content = True

    option_spec = PharmpyAbstractCodeDirective.option_spec

    def _lines(self) -> List[str]:
        return [
            '.. container:: pharmpy-snippet',
            '',
            *_indent(3, self._input())
        ]


def _indent(n: int, lines: Union[List[str],Iterator[str]]):
    return map(lambda line: (' '*n + line) if line else line, lines)
