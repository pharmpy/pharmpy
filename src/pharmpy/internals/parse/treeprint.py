"""
Pretty printing of tree-structures. Inspired by the 'tree' command (Steve
Baker). In development.
"""
from __future__ import annotations

import logging
from collections import namedtuple
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Generic, Optional, Tuple, TypeVar, Union


class BranchStyle(Enum):
    SIMPLE = 1
    INLINE = 2
    BUSY = 3


Indent = namedtuple('Indent', ['header', 'node', 'fork'])

CharSet = namedtuple('CharSet', ['fork', 'horiz', 'vert', 'inline', 'lfork', 'end'])


@dataclass(frozen=True)
class NodeStyle:
    branch: BranchStyle = BranchStyle['SIMPLE']
    indent: Indent = Indent(header=0, node=1, fork=1)
    char: CharSet = CharSet(fork='├', horiz='│', vert='─', inline='┌', lfork='└', end=' ')

    def __post_init__(self):
        log = logging.getLogger(__name__ + '.' + self.__class__.__name__)
        if self.indent.header > self.indent.node:
            log.warning("Indent of header > indent of node. Might break assumptions.")
        for x in self.char:
            if len(x) != 1:
                raise ValueError(f'len({x}) != 1 (got {len(x)})')


T = TypeVar('T')


class Node(Generic[T]):
    def __init__(
        self,
        obj: T,
        cls_str: Union[str, Callable[[T], str]] = '__str__',
        style: Optional[NodeStyle] = None,
        children: Tuple[Node, ...] = (),
    ):
        self._obj = obj
        self._cls_str = cls_str
        self._style = style
        self._parent = None
        self._children = children
        # NOTE This links the children to the new parent and forces to build
        # the tree from bottom to top
        for child in children:
            child._parent = self

    @property
    def style(self):
        if self._style:
            return self._style
        elif self._parent:
            return self._parent.style
        else:
            return NodeStyle()

    @style.setter
    def style(self, **kwargs):
        self._style = NodeStyle(**kwargs)

    @property
    def lines(self):
        if isinstance(self._cls_str, str):
            attr = getattr(self._obj, self._cls_str)
            try:
                out = str(attr())
            except TypeError:
                out = str(attr)
        else:
            out = str(self._cls_str(self._obj))

        return out.splitlines() if isinstance(out, str) else out

    def _prefix(self, lines, early_branch, ind_used):
        if self.style.branch != BranchStyle.SIMPLE:
            raise NotImplementedError('only BranchStyle.SIMPLE suppported')
        ind = self.style.indent
        char = self.style.char
        if early_branch:
            ch_first, ch_after = (char.fork, char.horiz)
        else:
            ch_first, ch_after = (char.lfork, ' ')

        vnew = char.vert * ind.node
        vext = char.vert * ind_used.header
        for i, line in enumerate(lines):
            if i == 0:
                lines[i] = ' ' * ind.fork + ch_first + vnew + vext + char.end + line[len(vext) :]
                continue
            lines[i] = ' ' * ind.fork + ch_after + ' ' * ind.node + line

        return lines

    def __str__(self):
        ind = self.style.indent
        lines = self.lines
        lines[0] = ' ' * ind.header + lines[0]

        idx_last = len(self._children) - 1
        for idx, node in enumerate(self._children):
            nlines = str(node).splitlines()
            if not nlines:
                continue
            nlines = self._prefix(nlines, idx < idx_last, node.style.indent)
            lines += nlines

        return '\n'.join(lines)

    def __repr__(self):
        head = f'{self.__class__.__name__}({self._obj.__class__.__name__}) {repr(self._obj)}'
        lines = [' ' + line for child in self._children for line in repr(child).splitlines()]
        return '\n'.join([head] + lines)


def from_ast(ast_node):
    from ast import iter_child_nodes

    children = tuple(map(from_ast, iter_child_nodes(ast_node)))
    root = Node(ast_node, children=children)
    return root


if __name__ == '__main__':
    with open(__file__, 'r') as source:
        from ast import parse

        root = parse(source.read())

    root = from_ast(root)
    print('=' * 40, 'repr(...)', '=' * 40)
    print(repr(root))
    print()
    print('=' * 40, 'str(...)', '=' * 40)
    print(str(root))
