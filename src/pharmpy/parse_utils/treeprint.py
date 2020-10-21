#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

"""
Pretty printing of tree-structures. Inspired by the 'tree' command (Steve
Baker). In development.
"""

import logging
from collections import namedtuple
from enum import Enum

BranchStyle = Enum('BranchStyle', ['SIMPLE', 'INLINE', 'BUSY'])

Indent = namedtuple('Indent', ['header', 'node', 'fork'])

CharSet = namedtuple('CharSet', ['fork', 'horiz', 'vert', 'inline', 'lfork', 'end'])


class NodeStyle(object):
    __defaults = dict(
        branch=BranchStyle['SIMPLE'],
        indent=Indent(header=0, node=1, fork=1),
        char=CharSet(fork='├', horiz='│', vert='─', inline='┌', lfork='└', end=' '),
    )

    def __init__(self, **styling):
        for key, val_default in self.__defaults.items():
            try:
                val = styling.pop(key)
            except KeyError:
                val = val_default
            finally:
                setattr(self, key, val)
        self.verify()

    def verify(self):
        log = logging.getLogger(__name__ + '.' + self.__class__.__name__)
        if self.indent.header > self.indent.node:
            log.warning("Indent of header > node might break assumptions")
        charlen = set(len(x) for x in self.char[-1])
        if charlen != {1}:
            raise ValueError('len(char)!=1 for some char: %s' % (charlen,))


class Node(object):
    _cls_str = '__str__'
    _cls_map = {}

    def __new__(cls, obj):
        cls_obj = obj.__class__
        try:
            cls_node = cls._cls_map[cls_obj]
        except KeyError:

            def __init__(self, obj):
                self._obj = obj
                cls.__init__(self)

            cls_name = cls_obj.__name__ + cls.__name__
            cls_node = type(cls_name, (cls,), {'__init__': __init__})
            cls._cls_map[cls_obj] = cls_node
        instance = super(cls, cls_node).__new__(cls_node)
        return instance

    def __init__(self):
        self._style = None
        self._parent = None
        self._nodes = list()

    def add(self, *nodes):
        for node in nodes:
            node._parent = self
            self._nodes += [node]

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
        try:
            attr = getattr(self._obj, self._cls_str)
        except TypeError:
            out = str(self._cls_str(self._obj))
        else:
            try:
                out = str(attr())
            except TypeError:
                out = str(attr)
        finally:
            if isinstance(out, str):
                return out.splitlines()
            else:
                return out

    @property
    def type_map(self):
        cdict = {self._obj.__class__: self.__class__}
        for node in self._nodes:
            cdict.update(node.type_map)
        return cdict

    @classmethod
    def set_formatter(cls, formatter):
        cls._cls_str = staticmethod(formatter)

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

        idx_last = len(self._nodes) - 1
        for idx, node in enumerate(self._nodes):
            nlines = str(node).splitlines()
            if not nlines:
                continue
            nlines = self._prefix(nlines, idx < idx_last, node.style.indent)
            lines += nlines

        return '\n'.join(lines)

    def __repr__(self):
        head = '%s %s' % (self.__class__.__name__, repr(self._obj))
        lines = []
        for node in self._nodes:
            lines += [' ' + line for line in repr(node).splitlines()]
        return '\n'.join([head] + lines)


def from_ast(ast_node):
    root = Node(ast_node)
    try:
        fields = root._fields
    except AttributeError:
        fields = []
    for field in fields:
        node = getattr(root, field)
    for child in ast.iter_child_nodes(ast_node):
        node = from_ast(child)
        root.add(node)
    return root


if __name__ == '__main__':
    import ast

    with open(__file__, 'r') as source:
        root = ast.parse(source.read())
    root = from_ast(root)
    print(str(root))
    # print(repr(root))
