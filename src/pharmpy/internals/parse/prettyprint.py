"""Pretty printing of :class:`generic.AttrTree` via :class:`treeprint.Node`."""
from __future__ import annotations

from functools import partial
from typing import Literal, Union

from .tree import Leaf, Tree
from .treeprint import Node

MAXLEN = 60


def _preview(content: str):
    """Represents str and cut into shorts if too long"""
    cut = len(content) - MAXLEN
    if cut > 0:
        preview = '"' + repr(content[0:MAXLEN])[1:-1] + '"'
        preview = '%s+%d' % (preview, cut)
    else:
        preview = '"' + repr(content)[1:-1] + '"'
    return preview


def _formatter(content: Union[bool, Literal['full']], ast_node):
    """Formats AST node (Tree and Leaf) for treeprint"""
    if content:
        if content == 'full':
            lines = [ast_node.rule] + str(ast_node).splitlines(True)
            lines[1:] = [_preview(line) for line in lines[1:]]
            return '\n'.join(lines)
        else:
            return '%s %s' % (ast_node.rule, _preview(str(ast_node)))
    else:
        return '%s' % (ast_node.rule,)


def _format_tree(content: Union[bool, Literal['full']], ast_tree: Tree):
    return _formatter(content, ast_tree)


def _format_token(content: Union[bool, Literal['full']], ast_token: Leaf):
    return _formatter(content, ast_token)


def transform(ast_tree_or_token: Union[Tree, Leaf], content: Union[bool, Literal['full']] = True):
    """
    Traverses tree and generates :class:`treeprint.Node`, which can format a
    multiline (command) tree-styled string.

    Args:
        content: If True, include text preview (or all if 'full') on each node.

    Returns:
        Multiline string, ready for the printing press.
    """

    if isinstance(ast_tree_or_token, Leaf):
        formatter = partial(_format_token, content)
        return Node(ast_tree_or_token, cls_str=formatter, children=())
    elif isinstance(ast_tree_or_token, Tree):
        nodes = ast_tree_or_token.children
        if isinstance(nodes, str):
            raise TypeError(
                "'children' of tree appears to be 'str' (expects list/iterable): %s"
                % repr(ast_tree_or_token)
            )
        formatter = partial(_format_tree, content)
        children = tuple(transform(ast_node, content) for ast_node in list(nodes))
        return Node(ast_tree_or_token, cls_str=formatter, children=children)
    else:
        raise TypeError(
            "can't transform %s object (is not a Lark Tree or Token')"
            % repr(ast_tree_or_token.__class__.__name__)
        )
