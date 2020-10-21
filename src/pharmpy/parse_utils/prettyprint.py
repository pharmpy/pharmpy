"""Pretty printing of :class:`generic.AttrTree` via :class:`treeprint.Node`."""
from functools import partial

import lark

from . import treeprint

MAXLEN = 60


def _preview(content):
    """Represents str and cut into shorts if too long"""
    cut = len(content) - MAXLEN
    if cut > 0:
        preview = '"' + repr(content[0:MAXLEN])[1:-1] + '"'
        preview = '%s+%d' % (preview, cut)
    else:
        preview = '"' + repr(content)[1:-1] + '"'
    return preview


def _formatter(content, ast_node):
    """Formats AST node (Tree and Token) for treeprint"""
    if not content:
        return '%s' % (ast_node.rule,)
    elif content != 'full':
        return '%s %s' % (ast_node.rule, _preview(str(ast_node)))
    lines = [ast_node.rule] + str(ast_node).splitlines(True)
    lines[1:] = [_preview(line) for line in lines[1:]]
    return '\n'.join(lines)


def _format_tree(content, ast_tree):
    return _formatter(content, ast_tree)


def _format_token(content, ast_token):
    return _formatter(content, ast_token)


def transform(ast_tree, content=True):
    """
    Traverses tree and generates :class:`treeprint.Node`, which can format a
    multiline (command) tree-styled string.

    Args:
        content: If True, include text preview (or all if 'full') on each node.

    Returns:
        Multiline string, ready for the printing press.
    """

    if not any(isinstance(ast_tree, x) for x in [lark.Tree, lark.lexer.Token]):
        raise TypeError(
            "can't transform %s object (is not a Lark Tree or Token')"
            % repr(ast_tree.__class__.__name__)
        )
    tree = treeprint.Node(ast_tree)
    try:
        nodes = ast_tree.children
    except AttributeError:
        tree.set_formatter(partial(_format_token, content))
    else:
        if isinstance(nodes, str):
            raise TypeError(
                "'children' of tree appears to be 'str' (expects list/iterable): %s"
                % repr(ast_tree)
            )
        tree.set_formatter(partial(_format_tree, content))
        for ast_node in list(nodes):
            node = transform(ast_node, content)
            tree.add(node)
    return tree
