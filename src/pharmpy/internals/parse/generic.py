"""
Generic parser using lark.

Subclass :class:`GenericParser` (remember to set :attr:`GenericParser.grammar`
to point to your grammar file) to define a powerful parser.
"""
from __future__ import annotations

import re
from abc import ABC
from itertools import chain
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, Union

from lark import Lark, Transformer, Tree, Visitor
from lark.lexer import Token

from . import prettyprint
from .tree import Leaf as ImmutableLeaf
from .tree import Tree as ImmutableTree

TYPES_OF_NEWLINE = frozenset(('CONT', 'NEWLINE'))
TYPES_OF_COMMENT = frozenset(('COMMENT',))


class AttrToken(ImmutableLeaf):
    def __repr__(self):
        return '%s(%s, %s)' % (self.__class__.__name__, repr(self.rule), repr(self.value))

    def __str__(self):
        return str(self.value)


def _newline_node():
    return AttrToken('NEWLINE', '\n')


def eval_token(token: AttrToken) -> Union[int, float, str]:
    """Evaluated value (str, int, float)."""
    if token.rule in ('DIGIT', 'INT', 'SIGNED_INT'):
        return int(token.value)
    elif token.rule in (
        'DECIMAL',
        'EXP',
        'FLOAT',
        'FLOAT2',
        'NUMBER',
        'NUMERIC',
        'SIGNED_FLOAT',
        'SIGNED_NUMBER',
    ):
        return float(token.value)
    elif token.rule == 'NEG_INF':
        return float('-INF')
    elif token.rule == 'POS_INF':
        return float('INF')
    else:
        return token.value


def _parse_create_input(rule: Optional[str], items):
    try:
        names, items = zip(*items.items())
    except AttributeError:
        if isinstance(items, str):
            raise TypeError(str)
        length = len(items)
        names: Tuple[Optional[str]] = (None,) * length
    if not items:
        raise ValueError(f'refusing empty tree {repr(rule)} (only tokens are childless)')

    return (rule, list(zip(names, items)), [])


class AttrTree(ImmutableTree['AttrTree', 'AttrToken']):
    """Tree with attribute access.

    Created in :meth:`GenericParser.parse` by :meth:`transform`, from :class:`lark.Tree`.

    Attributes:
        self.rule: Name, in common with :class:`.AttrToken`.
        self.rules: Names of children.
        self.eval: Transformed data type, in common with :class:`.AttrToken`.
        self.tokens: Recursive tokens as (flattened) list.
        self.debug: Treeview str, formatted for debugging.

    Can be instantiated with :meth:`.__init__`, via :class:`lark.Tree`, or alternative constructors:
        1. :meth:`.transform` (transform recursively object of class :class:`lark.Tree`).
        2. :meth:`.create` (create from nested iterators).
    """

    @staticmethod
    def create(rule: Optional[str], items) -> AttrTree:
        """Alternative constructor: Creates new tree from (possibly nested) iterables.

        Only non-iterable items become leaves (i.e. content of token nodes). All others are trees.

        Handling of missing names (e.g. lists):
            1. Tree: Children are moved to parent.
            2. Token: Native naming scheme, __ANON_%d, of Lark is used.

        Args:
            rule: Name of root tree (__ANON_0 if False).
            items: Child (tree) nodes or content (token) nodes.

        Raises:
            TypeError: 'items' not iterable or instance of 'str' (only leaves shall contain 'str').
            ValueError: 'items' empty (trees can't be empty).

        .. note:: Please follow convention of all lower/upper case for trees/tokens.
        """

        _anon_count = 0
        stack = [_parse_create_input(rule, items)]

        while True:
            rule_name, args, children = stack[-1]
            i = len(children)
            if i == len(args):
                # NOTE We flatten anonymous trees
                flattened = tuple(
                    chain.from_iterable(
                        child if isinstance(child, tuple) else (child,) for child in children
                    )
                )
                tree = (
                    flattened
                    if not rule_name
                    else AttrTree(
                        rule_name,
                        flattened,
                    )
                )
                stack.pop()
                if stack:
                    # NOTE We had this tree as child of parent
                    stack[-1][-1].append(tree)
                    continue
                else:
                    return tree if isinstance(tree, AttrTree) else AttrTree('__ANON_0', tree)

            name, thing = args[i]

            if isinstance(thing, (AttrTree, AttrToken)):
                # NOTE Do not convert existing nodes
                children.append(thing)
            elif isinstance(thing, (str, int, float, type(None))):
                # NOTE leaf
                if not name:
                    _anon_count += 1
                    name = '__ANON_%d' % (_anon_count,)
                children.append(AttrToken(name, str(thing)))
            else:
                # NOTE recurse
                stack.append(_parse_create_input(name, thing))

    # -- public interface ----------------------------------------------
    @property
    def rules(self) -> List[str]:
        """All rules of (immediate) children."""
        return [node.rule for node in self.children]

    def first_index(self, rule: str) -> int:
        """Returns index of first child matching 'rule', or -1."""
        return next((i for i, child in enumerate(self.children) if child.rule == rule), -1)

    def find(self, rule) -> Optional[Union[AttrTree, AttrToken]]:
        """Returns first child matching 'rule', or None."""
        i = self.first_index(rule)
        return None if i == -1 else self.children[i]

    def first_branch(self, *rules) -> Optional[Union[AttrTree, AttrToken]]:
        current = self
        for rule in rules:
            if not isinstance(current, AttrTree):
                return None
            current = current.find(rule)
        return current

    def set(self, rule, new_child) -> AttrTree:
        """Sets first child matching rule. Raises if none."""
        i = self.first_index(rule)
        if i == -1:
            raise NoSuchRuleException(rule, self)

        return AttrTree(self.rule, self.children[:i] + (new_child,) + self.children[i + 1 :])

    def partition(
        self, rule
    ) -> Tuple[
        Tuple[Union[AttrTree, AttrToken], ...],
        Tuple[Union[AttrTree, AttrToken], ...],
        Tuple[Union[AttrTree, AttrToken], ...],
    ]:
        """Partition children into (head, item, tail).

        Search for child item 'rule' and return the part before it (head), the item, and the part
        after it (tail). If 'rule' is not found, return (children, [], [])."""

        i = self.first_index(rule)
        return (
            (self.children, (), ())
            if i == -1
            else (self.children[:i], (self.children[i],), self.children[i + 1 :])
        )

    def tree_walk(self):
        """Generator for iterating depth-first (i.e. parse order) over children."""
        for child in self.children:
            yield child
            if isinstance(child, AttrTree):
                yield from child.tree_walk()

    def remove(self, rule):
        """Remove all children with rule. Not recursively"""
        return AttrTree(
            self.rule,
            tuple(child for child in self.children if child.rule != rule),
        )

    def _clean_ws(self, new_children: Sequence[Union[AttrTree, AttrToken]]):
        new_children_clean = []
        last_index = len(new_children) - 1

        prev_rule = None

        for i, child in enumerate(new_children):
            if child.rule in TYPES_OF_NEWLINE:
                if prev_rule in TYPES_OF_NEWLINE or prev_rule == 'verbatim':
                    continue
                if i == last_index:
                    continue
            if re.search('\n{2,}', str(child)):
                new_children_clean.append(_newline_node())
            else:
                new_children_clean.append(child)

            prev_rule = child.rule

        return new_children_clean

    def add_node(self, node, following_node=None, comment=False):
        new_children = self._clean_ws(self.children) if comment else list(self.children)

        if following_node is None:
            if not comment:
                new_children.append(_newline_node())
            new_children.append(node)
        else:
            index = self.children.index(following_node)
            new_children.insert(index, node)

            newline_node = new_children[index - 1]
            new_children.insert(index + 1, newline_node)

        new_children_clean = self._clean_ws(new_children)
        return AttrTree(self.rule, tuple(new_children_clean))

    def add_comment_node(self, comment, adjacent_node=None):
        comment_node = AttrToken('COMMENT', f' ; {comment}')
        return self.add_node(comment_node, adjacent_node, comment=True)

    def add_newline_node(self):
        return AttrTree(self.rule, self.children + (_newline_node(),))

    def subtrees(self, rule) -> Iterable[AttrTree]:
        for child in self.children:
            if isinstance(child, AttrTree) and child.rule == rule:
                yield child

    def subtree(self, rule) -> AttrTree:
        try:
            return next(iter(self.subtrees(rule)))
        except StopIteration:
            raise NoSuchRuleException(f'No subtree "{rule}" in {repr(self)}.')

    def subtree_at(self, i: int) -> AttrTree:
        child = self.children[i]
        if isinstance(child, AttrTree):
            return child

        raise ValueError(f'The is no subtree at index {i}')

    def leaves(self, rule) -> Iterable[AttrToken]:
        for child in self.children:
            if isinstance(child, AttrToken) and child.rule == rule:
                yield child

    def leaf(self, rule) -> AttrToken:
        try:
            return next(iter(self.leaves(rule)))
        except StopIteration:
            raise NoSuchRuleException(f'No leaf "{rule}" in {repr(self)}.')

    def leaf_at(self, i: int) -> AttrToken:
        child = self.children[i]
        if isinstance(child, AttrToken):
            return child

        raise ValueError(f'The is no subtree at index {i}')

    def replace_first(self, child: Union[AttrTree, AttrToken]) -> AttrTree:
        i = self.first_index(child.rule)
        if i == -1 or self.children[i] == child:
            return self

        new_children = self.children[:i] + (child,) + self.children[i + 1 :]
        return AttrTree(self.rule, new_children)

    def map(
        self, fn: Callable[[Union[AttrTree, AttrToken]], Union[AttrTree, AttrToken]]
    ) -> AttrTree:
        return AttrTree(self.rule, tuple(map(fn, self.children)))

    @property
    def tokens(self) -> Iterable[AttrToken]:
        """All tokens in depth-first order."""
        for child in self.tree_walk():
            if isinstance(child, AttrToken):
                yield child

    @property
    def debug(self, *args, **kwargs):
        """Debug formatted tree structure."""
        return str(prettyprint.transform(self, *args, **kwargs))

    def treeprint(self, indent=''):
        """Prints debug formatted tree structure."""
        print(self.debug)

    # -- private methods -----------------------------------------------
    def __len__(self):
        return len(self.children)

    def __str__(self):
        return ''.join(str(x) for x in self.children)

    def __repr__(self):
        return '%s(%s, %s)' % (self.__class__.__name__, repr(self.rule), repr(self.children))


class NoSuchRuleException(AttributeError):
    """Rule not found (raised by :class:`AttrTree` for unknown children)."""

    def __init__(self, rule, tree: Optional[AttrTree] = None):
        post = '' if tree is None else f' ({repr(tree.rule)})'
        super().__init__(f'no {repr(rule)} child in tree{post}')


class GenericParser(ABC):
    """
    Generic parser using lark.

    Inherit to define a parser, say ThetaRecordParser for NONMEM, with the workflow:

    1. Lex and parse a 'buffer' using Lark_ (from :attr:`grammar` file). Builds AST.
    2. Convert to :class:`AttrTree` for immutability and convenience traversal.

    Attributes:
        self.buffer: Buffer parsed by :meth:`parse`.
        self.grammar: Path to grammar file.
        self.root: Root of final tree. Instance of :class:`AttrTree`.

    .. _Lark:
        https://github.com/lark-parser/lark
    """

    """:class:`AttrTree` implementation."""
    AttrTree = AttrTree

    lark: Lark
    lark_options = dict(
        start='root',
        parser='lalr',
        keep_all_tokens=True,
        propagate_positions=False,
        maybe_placeholders=False,
        debug=False,
        cache=False,
    )
    post_process: Tuple[Union[Visitor, Transformer, Callable[[str, Tree], Tree]], ...] = ()

    def __init__(self, buf=None):
        self.root = self.parse(buf)

    def parse(self, buf):
        """Parses a buffer, transforms and constructs :class:`AttrTree` object.

        Args:
            buf: Buffer to parse.
        """
        self.buffer = buf
        if self.buffer is None:
            return None

        root = self.lark.parse(self.buffer)

        for processor in self.post_process:
            if isinstance(processor, Visitor):
                processor.visit(root)
            elif isinstance(processor, Transformer):
                root = processor.transform(root)
            elif isinstance(processor, Callable):
                root = processor(self.buffer, root)
            else:
                raise TypeError(f'Processor {processor} must be a Visitor or a Transformer')

        return _from_lark_tree(root)

    def __str__(self):
        if not self.root:
            return repr(self)
        lines = str(prettyprint.transform(self.root)).splitlines()
        return '\n'.join(lines)


def _remove_token_and_space(node: Union[AttrTree, AttrToken], rule: str):
    if isinstance(node, AttrTree):
        return remove_token_and_space(node, rule, recursive=True)
    else:
        return node


def remove_token_and_space(tree: AttrTree, rule: str, recursive: bool = False):
    """Remove all tokens with rule and any WS before it"""
    new_nodes = []
    for node in tree.children:
        if node.rule == rule:
            if len(new_nodes) > 0 and new_nodes[-1].rule == 'WS':
                new_nodes.pop()
        else:
            new_nodes.append(node)

    if recursive:
        new_nodes = (_remove_token_and_space(node, rule) for node in new_nodes)

    return AttrTree(tree.rule, tuple(new_nodes))


def insert_before_or_at_end(
    tree: ImmutableTree, rule: str, nodes: Iterable[Union[ImmutableTree, ImmutableLeaf]]
):
    """Insert nodes before rule or if rule does not exist at end"""
    kept = []
    found = False
    for node in tree.children:
        if node.rule == rule:
            kept.extend(nodes)
            found = True
        kept.append(node)
    if not found:
        kept.extend(nodes)

    return AttrTree(tree.rule, tuple(kept))


def insert_after(
    tree: ImmutableTree, rule: str, nodes: Iterable[Union[ImmutableTree, ImmutableLeaf]]
):
    """Insert nodes after rule"""
    kept = []
    for node in tree.children:
        kept.append(node)
        if node.rule == rule:
            kept.extend(nodes)

    return AttrTree(tree.rule, tuple(kept))


def _from_lark_tree(tree: Tree) -> AttrTree:
    return AttrTree(
        tree.data,
        tuple(
            _from_lark_tree(child) if isinstance(child, Tree) else _from_lark_token(child)
            for child in tree.children
        ),
    )


def _from_lark_token(token: Token) -> AttrToken:
    return AttrToken(token.type, token.value)
