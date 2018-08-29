#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

"""
Generic parser using lark-parser.

Subclass :class:`GenericParser` (remember to set :attr:`GenericParser.grammar`
to point to your grammar file) to define a powerful parser.

(Optional) Subclass :class:`AttrTree` and :class:`AttrToken` (and set
:attr:`GenericParser.AttrTree` and :attr:`GenericParser.AttrTree.AttrToken`).

.. _Google Python Style Guide:
   http://google.github.io/styleguide/pyguide.html
"""

from pathlib import Path

from lark import Lark
from lark import Transformer
from lark import Tree
from lark.lexer import Token

from . import prettyprint


def item_rule(item):
    """Rule of a tree or token"""
    try:
        return item.data
    except AttributeError:
        try:
            return item.type
        except AttributeError:
            return ''


def new_item(rule):
    """Create empty Tree or Token (if rule is upper case only)"""
    if rule == rule.upper():
        return Token(rule, '')
    return Tree(rule, [])


class AttrToken(Token):
    """
    Token with attribute access.

    Created by :method:`AttrTree.transform`, from :class:`lark.lexer.Token`
    (see 'token' argument of :method:`__new__`).

    Override this and :class:`AttrTree` if you need special behaviour for your
    implementation (subclass) of :class:`GenericParser`.

    Attributes:
        rule (str): Rule name (in common with :class:`AttrTree`).
        eval: Transformed data type (in common with :class:`AttrTree`).
    """

    __slots__ = ()

    @classmethod
    def transform(cls, token, **kwargs):
        """Alternative constructor: from Token (with optional overrides)."""
        kwargs = {'type_': token.type, 'value': token.value, 'pos_in_stream': token.pos_in_stream,
                  'line': token.line, 'column': token.column, **kwargs}
        return cls(**kwargs)

    def replace(self, value):
        """Returns new Token (of same rule) with its content replaced."""
        return self.transform(token=self, value=value)

    @property
    def rule(self):
        """Rule name (synonymous with 'type')"""
        return self.type

    @rule.setter
    def rule(self, value):
        self.type = value

    @property
    def eval(self):
        if self.type == 'INT':
            return int(self.value)
        elif self.type == 'NUMERIC':
            return float(self.value)
        else:
            return str(self.value)

    def __repr__(self):
        return '%s(%s, %s)' % (self.__class__.__name__, repr(self.rule), repr(self.value))

    def __hash__(self):
        return hash(repr(self))

    def __eq__(self, other):
        return hash(self) == hash(other)


class NoSuchRuleException(AttributeError):
    """Rule not found exceptions"""

    def __init__(self, rule, tree=None):
        try:
            post = ' (%s)' % (repr(tree.rule),)
        except AttributeError:
            post = ''
        super().__init__('no %s child in tree%s' % (repr(rule), post))


class AttrTree(Tree):
    """
    Tree with attribute access.

    Created in :method:`GenericParser.parse` by :method:`transform`, from
    :class:`lark.Tree`.

    Override this and :class:`.AttrToken` if you need special behaviour for
    your implementation (subclass) of :class:`GenericParser`.

    Attributes:
        rule: Rule name (in common with :class:`.AttrToken`).
        rules: All rule names (for :attr:`.children`).
        eval: Transformed data type (in common with :class:`.AttrToken`).
    """

    __dict__ = {'data': None, 'children': None, '_meta': None}

    AttrToken = AttrToken

    @classmethod
    def transform(cls, tree, **kwargs):
        """Alternative constructor: From Tree (with optional overrides)."""
        kwargs = {'data': tree.data, 'children': tree.children, 'meta': tree._meta, **kwargs}
        children = kwargs['children'].copy()
        for i, child in enumerate(children):
            if isinstance(child, Tree):
                children[i] = cls.transform(tree=child)
            elif isinstance(child, Token):
                children[i] = cls.AttrToken.transform(token=child)
            else:
                children[i] = cls.AttrToken('STRING', str(child))
        kwargs['children'] = children
        return cls(**kwargs)

    @classmethod
    def create(cls, rule, items, _anon_count=0, _list=False):
        """Alternative constructor: Creates new tree from (possibly nested) iterables.

        Only non-iterable items become leaves (i.e. content of token nodes). All others are trees.

        Missing (False-evaluating) names:
            1. Tree: Children are moved up.
            2. Token: The __ANON_%d native naming scheme of Lark is used.

        Args:
            items: Children (tree nodes) or content (token nodes).
            rule: Name of root tree. __ANON_0 if False.
            _anon_count: Internal. Anonymous numbering offset.
            _list: Internal. Recursion state. Drop 'rule' & return list of children, which are
                orphaned if name is False.

        Raises:
            TypeError: 'items' not iterable or instance of 'str' (only tokens contain 'str').
            ValueError: 'items' empty (trees can't be empty).

        NOTE: Please follow convention of all lower case for trees and all upper for tokens.
        """

        def rule_or_anon(rule, count):
            """Returns (rule, count). Rule is numbered anonymous if False (and count increased)."""
            if rule:
                return rule, count
            else:
                return '__ANON_%d' % (count+1), count+1

        def non_iterable(rule, items):
            raise TypeError('%s object is not iterable (of children for tree %s)' %
                            repr(items.__class__.__name__), repr(rule))

        # determine mode of operation; 'items' dict-like OR just iterable?
        try:
            names, items = zip(*items.items())
        except AttributeError:
            if isinstance(items, str):
                non_iterable(rule, items)
            try:
                length = len(items)
            except TypeError:
                non_iterable(rule, items)
            names = [None]*length
        if not items:
            raise ValueError('refusing empty tree %s (only tokens are childless)' % repr(rule))

        # create the nodes
        new_nodes = []
        for name, thing in zip(names, items):
            try:  # try to recurse down
                nodes, _anon_count = cls.create(name, thing, _anon_count=_anon_count, _list=True)
            except TypeError:  # looks like a leaf
                try:  # don't convert existing nodes (to leaves)
                    name = thing.rule
                except AttributeError:
                    name, _anon_count = rule_or_anon(name, _anon_count)
                    new_nodes += [cls.AttrToken(name, str(thing))]
                else:  # node already, won't recreate
                    new_nodes += [thing]
            else:
                new_nodes += nodes
        # list (and counter) for recursion
        if _list:
            return [cls(rule, new_nodes)] if rule else new_nodes, _anon_count

        # tree to external caller
        if rule:
            return cls(rule, new_nodes)
        return cls('__ANON_0', new_nodes)

    # -- public interface ----------------------------------------------
    @property
    def rule(self):
        """Rule name (synonymous with 'data')."""
        return self.data

    @rule.setter
    def rule(self, value):
        self.data = value

    @property
    def rules(self):
        """All rule names of (immediate) children."""
        return [node.rule for node in self.children]

    @property
    def eval(self):
        """Evaluated value (self)."""
        return self

    def find(self, rule):
        """Gets first child matching rule, or None if none."""
        return next((child for child in self.children if child.rule == rule), None)

    def all(self, rule):
        """Gets all children matching rule, or [] if none."""
        return list(filter(lambda child: child.rule == rule, self.children))

    def set_child(self, rule, value):
        """Sets first child matching rule (raises if none)."""
        for i, child in enumerate(self.children):
            if child.rule == rule:
                self.children[i] = value
                return
        raise NoSuchRuleException(rule, self)

    def tree_walk(self):
        """Generator for (depth-first, i.e. parse order) iter over children."""
        for child in self.children:
            yield child
            try:
                yield from child.tree_walk()
            except AttributeError:
                continue

    def set(self, value, rule=None):
        """Sets value for first token matching rule (or leaf if None)."""
        if rule:
            token = self.find(rule)
        else:
            if not len(self.children) == 1:
                raise AssertionError('no rule but node not leaf (ambigiuous)')
            token = self.children[0]
            rule = token.rule
        try:
            new = token.replace(value)
        except AttributeError as e:
            raise TypeError('child not a token: %s' % (repr(token),)) from e
        self.set_child(rule, new)

    def treeprint(self, content=True, indent=''):
        """Formats tree structure (for grammar debugging purposes)."""
        lines = str(prettyprint.transform(self, content)).splitlines()
        print('\n'.join(indent + line for line in lines))

    # -- private methods -----------------------------------------------
    def __len__(self):
        return len(self.children)

    def __setattr__(self, attr, value):
        if attr in dir(self):
            return object.__setattr__(self, attr, value)
        self.set_child(attr, value)

    def __getattr__(self, attr):
        if attr in dir(self):
            return object.__getattribute__(self, attr)
        child = self.find(attr)
        if child is None:
            raise NoSuchRuleException(attr, self)
        return child.eval

    def __str__(self):
        return ''.join(str(x) for x in self.children)

    def __repr__(self):
        return '%s(%s, %s)' % (self.__class__.__name__, repr(self.rule), repr(self.children))

    def __hash__(self):
        return hash(repr(self))

    def __eq__(self, other):
        return hash(self) == hash(other)


class PreParser(Transformer):
    """
    Pre-parsing transformer to shape tree (after initial AST).

    Methods after rule's (Tree's or Token's) will be visited by the Lark parser visitor, with
    children as argument, and the return value replaces).
    """

    # -- public interface ----------------------------------------------
    @classmethod
    def first(cls, rule, tree_or_items):
        """Returns first item in tree or list, matching rule"""
        return cls._find(rule, tree_or_items, 'single')

    @classmethod
    def split(cls, rule, tree_or_items):
        """Splits tree or list on rule, into tuple (pre, match, post)"""
        return cls._find(rule, tree_or_items, 'split')

    @classmethod
    def all(cls, rule, tree_or_items):
        """Returns all items in tree or list, matching rule"""
        return cls._find(rule, tree_or_items, 'all')

    @classmethod
    def flatten(cls, tree_or_items):
        """Flattens tree or list into list"""
        items = []
        for item in cls._as_items(tree_or_items):
            try:
                items += item
            except TypeError:
                items += [item]
        return items

    # -- private methods -----------------------------------------------
    @classmethod
    def _as_items(cls, tree_or_items):
        try:
            return list(tree_or_items)
        except TypeError:
            try:
                return tree_or_items.children
            except AttributeError:
                return [tree_or_items]

    @classmethod
    def _find(cls, rule, tree_or_items, mode):
        split = (mode == 'split')
        first, found = None, []
        if split:
            pre, post = [], []
        elif mode != 'all':
            first = True
        for item in cls._as_items(tree_or_items):
            m = (item_rule(item) == rule)
            if m and first:
                return item
            elif split:
                if not first and m:
                    first = item
                elif not first:
                    pre.append(item)
                else:
                    post.append(item)
            else:
                found.append(item)
        if split:
            return (pre, first, post)
        elif not first:
            return found


class GenericParser:
    """
    Generic parser using lark-parser.

    Meant to be inherited for defining specific parsers (say, ThetaRecordParser for NONMEM API).
    Will do the following:

    1. Lex and parse a 'buffer' (str, not bytes!) using Lark (see Lark_) and (from file in
        ``grammar`` attribute), to build AST.
    2. Transform tree with transformer (see :class:`PreParser`).
    3. Convert to :class:`AttrTree` for tree traversal (attribute access magic)

    Attributes:
        non_empty (list): Insert empty placeholders if missing. Dict map of rule -> (pos, name),
            where a Tree or Token (if uppercase) will be inserted at 'pos' of the children of
            'rule', if none exists.
        buffer (str): Buffer parsed (see :method:`parse`).
        grammar (str): Path to grammar file. Set in subclass.
        root (str): Root of final tree (instance of :class:`AttrTree`)

    .. _Lark:
        https://github.com/lark-parser/lark
    """

    """Children (of rules) to create placeholder for if missing"""
    non_empty = []

    """Pre-parsing transformer"""
    _transformer = None

    """:class:`AttrTree` implementation."""
    AttrTree = AttrTree

    lark_options = dict(
        ambiguity='resolve',
        debug=False,
        keep_all_tokens=True,
        lexer='dynamic',
        parser='earley',
        start='root',
    )

    def __init__(self, buf=None, **lark_options):
        self.lark_options.update(lark_options)
        if self._transformer:
            self.lark_options['transformer'] = self._transformer
        self.root = self.parse(buf)

    def parse(self, buf):
        """
        Parses a buffer, transforms and constructs :class:`AttrTree` object.

        Args:
            buf (str): Buffer to parse.
        """
        self.buffer = buf
        if self.buffer is None:
            return None

        grammar = Path(self.grammar).resolve()
        with open(str(grammar), 'r') as fh:
            self.lark = Lark(fh, **self.lark_options)
            root = self.lark.parse(self.buffer)

        if self.non_empty:
            root = self.insert(root, self.non_empty)
        return self.AttrTree.transform(tree=root)

    @classmethod
    def insert(cls, item, non_empty):
        """
        Inserts missing Tree/Token amongst children (see :attr:`non_empty`).

        Args:
            item: Tree to recurse.
            non_empty: Dict of rule -> (pos, name) tuple.
        """
        if not non_empty or isinstance(item, Token):
            return item
        try:
            pos, name = non_empty[item_rule(item)]
        except KeyError:
            pass
        else:
            if not any(item_rule(child) == name for child in item.children):
                item.children.insert(pos, new_item(name))
        for i, child in enumerate(item.children):
            item.children[i] = cls.insert(child, non_empty)
        return item

    def __str__(self):
        if not self.root:
            return repr(self)
        lines = str(prettyprint.transform(self.root)).splitlines()
        return '\n'.join(lines)
