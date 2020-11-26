"""
Generic parser using lark-parser.

Subclass :class:`GenericParser` (remember to set :attr:`GenericParser.grammar`
to point to your grammar file) to define a powerful parser.
"""
import copy
import re

from lark import Tree
from lark.lexer import Token

from . import prettyprint


class NoSuchRuleException(AttributeError):
    """Rule not found (raised by :class:`AttrTree` for unknown children)."""

    def __init__(self, rule, tree=None):
        try:
            post = ' (%s)' % (repr(tree.rule),)
        except AttributeError:
            post = ''
        super().__init__('no %s child in tree%s' % (repr(rule), post))


def rule_of(item):
    """Rule of a tree or token. Convenience function (will not raise)."""

    try:
        return item.data
    except AttributeError:
        try:
            return item.type
        except AttributeError:
            return ''


def empty_rule(rule):
    """Create empty Tree or Token, depending on (all) upper-case or not."""

    if rule == rule.upper():
        return Token(rule, '')
    return Tree(rule, [])


class AttrToken(Token):
    """Token with attribute access.

    Created by :meth:`AttrTree.transform` from :class:`lark.lexer.Token`.

    Attributes:
        self.rule: Name, in common with :class:`AttrTree`.
        self.eval: Transformed data type, in common with :class:`AttrTree`.

    Can be instantiated with :meth:`.__init__`(rule, content), via :class:`lark.lexer.Token`, or
    alternative constructor :meth:`.transform` (transform object of class
    :class:`lark.lexer.Token`).
    """

    __slots__ = ()

    @classmethod
    def transform(cls, token, **kwargs):
        """Alternative constructor: From Token (with optional overrides)."""
        kwargs = {
            'type_': token.type,
            'value': token.value,
            'pos_in_stream': token.pos_in_stream,
            'line': token.line,
            'column': token.column,
            **kwargs,
        }
        return cls(**kwargs)

    @property
    def rule(self):
        """Rule name (synonymous with 'type')"""
        return self.type

    @rule.setter
    def rule(self, value):
        self.type = value

    @property
    def eval(self):
        """Evaluated value (str, int, float)."""
        if self.type in {'DIGIT', 'INT', 'SIGNED_INT'}:
            return int(self.value)
        elif self.type in {
            'DECIMAL',
            'EXP',
            'FLOAT',
            'NUMBER',
            'NUMERIC',
            'SIGNED_FLOAT',
            'SIGNED_NUMBER',
        }:
            return float(self.value)
        elif self.type == 'NEG_INF':
            return float('-INF')
        elif self.type == 'POS_INF':
            return float('INF')
        else:
            return str(self.value)

    def replace(self, value):
        """Returns copy (same rule), but with content replaced."""
        return self.transform(token=self, value=value)

    def __deepcopy__(self, memo):
        return AttrToken(self.type, self.value, self.pos_in_stream, self.line, self.column)

    def __repr__(self):
        return '%s(%s, %s)' % (self.__class__.__name__, repr(self.rule), repr(self.value))

    def __hash__(self):
        return hash(repr(self))

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __str__(self):
        return str(self.value)


class AttrTree(Tree):
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

        Handling of missing names (e.g. lists):
            1. Tree: Children are moved to parent.
            2. Token: Native naming scheme, __ANON_%d, of Lark is used.

        Args:
            items: Child (tree) nodes or content (token) nodes.
            rule: Name of root tree (__ANON_0 if False).
            _anon_count: Internal. Anonymous numbering offset.
            _list: Internal. Recursion state. Drop 'rule' & return list of children, which are
                orphaned if name is False.

        Raises:
            TypeError: 'items' not iterable or instance of 'str' (only leaves shall contain 'str').
            ValueError: 'items' empty (trees can't be empty).

        .. note:: Please follow convention of all lower/upper case for trees/tokens.
        """

        def non_iterable(rule, items):
            raise TypeError(
                '%s object is not iterable (of children for tree %s)'
                % repr(items.__class__.__name__),
                repr(rule),
            )

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
            names = [None] * length
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
                    if not name:
                        _anon_count += 1
                        name = '__ANON_%d' % (_anon_count,)
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
        """All rules of (immediate) children."""
        return [node.rule for node in self.children]

    @property
    def eval(self):
        """Evaluated value (self)."""
        return self

    def find(self, rule):
        """Returns first child matching 'rule', or None."""
        return next((child for child in self.children if child.rule == rule), None)

    def all(self, rule):
        """Returns all children matching rule, or []."""
        return list(filter(lambda child: child.rule == rule, self.children))

    def set(self, rule, value):
        """Sets first child matching rule. Raises if none."""
        for i, child in enumerate(self.children):
            if child.rule == rule:
                self.children[i] = value
                return
        raise NoSuchRuleException(rule, self)

    def partition(self, rule):
        """Partition children into (head, item, tail).

        Search for child item 'rule' and return the part before it (head), the item, and the part
        after it (tail). If 'rule' is not found, return (children, [], [])."""
        head, item, tail = [], [], []
        for node in self.children:
            if not item and node.rule == rule:
                item += [node]
            elif not item:
                head += [node]
            else:
                tail += [node]
        return (head, item, tail)

    def tree_walk(self):
        """Generator for iterating depth-first (i.e. parse order) over children."""
        for child in self.children:
            yield child
            try:
                yield from child.tree_walk()
            except AttributeError:
                continue

    def remove(self, rule):
        """Remove all children with rule. Not recursively"""
        new_children = []
        for child in self.children:
            if child.rule != rule:
                new_children.append(child)
        self.children = new_children

    def remove_node(self, node):
        new_children = []
        comment_flag = False
        for child in self.children:
            if child.rule == 'COMMENT' or child.rule == 'NEWLINE':
                if not comment_flag:
                    new_children.append(child)
                else:
                    if str(child).startswith('\n\n'):
                        new_children.append(child.replace(str(child)[1:]))
                    comment_flag = False
            elif str(child.eval) != str(node.eval):
                new_children.append(child)
                if child.rule == 'statement':
                    comment_flag = False
            else:
                comment_flag = True
        self.children = new_children

    @staticmethod
    def _newline_node():
        return AttrToken('WS_ALL', '\n')

    def _clean_ws(self, new_children):
        new_children_clean = []
        types_of_newline = ['WS_ALL', 'NEWLINE']
        last_index = len(new_children) - 1

        prev_rule = None

        for i, child in enumerate(new_children):
            if child.rule in types_of_newline:
                if prev_rule in types_of_newline or prev_rule == 'verbatim':
                    continue
                if i == last_index:
                    continue
            if re.search('\n{2,}', str(child)):
                new_children_clean.append(self._newline_node())
            else:
                new_children_clean.append(child)

            prev_rule = child.rule

        return new_children_clean

    def add_node(self, node, following_node=None, comment=False):
        new_children = copy.deepcopy(self.children)

        if comment:
            new_children = self._clean_ws(new_children)

        if following_node is None:
            if not comment:
                new_children.append(self._newline_node())
            new_children.append(node)
        else:
            index = self.children.index(following_node)
            new_children.insert(index, node)

            newline_node = copy.deepcopy(new_children[index - 1])
            new_children.insert(index + 1, newline_node)

        new_children_clean = self._clean_ws(new_children)
        self.children = new_children_clean

    def add_comment_node(self, comment, adjacent_node=None):
        comment_node = AttrToken('COMMENT', f' ; {comment}')
        self.add_node(comment_node, adjacent_node, comment=True)

    def add_newline_node(self):
        self.children.append(self._newline_node())

    def get_last_node(self):
        for node in self.children:
            last_node = node
        return last_node

    @property
    def tokens(self):
        """All tokens as flattened list."""
        items = []
        for item in self.children:
            try:
                items += item.tokens
            except AttributeError:
                items += [item]
        return items

    @property
    def debug(self, *args, **kwargs):
        """Debug formatted tree structure."""
        return str(prettyprint.transform(self, *args, **kwargs))

    def treeprint(self, indent=''):
        """Prints debug formatted tree structure."""
        print(self.debug)

    def set_child(self, attr, value):
        self.find(attr).value = value

    # -- private methods -----------------------------------------------
    def __len__(self):
        return len(self.children)

    def __setattr__(self, attr, value):
        if attr in ['data', 'children', '_meta']:
            object.__setattr__(self, attr, value)
        else:
            self.set_child(attr, value)

    def __getattr__(self, attr):
        if attr in ['data', 'children', '_meta']:
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


class GenericParser:
    """
    Generic parser using lark-parser.

    Inherit to define a parser, say ThetaRecordParser for NONMEM, with the workflow:

    1. Lex and parse a 'buffer' using Lark_ (from :attr:`grammar` file). Builds AST.
    2. Shape AST by options (see :attr:`non_empty`).
    3. Convert to :class:`AttrTree` for convenient traversal (attribute access).

    Attributes:
        self.non_empty: Insert empty placeholders if missing. Dict of rule -> (pos, name), where a
            Tree or Token (if uppercase) will be inserted at 'pos' of the children of 'rule', if
            none exists.
        self.buffer: Buffer parsed by :meth:`parse`.
        self.grammar: Path to grammar file.
        self.root: Root of final tree. Instance of :class:`AttrTree`.

    .. _Lark:
        https://github.com/lark-parser/lark
    """

    """Children to create if missing"""
    non_empty = []

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
        if self.non_empty:
            root = self.insert(root, self.non_empty)
        return self.AttrTree.transform(tree=root)

    @classmethod
    def insert(cls, item, non_empty):
        """Inserts missing Tree/Token amongst children (see :attr:`non_empty`).

        Args:
            item: Tree to recurse.
            non_empty: Dict of rule -> (pos, name) tuple.
        """
        if not non_empty or isinstance(item, Token):
            return item
        for d in non_empty:
            try:
                pos, name = d[rule_of(item)]
            except KeyError:
                pass
            else:
                if not any(rule_of(child) == name for child in item.children):
                    item.children.insert(pos, empty_rule(name))
        for i, child in enumerate(item.children):
            item.children[i] = cls.insert(child, non_empty)
        return item

    def __str__(self):
        if not self.root:
            return repr(self)
        lines = str(prettyprint.transform(self.root)).splitlines()
        return '\n'.join(lines)


def remove_token_and_space(tree, rule, recursive=False):
    """Remove all tokens with rule and any WS before it"""
    new_nodes = []
    for node in tree.children:
        if node.rule == rule:
            if len(new_nodes) > 0 and new_nodes[-1].rule == 'WS':
                new_nodes.pop()
        else:
            new_nodes.append(node)
    tree.children = new_nodes
    if recursive:
        for node in tree.children:
            if hasattr(node, 'children'):
                remove_token_and_space(node, rule, recursive=True)


def insert_before_or_at_end(tree, rule, nodes):
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
    tree.children = kept


def insert_after(tree, rule, nodes):
    """Insert nodes after rule"""
    kept = []
    for node in tree.children:
        kept.append(node)
        if node.rule == rule:
            kept.extend(nodes)
    tree.children = kept
