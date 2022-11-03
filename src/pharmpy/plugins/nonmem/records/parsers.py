from pathlib import Path
from typing import Iterator, List, Tuple, Union

from lark import Lark, Token, Transformer, Tree, Visitor
from lark.tree import Meta

from pharmpy.internals.parse import GenericParser

grammar_root = Path(__file__).resolve().parent / 'grammars'


def install_grammar(cls):
    grammar = Path(grammar_root / cls.grammar_filename).resolve()
    with open(str(grammar), 'r') as fh:
        cls.lark = Lark(fh, **{**GenericParser.lark_options, **getattr(cls, 'grammar_options', {})})
    return cls


class RecordParser(GenericParser):
    pass


@install_grammar
class AbbreviatedRecordParser(RecordParser):
    grammar_filename = 'abbreviated_record.lark'


@install_grammar
class SimulationRecordParser(RecordParser):
    grammar_filename = 'simulation_record.lark'


@install_grammar
class ProblemRecordParser(RecordParser):
    grammar_filename = 'problem_record.lark'
    non_empty = [
        {'root': (0, 'raw_title')},
        {'raw_title': (0, 'REST_OF_LINE')},
    ]


class InitOrLow(Visitor):
    def theta(self, tree):
        assert tree.data == 'theta'
        other = {'init', 'up'}
        subtrees = list(filter(lambda child: isinstance(child, Tree), tree.children))
        is_low = any(tree.data in other for tree in subtrees)
        for tree in subtrees:
            if tree.data == 'init_or_low':
                tree.data = 'low' if is_low else 'init'


WS = {' ', '\x00', '\t'}


def _tokenize_ignored_characters(s: str, i: int, j: int):
    # TODO propagate line/column information
    head = i
    while i < j:
        first = s[head]
        head += 1
        if first in WS:
            while head < j and s[head] in WS:
                head += 1

            yield Token('WS', s[i:head], start_pos=i, end_pos=head)

        elif first == ';':
            while head < j and s[head] != '\n' and s[head] != '\r':
                head += 1

            yield Token('COMMENT', s[i:head], start_pos=i, end_pos=head)

        elif first == '\r':
            assert head < j and s[head] == '\n'
            head += 1

            yield Token('NEWLINE', s[i:head], start_pos=i, end_pos=head)

        else:
            assert first == '\n'
            yield Token('NEWLINE', s[i:head], start_pos=i, end_pos=head)

        i = head


def _item_range(x: Union[Tree, Token]) -> Tuple[int, int]:
    if isinstance(x, Tree):
        i = x.meta.start_pos
        j = x.meta.end_pos
    else:
        i = x.start_pos
        j = x.end_pos

    assert isinstance(i, int)
    assert isinstance(j, int)
    return (i, j)


def _interleave_ignored(source: str, it: Iterator[Union[Tree, Token]]):
    x = next(it)
    yield x

    i = _item_range(x)[1]

    x = next(it)
    while True:
        j, k = _item_range(x)

        if i < j:
            yield from _tokenize_ignored_characters(source, i, j)

        yield x

        try:
            x = next(it)
        except StopIteration:
            break

        i = k


def interleave_ignored(source: str, children: List[Union[Tree, Token]]):
    if len(children) < 2:
        return children
    else:
        return list(_interleave_ignored(source, iter(children)))


class InterleaveIgnored(Transformer):
    def __init__(self, source: str):
        self.source = source

    def __default__(self, data, children, meta):
        new_children = interleave_ignored(self.source, children)
        return Tree(data, new_children, meta)


def insert_ignored(source, tree):
    new_tree = InterleaveIgnored(source).transform(tree)
    final_meta = Meta()
    # TODO propagate line/column information
    final_meta.start_pos = 0
    final_meta.end_pos = len(source)
    final_meta.empty = False
    final_children = (
        list(_tokenize_ignored_characters(source, 0, new_tree.meta.start_pos))
        + new_tree.children
        + list(_tokenize_ignored_characters(source, new_tree.meta.end_pos, len(source)))
    )
    final_tree = Tree(new_tree.data, final_children, final_meta)
    return final_tree


@install_grammar
class ThetaRecordParser(RecordParser):
    grammar_filename = 'theta_record.lark'
    grammar_options = dict(
        propagate_positions=True,
    )
    non_empty = [
        {'comment': (1, 'COMMENT')},
    ]
    post_process = [InitOrLow(), insert_ignored]


@install_grammar
class OmegaRecordParser(RecordParser):
    grammar_filename = 'omega_record.lark'
    grammar_options = dict(
        parser='earley',
        lexer='dynamic',
        ambiguity='resolve',
    )
    non_empty = [
        {'comment': (1, 'COMMENT')},
    ]


@install_grammar
class OptionRecordParser(RecordParser):
    grammar_filename = 'option_record.lark'


@install_grammar
class DataRecordParser(RecordParser):
    grammar_filename = 'data_record.lark'


@install_grammar
class CodeRecordParser(RecordParser):
    grammar_filename = 'code_record.lark'
    grammar_options = dict(
        parser='earley',
        lexer='dynamic',
        ambiguity='resolve',
    )
