from pathlib import Path

from lark import Lark, Tree, Visitor

from pharmpy.internals.parse import GenericParser, InsertMissing, with_ignored_tokens

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
    grammar_options = dict(
        propagate_positions=True,
    )
    post_process = (with_ignored_tokens,)


@install_grammar
class SimulationRecordParser(RecordParser):
    grammar_filename = 'simulation_record.lark'
    grammar_options = dict(
        propagate_positions=True,
    )
    post_process = (with_ignored_tokens,)


@install_grammar
class ProblemRecordParser(RecordParser):
    grammar_filename = 'problem_record.lark'
    post_process = (
        InsertMissing(
            (
                {'root': ((0, 'raw_title'),)},
                {'raw_title': ((0, 'REST_OF_LINE'),)},
            )
        ),
    )


class InitOrLow(Visitor):
    def theta(self, tree):
        assert tree.data == 'theta'
        other = {'init', 'up'}
        subtrees = list(filter(lambda child: isinstance(child, Tree), tree.children))
        is_low = any(tree.data in other for tree in subtrees)
        for tree in subtrees:
            if tree.data == 'init_or_low':
                tree.data = 'low' if is_low else 'init'


@install_grammar
class ThetaRecordParser(RecordParser):
    grammar_filename = 'theta_record.lark'
    grammar_options = dict(
        propagate_positions=True,
    )
    post_process = (
        InsertMissing(({'comment': ((1, 'COMMENT'),)},)),
        InitOrLow(),
        with_ignored_tokens,
    )


@install_grammar
class OmegaRecordParser(RecordParser):
    grammar_filename = 'omega_record.lark'
    grammar_options = dict(
        propagate_positions=True,
    )
    post_process = (InsertMissing(({'comment': ((1, 'COMMENT'),)},)), with_ignored_tokens)


@install_grammar
class OptionRecordParser(RecordParser):
    grammar_filename = 'option_record.lark'
    grammar_options = dict(
        propagate_positions=True,
    )
    post_process = (with_ignored_tokens,)


@install_grammar
class DataRecordParser(RecordParser):
    grammar_filename = 'data_record.lark'
    grammar_options = dict(
        propagate_positions=True,
    )
    post_process = (with_ignored_tokens,)


@install_grammar
class CodeRecordParser(RecordParser):
    grammar_filename = 'code_record.lark'
    grammar_options = dict(
        propagate_positions=True,
    )
    post_process = (with_ignored_tokens,)
