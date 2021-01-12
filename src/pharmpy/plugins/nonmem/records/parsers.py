from pathlib import Path

from lark import Lark

from pharmpy.parse_utils import GenericParser

grammar_root = Path(__file__).parent.resolve() / 'grammars'


def install_grammar(cls):
    grammar = Path(grammar_root / cls.grammar_filename).resolve()
    with open(str(grammar), 'r') as fh:
        cls.lark = Lark(fh, **GenericParser.lark_options)
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


@install_grammar
class ThetaRecordParser(RecordParser):
    grammar_filename = 'theta_record.lark'
    non_empty = [
        {'comment': (1, 'COMMENT')},
    ]


@install_grammar
class OmegaRecordParser(RecordParser):
    grammar_filename = 'omega_record.lark'
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
