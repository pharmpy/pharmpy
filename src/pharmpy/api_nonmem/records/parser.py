# -*- encoding: utf-8 -*-

from pathlib import Path

from pharmpy.parse_utils import GenericParser

grammar_root = Path(__file__).parent.resolve() / 'grammars'
assert grammar_root.is_dir()


class RecordParser(GenericParser):
    def __init__(self, buf):
        self.grammar = grammar_root / self.grammar_filename
        super(RecordParser, self).__init__(buf)


class ProblemRecordParser(RecordParser):
    grammar_filename = 'problem_record.lark'
    non_empty = {
        'root': (0, 'text'),
        'text': (0, 'TEXT'),
        'comment': (1, 'COMMENT'),
    }


class ThetaRecordParser(RecordParser):
    grammar_filename = 'theta_record.lark'
    non_empty = {
        'comment': (1, 'COMMENT')
    }


class OmegaRecordParser(RecordParser):
    grammar_filename = 'omega_record.lark'
    non_empty = {
        'comment': (1, 'COMMENT')
    }


class OptionRecordParser(RecordParser):
    grammar_filename = 'option_record.lark'


class DataRecordParser(RecordParser):
    grammar_filename = 'data_record.lark'
