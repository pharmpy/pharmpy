# -*- encoding: utf-8 -*-

from pathlib import Path

from pysn.psn import GenericParser

grammar_root = Path(__file__).parent.resolve() / 'grammars'
assert grammar_root.is_dir()


class RecordParser(GenericParser):
    def __init__(self, buf):
        self.grammar = grammar_root / self.grammar_filename
        super(RecordParser, self).__init__(buf)


class ProblemRecordParser(RecordParser):
    grammar_filename = 'problem_record.g'
    non_empty = {
        'root': (0, 'text'),
        'text': (0, 'TEXT'),
        'comment': (1, 'TEXT'),
    }


class ThetaRecordParser(RecordParser):
    grammar_filename = 'theta_records.g'
    non_empty = {
        'comment': (1, 'TEXT')
    }
