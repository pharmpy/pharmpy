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

    class PreParser(GenericParser.PreParser):
        def root(self, items):
            if self.first('text', items) is None:
                items.insert(0, self.Tree('text', [self.Token('TEXT', '')]))
            return self.Tree('root', items)

        def comment(self, items):
            if self.first('TEXT', items) is None:
                items.insert(1, self.Token('TEXT', ''))
            return self.Tree('comment', items)


class ThetaRecordParser(RecordParser):
    grammar_filename = 'theta_records.g'

    class PreParser(GenericParser.PreParser):
        def parameter_def(self, items):
            pre, single, post = self.split('single', items)
            if single:
                return self.Tree('theta', pre + single.children + post)

            trees = []
            pre, multi, post = self.split('multiple', items)
            assert multi
            n_thetas = self.first('n_thetas', multi)
            for i in range(int(self.first('INT', n_thetas))):
                trees.append(self.Tree('theta', pre + multi.children + post))
            return trees

        def root(self, items):
            return self.Tree('root', self.flatten(items))
