#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

from pathlib import Path

# TODO: remove debug sys.path... when done debugging, of course
if True:
    from os import curdir
    from pathlib import Path
    import sys
    src = Path(__file__).resolve().parent.parent.parent.parent.parent
    sys.path.insert(0, str(src))
from pysn.psn import GenericParser


grammar_root = Path(__file__).parent.resolve() / 'grammars'
assert grammar_root.is_dir()


class RecordParser(GenericParser):
    def __init__(self, buf):
        self.grammar = grammar_root / self.grammar_filename
        super(RecordParser, self).__init__(buf)


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


if __name__ == '__main__':
    from textwrap import dedent
    testdata = ['''
        123 FIX ; TEST
        (456)x3 ; TEST
    ''', '''
      (,.1)
      (3)x 2
      (0,0.00469307) ; CL
      (0,1.00916) ; V
      (-.99,.1)
    ''', '''

      (0, 0.00469307) ; CL

      12.3 ; V
      ; comment
        ; comment
      (3) x 5

    ''']
    # for data in testdata:
    data = testdata[1]
    buf = dedent(data)
    parser = ThetaRecordParser(buf)
    root = parser.root

    # import pdb; pdb.set_trace()
    print(root)
    print('\n')
    root.treeprint('full')

    thetas = root.all('theta')
    print('thetas[1]:', repr(thetas[1]))
    print('thetas[0].init:', repr(str(thetas[0].init)))
    print('thetas[0].init.NUMERIC:', repr(thetas[0].init.NUMERIC))
    # print(root.parameter.init.NUMERIC)
