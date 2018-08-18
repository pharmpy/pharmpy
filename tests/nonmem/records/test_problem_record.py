# -*- encoding: utf-8 -*-

import pytest


@pytest.fixture(scope='class', autouse=True)
def canonical_name(request):
    """Inject canonical name into all classes"""
    request.cls.canonical_name = 'PROBLEM'


@pytest.fixture(scope='class')
def content_parse(api, request):
    """Inject content parsing and logging method (without record class)"""
    def parse(cls, buf):
        for i, line in enumerate(buf.splitlines()):
            print('%d: %s' % (i, repr(line)))
        tree = api.records.parser.ProblemRecordParser(buf)
        assert tree.root is not None
        print(str(tree) + '\n')
        return tree.root
    request.cls.parse = parse
    yield


@pytest.mark.usefixtures('content_parse')
class TestParser:
    def parse_assert(self, buf, text, comments=[]):
        """Parses buf and assert text 'text' and comments 'comments'"""
        root = self.parse(buf)
        assert str(root) == buf
        assert str(root.text) == text
        nodes = filter(lambda x: x.rule == 'comment', root.tree_walk())
        assert list(map(lambda x: str(getattr(x, 'TEXT')), nodes)) == comments

    def test_empties(self):
        self.parse_assert('', '')
        self.parse_assert(' ', '')
        self.parse_assert('\n', '')
        self.parse_assert(' \n ', '')
        self.parse_assert(' \n \n', '')

    def test_names(self):
        self.parse_assert('A', 'A')
        self.parse_assert(' ABC ', 'ABC')
        self.parse_assert(' A ; B ; C ', 'A ; B ; C')
        self.parse_assert(' A ; B \n', 'A ; B')
        self.parse_assert(' A ; B \n\n  ; some comment\n', 'A ; B', ['some comment'])
        self.parse_assert(' A \n ; A B ; D \n ; ', 'A', ['A B ; D', ''])


@pytest.mark.usefixtures('create_record')
class TestRecordCreate:
    def test_init(self):
        rec = self.create_record('PROB')
        rec = self.create_record('PROBLEM ABC')
        assert rec.string == 'ABC'
        rec = self.create_record('PROBLEM   A;BC \n\n')
        assert rec.string == 'A;BC'
        rec = self.create_record('PROBLEMA ABC', fail=True)
        rec = self.create_record('PROBLEE', fail=True)


@pytest.mark.usefixtures('create_record')
class TestRecordMutability:
    def test_string(self):
        rec = self.create_record('PROB')
        assert rec.string == ''
        rec.string = 'PHENO  MODEL'
        assert rec.string == 'PHENO  MODEL'
