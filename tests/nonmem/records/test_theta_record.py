# -*- encoding: utf-8 -*-

import pytest


@pytest.fixture(scope='class', autouse=True)
def canonical_name(request):
    """Inject canonical name into all classes"""
    request.cls.canonical_name = 'THETA'


@pytest.fixture(scope='class')
def content_parse(api, request):
    """Inject content parsing and logging method (without record class)"""
    def parse(cls, buf):
        for i, line in enumerate(buf.splitlines()):
            print('%d: %s' % (i, repr(line)))
        tree = api.records.parser.ThetaRecordParser(buf)
        assert tree.root is not None
        print(str(tree) + '\n')
        return tree.root
    request.cls.parse = parse
    yield


@pytest.mark.usefixtures('content_parse', 'random_data')
class TestParser:
    def parse_assert(self, buf, inits=[], comments=[]):
        """Parses buf and assert inits"""
        root = self.parse(buf)
        assert str(root) == buf
        if inits:
            nodes = filter(lambda x: x.rule == 'init', root.tree_walk())
            assert list(map(lambda x: getattr(x, 'NUMERIC'), nodes)) == inits
        if comments:
            nodes = filter(lambda x: x.rule == 'comment', root.tree_walk())
            assert list(map(lambda x: str(getattr(x, 'TEXT')), nodes)) == comments
        return root

    def test_single_inits(self):
        for val in self.data(5).int():
            self.parse_assert(str(val), [val])

        for val in self.data(5).float():
            self.parse_assert(str(val), [val])

    def test_padded_inits(self):
        data = self.data(5)
        for lpad, val, rpad in zip(data.pad(), data.float(), data.pad()):
            self.parse_assert(str(lpad) + str(val) + str(rpad), [val])

        data = self.data(5)
        for lpad, val, rpad in zip(data.pad(nl=True), data.float(), data.pad(nl=True)):
            self.parse_assert(str(lpad) + str(val) + str(rpad), [val])

        data = self.data(5)
        for val in data.pad(nl=True):
            self.parse_assert(str(val), [])

    def test_comments(self):
        self.parse_assert(';', [])
        self.parse_assert(';A', [], ['A'])
        self.parse_assert('; ABC ', [], ['ABC'])
        self.parse_assert(' ; B ; C ', [], ['B ; C'])
        self.parse_assert(' ; B\n; C', [], ['B', 'C'])
        self.parse_assert(' ; B \n\n  ;   A B C\n', [], ['B', 'A B C'])
        self.parse_assert(' \n ; A B C\n ;', [], ['A B C', ''])


@pytest.mark.usefixtures('create_record')
class TestRecordCreate:
    def test_init(self):
        rec = self.create_record('THET')
        rec = self.create_record('THETA 0')
        rec = self.create_record('THETA   12.3 \n\n')
        rec = self.create_record('THETAA 123', fail=True)
        rec = self.create_record('THEETA', fail=True)


@pytest.mark.usefixtures('create_record')
class TestRecordMutability:
    pass
