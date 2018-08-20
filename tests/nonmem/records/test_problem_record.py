# -*- encoding: utf-8 -*-

import pytest


@pytest.fixture
def parse_assert(api):
    """Returns function for parsing with ProblemRecordParser. Basic logging/asserts."""

    def func(buf, text=None, comments=[]):
        for i, line in enumerate(buf.splitlines()):
            print('%d: %s' % (i, repr(line)))
        tree = api.records.parser.ProblemRecordParser(buf)
        assert tree.root is not None
        print(str(tree) + '\n')
        root = tree.root

        assert str(root) == buf
        if text:
            assert str(root.text) == text
        if comments:
            nodes = filter(lambda x: x.rule == 'comment', root.tree_walk())
            assert list(map(lambda x: str(getattr(x, 'TEXT')), nodes)) == comments
        return root

    return func


# -- ONLY PARSER -----------------------------------------------------------------------------------


@pytest.mark.usefixtures('parse_assert')
@pytest.mark.parametrize('buf,text', [
    ('', ''),
    (' ', ''),
    ('\n', ''),
    (' \n ', ''),
    (' \n \n', ''),
])
def test_empty(parse_assert, buf, text):
    parse_assert(buf, text)


@pytest.mark.usefixtures('parse_assert')
@pytest.mark.parametrize('buf,text,comments', [
    ('A', 'A', []),
    (' ABC ', 'ABC', []),
    (' A ; B ; C ', 'A ; B ; C', []),
    (' A ; B \n', 'A ; B', []),
    (' A ; B \n\n  ; some comment\n', 'A ; B', ['some comment']),
    (' A \n ; A B ; D \n ; ', 'A', ['A B ; D', '']),
])
def test_text_comments(parse_assert, buf, text, comments):
    parse_assert(buf, text, comments)


# -- RECORD CLASS ----------------------------------------------------------------------------------


@pytest.mark.usefixtures('create_record')
@pytest.mark.parametrize('buf,string', [
    ('PROB', ''),
    ('PROBLEM ABC', 'ABC'),
    ('PROBLEM   A;BC \n\n', 'A;BC'),
])
def test_create(create_record, buf, string):
    rec = create_record(buf)
    assert rec.string == string


@pytest.mark.usefixtures('create_record')
@pytest.mark.parametrize('buf,string,new_string', [
    ('PROB', '', 'PHENO  MODEL'),
])
def test_modify_string(create_record, buf, string, new_string):
    rec = create_record(buf)
    assert rec.string == ''
    rec.string = new_string
    assert rec.string == new_string
