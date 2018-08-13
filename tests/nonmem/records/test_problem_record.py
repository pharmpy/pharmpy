
import pytest


@pytest.fixture
def parse_assert(nm_api):
    def parse(buf, text, comments=[]):
        tree = nm_api.records.parser.ProblemRecordParser(buf)
        rbuf = repr(buf)
        print(str(tree) + '\n')
        root = tree.root
        assert str(root) == buf
        assert str(root.text) == text
        nodes = filter(lambda x: x.rule == 'comment', root.tree_walk())
        assert list(map(lambda x: str(getattr(x, 'TEXT')), nodes)) == comments
    return parse


def test_empties(parse_assert):
    parse_assert('', '')
    parse_assert(' ', '')
    parse_assert('\n', '')
    parse_assert(' \n ', '')
    parse_assert(' \n \n', '')


def test_names(parse_assert):
    parse_assert('A', 'A')
    parse_assert(' ABC ', 'ABC')
    parse_assert(' A ; B ; C ', 'A ; B ; C')
    parse_assert(' A ; B \n', 'A ; B')
    parse_assert(' A ; B \n\n  ; some comment\n', 'A ; B', ['some comment'])
    parse_assert(' A \n ; A B ; D \n ; ', 'A', ['A B ; D', ''])
