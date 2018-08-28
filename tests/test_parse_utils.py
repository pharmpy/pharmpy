
import textwrap

import pytest

from pysn.parse_utils.generic import AttrToken
from pysn.parse_utils.generic import AttrTree
from pysn.parse_utils import prettyprint


def assert_create(expect, *args, **kwargs):
    """Help asserter for AttrTree.create tests."""

    obj = AttrTree.create(*args, **kwargs)
    assert isinstance(obj, AttrTree)
    for item in obj.children:
        if 'tree' in item.rule:
            assert isinstance(item, AttrTree)
        elif 'LEAF' in item.rule:
            assert isinstance(item, AttrToken)
    pretty = str(prettyprint.transform(obj, content=True))
    expect = textwrap.dedent(expect).strip()
    assert pretty == expect


def test_tree_create_bad():
    """Test expected raising for planned non-legality."""

    # tree can't be empty (that's a leaf, i.e. a token)
    with pytest.raises(TypeError):  # "not iterable"
        AttrTree.create('root', None)
    with pytest.raises(ValueError):
        AttrTree.create('root', [])
    with pytest.raises(ValueError):
        AttrTree.create('root', dict())

    # tokens must have str as only "child", but trees can't have str as "children"
    with pytest.raises(TypeError):  # "not iterable"
        AttrTree.create('root', 'str is only for tokens')


def test_tree_create_shallow():
    """Test creating shallow trees."""

    out = """
    tree "content"
     └─ __ANON_1 "content"
    """
    assert_create(out, 'tree', ['content'])
    out = """
    __ANON_1 ""
     └─ __ANON_2 ""
    """
    assert_create(out, '', [''])
    out = """
    __ANON_1 "LEAF"
     ├─ __ANON_2 ""
     ├─ __ANON_3 "LEAF"
     └─ __ANON_4 ""
    """
    assert_create(out, '', ['', 'LEAF', ''])

    out = """
    root "LEAF"
     └─ A_TOKEN "LEAF"
    """
    assert_create(out, 'root', dict(A_TOKEN='LEAF'))
    out = """
    __ANON_1 ""
     └─ LEAF ""
    """
    assert_create(out, '', dict(LEAF=''))


def test_tree_create_deep():
    """Test creating trees with deep, actually usable, structures."""

    out = """
    root "(leaf #1) TEXT123 'some string maybe' !?#@123.456THE_END"
     ├─ firstLEAF "(leaf #1) "
     ├─ LEAF_copy "TEXT123 "
     ├─ LEAF_copy "'some string maybe' "
     ├─ tree "!?#@"
     │  └─ nested_tree "!?#@"
     │     └─ end_LEAF_node "!?#@"
     ├─ top_tree_again "123.456"
     │  └─ INTLEAF "123.456"
     └─ LEAF "THE_END"
    """
    assert_create(out, 'root',
                  dict(firstLEAF='(leaf #1) ', LEAF_copy=['TEXT123 ', "'some string maybe' "],
                       tree=dict(nested_tree=dict(end_LEAF_node='!?#@')),
                       top_tree_again=dict(INTLEAF=123.456), LEAF='THE_END'))


def test_tree_create_abuse():
    """Test weird abuse of the creator."""

    # It flattens to anonymous leaves? That.. is neat!
    out = """
    __ANON_1 "LEAF"
     └─ __ANON_2 "LEAF"
    """
    assert_create(out, '', [['LEAF']])
    out = """
    __ANON_1 "1234"
     ├─ __ANON_2 "1"
     ├─ __ANON_2 "2"
     ├─ __ANON_3 "3"
     └─ __ANON_4 "4"
    """
    assert_create(out, '', [[['1'], ['2']], ['3'], '4'])

    # just throwing stuff at the wall.. but it seems to stick!
    out = """
    __ANON_1 " (^._.^)~ hello! None...THE END (nope, here!)"
     ├─ item " (^._.^)~ hello! None"
     │  ├─ good_tree " (^._.^)~ hello! "
     │  │  └─ LEAF_A " (^._.^)~ hello! "
     │  └─ bad_tree "None"
     │     └─ LEAF_B "None"
     ├─ item "...THE END"
     │  └─ Btree "...THE END"
     │     └─ END_LEAF "...THE END"
     └─ item " (nope, here!)"
        └─ _LEAF_ " (nope, here!)"
    """
    assert_create(out, '',
                  dict(item=[dict(good_tree=dict(LEAF_A=' (^._.^)~ hello! '),
                                  bad_tree=dict(LEAF_B=None)),
                             dict(dict(Btree=[dict(END_LEAF='...THE END')])),
                             dict(_LEAF_=' (nope, here!)')]))
