import textwrap

import pytest

from pharmpy.internals.parse import AttrToken, AttrTree, prettyprint


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
        AttrTree.create('root', {})

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
    __ANON_0 ""
     └─ __ANON_1 ""
    """
    assert_create(out, '', [''])
    out = """
    __ANON_0 "LEAF"
     ├─ __ANON_1 ""
     ├─ __ANON_2 "LEAF"
     └─ __ANON_3 ""
    """
    assert_create(out, '', ['', 'LEAF', ''])

    out = """
    root "LEAF"
     └─ A_TOKEN "LEAF"
    """
    assert_create(out, 'root', dict(A_TOKEN='LEAF'))
    out = """
    __ANON_0 ""
     └─ LEAF ""
    """
    assert_create(out, '', dict(LEAF=''))


def test_tree_create_deep():
    """Test creating trees with deep, actually usable, structures."""

    inp = {
        'firstLEAF': '(leaf #1) ',
        'tree_anons': ['TEXT123 ', "'some string maybe' "],
        'tree': dict(nested_tree=dict(end_LEAF_node='!?#@')),
        'top_tree_again': dict(INTLEAF=123.456),
        'LEAF': 'THE_END',
    }
    out = """
    root "(leaf #1) TEXT123 'some string maybe' !?#@123.456THE_END"
     ├─ firstLEAF "(leaf #1) "
     ├─ tree_anons "TEXT123 'some string maybe' "
     │  ├─ __ANON_1 "TEXT123 "
     │  └─ __ANON_2 "'some string maybe' "
     ├─ tree "!?#@"
     │  └─ nested_tree "!?#@"
     │     └─ end_LEAF_node "!?#@"
     ├─ top_tree_again "123.456"
     │  └─ INTLEAF "123.456"
     └─ LEAF "THE_END"
    """
    assert_create(out, 'root', inp)


def test_tree_create_abuse():
    """Test weird abuse of the creator."""

    # It flattens to anonymous leaves? That.. is neat!
    out = """
    __ANON_0 "LEAF"
     └─ __ANON_1 "LEAF"
    """
    assert_create(out, None, [['LEAF']])
    out = """
    __ANON_0 "1234"
     ├─ __ANON_1 "1"
     ├─ __ANON_2 "2"
     ├─ __ANON_3 "3"
     └─ __ANON_4 "4"
    """
    assert_create(out, False, [[['1'], ['2']], ['3'], '4'])

    # just throwing stuff at the wall.. but it seems to stick!
    od = {'good_tree': dict(LEAF_A=' (^._.^)~ hello! '), 'bad_tree': dict(LEAF_B=None)}
    inp = {
        'item': [od, dict(dict(Btree=[dict(END_LEAF='...THE END')])), dict(_LEAF_=' (nope, here!)')]
    }
    out = """
    root " (^._.^)~ hello! None...THE END (nope, here!)"
     └─ item " (^._.^)~ hello! None...THE END (nope, here!)"
        ├─ good_tree " (^._.^)~ hello! "
        │  └─ LEAF_A " (^._.^)~ hello! "
        ├─ bad_tree "None"
        │  └─ LEAF_B "None"
        ├─ Btree "...THE END"
        │  └─ END_LEAF "...THE END"
        └─ _LEAF_ " (nope, here!)"
    """
    assert_create(out, 'root', inp)
