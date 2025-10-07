import pytest

from pharmpy.mfl.features import Ref


def test_init():
    ref1 = Ref('IIV')
    assert ref1.name == 'IIV'
    assert repr(ref1) == '@IIV'
    ref2 = Ref('iiv')
    assert ref2.name == 'IIV'
    assert repr(ref2) == '@IIV'


def test_eq():
    ref1 = Ref('IIV')
    assert ref1.name == 'IIV'
    ref2 = Ref('iiv')
    assert ref2.name == 'IIV'
    assert ref1 == ref2
    ref3 = Ref('x')
    assert ref3.name == 'X'
    assert ref1 != ref3

    assert ref1 != 1


def test_lt():
    ref1 = Ref('a')
    ref2 = Ref('b')
    assert ref1 < ref2

    with pytest.raises(TypeError):
        ref1 < 1
