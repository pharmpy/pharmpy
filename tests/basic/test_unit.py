import pytest
import sympy

from pharmpy.basic.unit import Unit


def test_init():
    unit1 = Unit('x')
    assert repr(unit1) == 'x'

    unit2 = Unit(unit1)
    assert unit1 == unit2


def test_unitless():
    assert Unit.unitless() == Unit(1)


def test_serialize():
    unit = Unit('x')
    unit_ser = unit.serialize()
    assert unit_ser == "Symbol('x')"
    assert Unit.deserialize(unit_ser) == unit


@pytest.mark.parametrize(
    'expr, unicode_ref',
    [('x', 'x'), ('x*y', 'x⋅y'), ('x^-1', 'x⁻¹'), ('x^0.01', 'x^0.01'), ('milliliter', 'ml')],
)
def test_unicode(expr, unicode_ref):
    assert Unit(expr).unicode() == unicode_ref


def test_sympify():
    unit = Unit('x')
    unit_sympy = unit._sympify_()
    assert unit_sympy == sympy.Symbol('x')
