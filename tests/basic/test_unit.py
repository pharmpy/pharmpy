import pytest
import sympy

from pharmpy.basic.unit import Quantity, Unit


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


def test_quantity():
    x = Quantity(2.5, Unit("mg"))
    assert x.value == 2.5
    assert x.unit == Unit("mg")
    assert repr(x) == "2.5 milligram"

    y = Quantity(1.5, Unit("mg"))
    assert x != y
    assert x == x
    assert x != Unit("mg")

    z = Quantity.parse("-2.5mg")
    assert z.value == -2.5
    assert z.unit == Unit("mg")

    with pytest.raises(ValueError):
        Quantity.parse("mg")
