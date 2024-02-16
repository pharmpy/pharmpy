import pytest

from pharmpy.basic import Expr


def test_init():
    expr = Expr("1")
    assert expr == 1
    expr2 = Expr(expr)
    assert expr == expr2


def test_symbol():
    expr = Expr.symbol("CL")
    assert expr.name == "CL"
    expr = Expr.integer(1)
    with pytest.raises(ValueError):
        expr.name
    expr = Expr.function("f", "t")
    assert expr.name == "f"


def test_piecewise():
    expr = Expr.piecewise(("1", "x > 0"), ("2", "x < 0"))
    assert expr.is_piecewise()
    assert expr.args[0][0] == Expr.integer(1)
    assert expr.args[1][0] == Expr.integer(2)


def test_integer():
    expr = Expr.integer(23)
    assert 23 == expr


@pytest.mark.parametrize(
    'expr,alternative',
    [
        (Expr.symbol("CL") + 1, 1 + Expr.symbol("CL")),
        (Expr.symbol("CL") * 2, 2 * Expr.symbol("CL")),
        (Expr.symbol("CL") / 2, (2 * Expr.symbol("CL")) / 4),
        (2 / Expr.symbol("CL"), 4 / (2 * Expr.symbol("CL"))),
        (Expr.symbol("CL") - 2, (2 + Expr.symbol("CL")) - 4),
        (2 - Expr.symbol("CL"), (4 - Expr.symbol("CL")) - 2),
        (Expr.integer(2) ** 2, Expr.integer(4) ** 1),
        (2 ** Expr.integer(2), 4 ** Expr.integer(1)),
        (-Expr.integer(2), Expr.integer(-2)),
        (+Expr.integer(2), Expr.integer(2)),
    ],
)
def test_operators(expr, alternative):
    assert expr == alternative
