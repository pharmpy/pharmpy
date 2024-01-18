import pytest

from pharmpy.basic import Expr


def test_symbol():
    expr = Expr.symbol("CL")
    assert expr.name == "CL"
    expr = Expr.integer(1)
    with pytest.raises(ValueError):
        expr.name


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
