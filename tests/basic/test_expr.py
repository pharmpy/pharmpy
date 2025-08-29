import pytest
import sympy

from pharmpy.basic import BooleanExpr
from pharmpy.basic.expr import Expr, ExprPrinter


def test_init_expr():
    expr = Expr("1")
    assert expr == 1
    expr2 = Expr(expr)
    assert expr == expr2
    with pytest.raises(TypeError):
        Expr(Exception)


def test_symbol():
    expr = Expr.symbol("CL")
    assert expr.name == "CL"
    expr = Expr.integer(1)
    with pytest.raises(ValueError):
        expr.name
    expr = Expr.function("f", "t")
    assert expr.name == "f"
    expr = Expr.function('f', ('x', 'y'))
    assert expr.name == "f"
    assert Expr.log(Expr.symbol("x")).name == 'log'


def test_piecewise():
    expr = Expr.piecewise(("1", "x > 0"), ("2", "x < 0"))
    assert expr.is_piecewise()

    assert expr.piecewise_args[0][0] == Expr.integer(1)
    assert expr.piecewise_args[1][0] == Expr.integer(2)

    expr = Expr.symbol("x")
    with pytest.raises(ValueError):
        expr.piecewise_args


def test_piecewise_fold():
    expr1 = Expr.piecewise(("1", "x > 0"), ("2", "x < 0"))
    expr2 = Expr.piecewise((expr1, "y > 0"))
    expr_fold = expr2.piecewise_fold()
    assert expr2 != expr_fold
    assert expr2.piecewise_args[0][0].is_piecewise()
    assert not expr_fold.piecewise_args[0][0].is_piecewise()


def test_expand():
    assert Expr('x*(y + z)').expand() == Expr('x*y + x*z')


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


@pytest.mark.parametrize(
    'expr, method_type, ref',
    [
        (Expr(1), float, 1.0),
        (Expr(1), int, 1),
        (Expr(-1), abs, 1),
    ],
)
def test_type_conversion(expr, method_type, ref):
    assert method_type(expr) == ref


def test_init_boolean_expr():
    expr1 = BooleanExpr(sympy.Eq(Expr('x'), Expr('y')))
    expr2 = BooleanExpr(expr1)
    assert expr1 == expr2
    expr3 = BooleanExpr(sympy.Gt(Expr('x'), Expr(0)))
    assert expr3.lhs == Expr('x')
    expr4 = BooleanExpr('x & y')
    assert expr4.free_symbols == {Expr.symbol('x'), Expr.symbol('y')}
    assert expr4 == BooleanExpr(expr4)


def test_true():
    expr = BooleanExpr.true()
    assert expr._sympy_() == sympy.true


def test_boolean_expr_free_symbols():
    expr1 = BooleanExpr(sympy.Eq(Expr('x'), Expr('y')))
    assert expr1.free_symbols == {Expr.symbol('x'), Expr.symbol('y')}
    expr2 = BooleanExpr(sympy.Gt(Expr('x'), Expr(0)))
    assert expr2.free_symbols == {Expr.symbol('x')}
    expr3 = BooleanExpr('x & y')
    assert expr3.free_symbols == {Expr.symbol('x'), Expr.symbol('y')}


def test_boolean_expr_args():
    expr1 = BooleanExpr(sympy.Eq(Expr('x'), Expr('y')))
    assert expr1.args[0] == Expr.symbol('x')
    assert expr1.args[1] == Expr.symbol('y')
    expr2 = BooleanExpr(sympy.Gt(Expr('x'), Expr(0)))
    assert expr2.args[0] == Expr.symbol('x')
    assert expr2.args[1] == Expr(0)
    expr3 = BooleanExpr('x | (y & z)')
    assert expr3.free_symbols == {Expr.symbol('x'), Expr.symbol('y'), Expr.symbol('z')}
    assert expr3.args[0] == Expr.symbol('x')
    assert expr3.args[1] == BooleanExpr('y & z')
    assert expr3.args[1].args[0] == Expr.symbol('y')


@pytest.mark.parametrize(
    'expr_lhs, expr_rhs',
    [
        (Expr('x'), Expr('y')),
        (Expr('x'), Expr('y + 1')),
        (Expr('x'), Expr.function('f', 't')),
    ],
)
def test_boolean_expr_args_rel(expr_lhs, expr_rhs):
    expr = BooleanExpr.eq(expr_lhs, expr_rhs)
    assert expr.args == (expr_lhs, expr_rhs)
    expr = BooleanExpr.ne(expr_lhs, expr_rhs)
    assert expr.args == (expr_lhs, expr_rhs)


@pytest.mark.parametrize(
    'expr, ref',
    [
        (BooleanExpr.eq(Expr('x'), Expr('y + 1')), 'x = y + 1'),
        (BooleanExpr.eq(Expr('y + 1'), Expr('x')), 'x = y + 1'),
    ],
)
def test_boolean_expr_unicode(expr, ref):
    assert expr.unicode() == ref


@pytest.mark.parametrize(
    'expr, ref',
    [
        (BooleanExpr(True), True),
        (BooleanExpr(False), False),
        (BooleanExpr(True & False), False),
        (BooleanExpr(True | False), True),
        (BooleanExpr('x & y'), False),
    ],
)
def test_boolean_expr_is_true(expr, ref):
    assert expr.is_true() == ref


@pytest.mark.parametrize(
    'expr, ref',
    [
        (BooleanExpr(True), False),
        (BooleanExpr(False), True),
        (BooleanExpr(True & False), True),
        (BooleanExpr(True | False), False),
        (BooleanExpr('x & y'), False),
    ],
)
def test_boolean_expr_is_false(expr, ref):
    assert expr.is_false() == ref


@pytest.mark.parametrize(
    'expr, ref',
    [
        (BooleanExpr(True), False),
        (BooleanExpr(False), False),
        (BooleanExpr(True & False), False),
        (BooleanExpr(True | False), False),
        (BooleanExpr('x & y'), True),
    ],
)
def test_boolean_expr_is_indeterminate(expr, ref):
    assert expr.is_indeterminate() == ref


@pytest.mark.parametrize(
    'expr, ref',
    [
        (Expr(1), '1'),
        (Expr('x'), 'x'),
        (Expr('x + y'), 'x + y'),
        (BooleanExpr.eq(Expr('x'), Expr('y + 1')), 'Eq(x, y + 1)'),
        (BooleanExpr.eq(Expr('y + 1'), Expr('x')), 'Eq(y + 1, x)'),
    ],
)
def test_printer(expr, ref):
    printer = ExprPrinter()
    assert str(printer._print(expr)) == ref


def test_first():
    expr = Expr.first("WGT", "ID")
    assert expr == Expr.function("first", ("WGT", "ID"))


def test_newind():
    expr = Expr.newind()
    assert expr == Expr.function("newind", ())


def test_forward():
    expr = Expr.forward(Expr.symbol('TIME'), Expr.symbol('AMT') > 0)
    assert expr == Expr.function("forward", ('TIME', 'AMT > 0'))
    assert isinstance(expr.args[0], Expr)
    assert isinstance(expr.args[1], BooleanExpr)


def test_loggamma():
    expr = Expr('x').loggamma()
    assert str(expr) == 'loggamma(x)'


def test_series():
    x = Expr.symbol('x')
    y = x.exp()
    ser = y.series(x, n=2)
    assert ser == x + 1


def test_hashability():
    x = Expr.symbol('x')
    assert isinstance(hash(x), int)
    y = BooleanExpr.true()
    assert isinstance(hash(y), int)
