import pytest
import sympy
from sympy import Symbol


def S(x):
    return Symbol(x, real=True)


@pytest.mark.usefixtures('parser')
@pytest.mark.parametrize('buf,symbol,expression', [
    ('$PRED\nY = THETA(1) + ETA(1) + EPS(1)', S('Y'), S('THETA(1)') + S('ETA(1)') + S('EPS(1)')),
    ('$PRED\nCL = 2', S('CL'), 2),
    ('$PRED\nCL = KA', S('CL'), S('KA')),
    ('$PRED\nG = BASE - LESS', S('G'), S('BASE') - S('LESS')),
    ('$PRED CL = THETA(1) * LEFT', S('CL'), S('THETA(1)') * S('LEFT')),
    ('$PRED D2 = WGT / SRC', S('D2'), S('WGT') / S('SRC')),
    ('$PRED D = W * B + C', S('D'), S('W') * S('B') + S('C')),
    ('$PRED D = W * (B + C)', S('D'), S('W') * (S('B') + S('C'))),
    ('$PRED D = A+B*C+D', S('D'), S('A') + (S('B') * S('C')) + S('D')),
    ('$PRED D = A**2', S('D'), S('A') ** 2),
    ('$PRED D = A - (-2)', S('D'), S('A') + 2),
    ('$PRED D = A - (+2)', S('D'), S('A') - 2),
    ('$PRED D = 2.5', S('D'), 2.5),
    ('$PRED CL = EXP(2)', S('CL'), sympy.exp(2)),
    ('$PRED CL = LOG(V + 1)', S('CL'), sympy.log(S('V') + 1)),
    ('$PRED CL = LOG10(3.5 + THETA(1))', S('CL'), sympy.log(3.5 + S('THETA(1)'), 10)),
    ('$PRED CL = (SIN(X) + COS(X))', S('CL'), sympy.sin(S('X')) + sympy.cos(S('X'))),
    ('$PRED S22 = ABS(1 + 2 + SIN(X))', S('S22'), sympy.Abs(3 + sympy.sin(S('X')))),
    ('$PRED CL = TAN(X) * EXP(Y)', S('CL'), sympy.tan(S('X')) * sympy.exp(S('Y'))),
    ('$PRED K_ = ATAN(1) - ASIN(X)/ACOS(X)', S('K_'), sympy.atan(1) - sympy.asin(S('X')) /
        sympy.acos(S('X'))),
    ('$PRED CL = INT(-2.2)', S('CL'), -2),
    ('$PRED CL = INT(0.2)', S('CL'), 0),
    ('$PRED CL = MOD(1, 2)', S('CL'), sympy.Mod(1, 2)),
    ('$PRED CL = GAMLN(2 + X)', S('CL'), sympy.loggamma(S('X') + 2)),
])
def test_single_assignments(parser, buf, symbol, expression):
    rec = parser.parse(buf).records[0]
    rec.root.treeprint()
    assert len(rec.statements) == 1
    assert rec.statements[0].symbol == symbol
    assert rec.statements[0].expression == expression
