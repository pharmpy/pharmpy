import pytest
import sympy
from sympy import Symbol


def S(x):
    return Symbol(x, real=True)


@pytest.mark.usefixtures('parser')
@pytest.mark.parametrize('buf,symbol,expression', [
    ('$PRED\nY = THETA(1) + ETA(1) + EPS(1)', S('Y'), S('THETA(1)') + S('ETA(1)') + S('EPS(1)')),
    ('$PRED\nCL = 2', S('CL'), 2),
    ('$PRED K=-1', S('K'), -1),
    ('$PRED K=-1.5', S('K'), -1.5),
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
    ('$PRED CL = GAMLN(2 + X)   ;COMMENT', S('CL'), sympy.loggamma(S('X') + 2)),
    ('$PRED IF (X.EQ.2) CL=23', S('CL'), sympy.Piecewise((23, sympy.Eq(S('X'), 2)))),
    ('$PRED IF (X.NE.1.5) CL=THETA(1)', S('CL'),
        sympy.Piecewise((S('THETA(1)'), sympy.Ne(S('X'), 1.5)))),
    ('$PRED IF (X.EQ.2+1) CL=23', S('CL'), sympy.Piecewise((23, sympy.Eq(S('X'), 3)))),
    ('$PRED IF (X < ETA(1)) CL=23', S('CL'), sympy.Piecewise((23, sympy.Lt(S('X'), S('ETA(1)'))))),
    ('$PK IF(AMT.GT.0) BTIME=TIME', S('BTIME'),
        sympy.Piecewise((S('TIME'), sympy.Gt(S('AMT'), 0)))),
    ('$PRED IF (X.EQ.2.AND.Y.EQ.3) CL=23', S('CL'),
        sympy.Piecewise((23, sympy.And(sympy.Eq(S('X'), 2), sympy.Eq(S('Y'), 3))))),
    ('$PRED IF (X.EQ.2.OR.Y.EQ.3) CL=23', S('CL'),
        sympy.Piecewise((23, sympy.Or(sympy.Eq(S('X'), 2), sympy.Eq(S('Y'), 3))))),
    ('$PRED IF (.NOT.X.EQ.2) CL=25', S('CL'),
        sympy.Piecewise((25, sympy.Not(sympy.Eq(S('X'), 2))))),
    ('$PRED IF (Q.EQ.(R+C)/D) L=0', S('L'),
        sympy.Piecewise((0, sympy.Eq(S('Q'), sympy.Mul(sympy.Add(S('R'), S('C')), 1 / S('D')))))),
    ('$PRED IF (Q.EQ.R+C/D) L=0', S('L'),
        sympy.Piecewise((0, sympy.Eq(S('Q'), S('R') + S('C') / S('D'))))),
])
def test_single_assignments(parser, buf, symbol, expression):
    rec = parser.parse(buf).records[0]
    assert len(rec.statements) == 1
    assert rec.statements[0].symbol == symbol
    assert rec.statements[0].expression == expression


@pytest.mark.usefixtures('parser')
@pytest.mark.parametrize('buf,symb_expr_arr', [
    ('$PRED\nIF (X.EQ.0) THEN\nY = 23\nZ = 9\nEND IF', [
        (S('Y'), sympy.Piecewise((23, sympy.Eq(S('X'), 0)))),
        (S('Z'), sympy.Piecewise((9, sympy.Eq(S('X'), 0))))]),
    ('$PRED IF (B0.LT.3) THEN\nCL = THETA(1)\nELSE\nCL = 23\nEND IF', [
        (S('CL'), sympy.Piecewise((S('THETA(1)'), S('B0') < 3), (23, True)))]),
    ('$PRED IF (B0.LT.3) THEN\nCL = THETA(1)\nKA = THETA(2)\n  ELSE  \nCL = 23\nKA=82\nEND IF', [
        (S('CL'), sympy.Piecewise((S('THETA(1)'), S('B0') < 3), (23, True))),
        (S('KA'), sympy.Piecewise((S('THETA(2)'), S('B0') < 3), (82, True)))]),
    ('$PRED    IF (A>=0.5) THEN    \n  VAR=1+2 \nELSE IF  (B.EQ.23)  THEN \nVAR=9.25\nEND IF  \n', [
        (S('VAR'), sympy.Piecewise((3, S('A') >= 0.5), (9.25, sympy.Eq(S('B'), 23))))]),
    ('$PRED   IF (A>=0.5) THEN   \n  VAR1=1+2 \nELSE IF  (B.EQ.23)  THEN \nVAR2=9.25\nEND IF  \n', [
        (S('VAR1'), sympy.Piecewise((3, S('A') >= 0.5))),
        (S('VAR2'), sympy.Piecewise((9.25, sympy.Eq(S('B'), 23))))]),
])
def test_block_if(parser, buf, symb_expr_arr):
    rec = parser.parse(buf).records[0]
    assert len(rec.statements) == len(symb_expr_arr)
    for statement, (symb, expr) in zip(rec.statements, symb_expr_arr):
        assert statement.symbol == symb
        assert statement.expression == expr


def test_pheno(parser):
    code = """$PK

IF(AMT.GT.0) BTIME=TIME
TAD=TIME-BTIME
      TVCL=THETA(1)*WGT
      TVV=THETA(2)*WGT
IF(APGR.LT.5) TVV=TVV*(1+THETA(3))
      CL=TVCL*EXP(ETA(1))
      V=TVV*EXP(ETA(2))
      S1=V
"""
    rec = parser.parse(code).records[0]
    assert len(rec.statements) == 8
    assert rec.statements[0].symbol == S('BTIME')
    assert rec.statements[1].symbol == S('TAD')
    assert rec.statements[2].symbol == S('TVCL')
    assert rec.statements[3].symbol == S('TVV')
    assert rec.statements[4].symbol == S('TVV')
    assert rec.statements[5].symbol == S('CL')
    assert rec.statements[6].symbol == S('V')
    assert rec.statements[7].symbol == S('S1')
    assert rec.statements[0].expression == sympy.Piecewise((S('TIME'), sympy.Gt(S('AMT'), 0)))
    assert rec.statements[1].expression == S('TIME') - S('BTIME')
    assert rec.statements[2].expression == S('THETA(1)') * S('WGT')
    assert rec.statements[3].expression == S('THETA(2)') * S('WGT')
    assert rec.statements[4].expression == sympy.Piecewise((S('TVV') * (1 + S('THETA(3)')),
                                                           sympy.Lt(S('APGR'), 5)))
    assert rec.statements[5].expression == S('TVCL') * sympy.exp(S('ETA(1)'))
    assert rec.statements[6].expression == S('TVV') * sympy.exp(S('ETA(2)'))
    assert rec.statements[7].expression == S('V')
    symbol_names = {s.name for s in rec.statements.free_symbols}
    assert symbol_names == {'AMT', 'BTIME', 'TIME', 'TAD', 'TVCL', 'THETA(1)', 'WGT', 'TVV',
                            'THETA(2)', 'APGR', 'THETA(3)', 'CL', 'ETA(1)', 'V', 'ETA(2)', 'S1'}
