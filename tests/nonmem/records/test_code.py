import pytest
import sympy

from pharmpy.basic import Expr
from pharmpy.internals.expr.funcs import PHI
from pharmpy.model import Assignment
from pharmpy.model.external.nonmem.records.code_record import CodeRecord


def _ensure_trailing_newline(buf):
    # FIXME: This should not be necessary
    return buf if buf[-1] == '\n' else buf + '\n'


def S(x):
    return sympy.Symbol(x)


@pytest.mark.usefixtures('parser')
@pytest.mark.parametrize(
    'buf,sym,expression',
    [
        (
            '$PRED\nY = THETA(1) + ETA(1) + EPS(1)',
            S('Y'),
            S('THETA(1)') + S('ETA(1)') + S('EPS(1)'),
        ),
        (  # NOTE: github issue #1362
            '$PRED\nY = THETA (1) + ETA(1) + EPS(1)',
            S('Y'),
            S('THETA(1)') + S('ETA(1)') + S('EPS(1)'),
        ),
        (
            '$PRED\nY = THETA(1) + ETA(1) + ERR(1)',
            S('Y'),
            S('THETA(1)') + S('ETA(1)') + S('EPS(1)'),
        ),
        ('$PRED\nCL = 2', S('CL'), 2),
        ('$PRED\n;FULL LINE COMMENT\n K=-1', S('K'), -1),
        ('$PRED K=-1.5', S('K'), -1.5),
        ('$PRED\nCL = KA', S('CL'), S('KA')),
        ('$PRED\nCL=KA', S('CL'), S('KA')),
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
        ('$PRED D = 1D-10', S('D'), 1e-10),
        (
            '$PK CL = PLOG(THETA(1))',
            S('CL'),
            sympy.Piecewise(
                (sympy.log(2.8e-103), S('THETA(1)') < 2.8e-103), (sympy.log(S('THETA(1)')), True)
            ),
        ),
        (
            '$PK CL = PLOG10(THETA(1))',
            S('CL'),
            sympy.Piecewise(
                (sympy.log(2.8e-103, 10), S('THETA(1)') < 2.8e-103),
                (sympy.log(S('THETA(1)'), 10), True),
            ),
        ),
        ('$PRED CL = ABS(X)', S('CL'), abs(S('X'))),
        ('$PRED CL = EXP(2)', S('CL'), sympy.exp(2)),
        ('$PRED CL = PEXP(2)', S('CL'), sympy.exp(2)),
        ('$PRED CL = PEXP(101)', S('CL'), sympy.exp(100)),
        ('$PRED CL = exp(2)', S('CL'), sympy.exp(2)),
        ('$PRED cL = eXp(2)', S('CL'), sympy.exp(2)),
        ('$PRED CL = DEXP(ETA(1))', S('CL'), sympy.exp(S('ETA(1)'))),
        (
            '$PRED CL = PDZ(ETA(1))',
            S('CL'),
            sympy.Piecewise((1 / 2.8e-103, abs(S('ETA(1)')) < 2.8e-103), (1 / S('ETA(1)'), True)),
        ),
        (
            '$PRED CL = PZR(ETA(1))',
            S('CL'),
            sympy.Piecewise((2.8e-103, abs(S('ETA(1)')) < 2.8e-103), (S('ETA(1)'), True)),
        ),
        (
            '$PRED CL = PNP(ETA(1))',
            S('CL'),
            sympy.Piecewise((2.8e-103, S('ETA(1)') < 2.8e-103), (S('ETA(1)'), True)),
        ),
        (
            '$PRED CL = PHE(ETA(1))',
            S('CL'),
            sympy.Piecewise((100, S('ETA(1)') > 100), (S('ETA(1)'), True)),
        ),
        (
            '$PRED CL = PNG(ETA(1))',
            S('CL'),
            sympy.Piecewise((0, S('ETA(1)') < 0), (S('ETA(1)'), True)),
        ),
        ('$PRED CL = LOG(V + 1)', S('CL'), sympy.log(S('V') + 1)),
        ('$PRED CL = DLOG(V + 1)', S('CL'), sympy.log(S('V') + 1)),
        ('$PRED CL = LOG10(3.5 + THETA(1))', S('CL'), sympy.log(3.5 + S('THETA(1)'), 10)),
        ('$PRED CL = (SIN(X) + COS(X))', S('CL'), sympy.sin(S('X')) + sympy.cos(S('X'))),
        ('$PRED CL = (DSIN(X) + DCOS(X))', S('CL'), sympy.sin(S('X')) + sympy.cos(S('X'))),
        ('$PRED S22 = ABS(1 + 2 + SIN(X))', S('S22'), sympy.Abs(3 + sympy.sin(S('X')))),
        ('$PRED CL = TAN(X) * EXP(Y)', S('CL'), sympy.tan(S('X')) * sympy.exp(S('Y'))),
        ('$PRED CL = DTAN(X) * DEXP(Y)', S('CL'), sympy.tan(S('X')) * sympy.exp(S('Y'))),
        ('$PRED CL = PTAN(X) * EXP(Y)', S('CL'), sympy.tan(S('X')) * sympy.exp(S('Y'))),
        (
            '$PRED K_ = ATAN(1) - ASIN(X)/ACOS(X)',
            S('K_'),
            sympy.atan(1) - sympy.asin(S('X')) / sympy.acos(S('X')),
        ),
        (
            '$PRED K_ = PATAN(1) - PASIN(X)/PACOS(X)',
            S('K_'),
            sympy.atan(1) - sympy.asin(S('X')) / sympy.acos(S('X')),
        ),
        ('$PRED CL = INT(-2.2)', S('CL'), -2),
        ('$PRED CL = DINT(-2.2)', S('CL'), -2),
        ('$PRED cl = int(-2.2)', S('CL'), -2),
        ('$PRED CL = INT(0.2)', S('CL'), 0),
        ('$PRED CL = SQRT(X)', S('CL'), sympy.sqrt(S('X'))),
        ('$PRED CL = DSQRT(X)', S('CL'), sympy.sqrt(S('X'))),
        (
            '$PRED CL = PSQRT(X)',
            S('CL'),
            sympy.Piecewise((0, S('X') < 0), (sympy.sqrt(S('X')), True)),
        ),
        ('$PRED CL = MOD(1, 2)', S('CL'), sympy.Mod(1, 2)),
        ('$PRED CL = DMOD(1, 2)', S('CL'), sympy.Mod(1, 2)),
        ('$PRED CL = GAMLN(2 + X)   ;COMMENT', S('CL'), sympy.loggamma(S('X') + 2)),
        ('$PRED C02 = PHI(2 + X)', S('C02'), PHI(S('X') + 2)),
        ('$PRED IF (X.EQ.2) CL=23', S('CL'), sympy.Piecewise((23, sympy.Eq(S('X'), 2)))),
        ('$PRED IF (X.EQ.2)CL=23', S('CL'), sympy.Piecewise((23, sympy.Eq(S('X'), 2)))),
        ('$PRED IF(X.EQ.2)CL=23', S('CL'), sympy.Piecewise((23, sympy.Eq(S('X'), 2)))),
        ('$PRED if (x.EQ.2) Cl=23', S('CL'), sympy.Piecewise((23, sympy.Eq(S('X'), 2)))),
        (
            '$PRED IF (X.NE.1.5) CL=THETA(1)',
            S('CL'),
            sympy.Piecewise((S('THETA(1)'), sympy.Ne(S('X'), 1.5))),
        ),
        (
            '$PRED IF (X.EQ.2+1) CL=23',
            S('CL'),
            sympy.Piecewise((23, sympy.Eq(S('X'), 3))),
        ),
        (
            '$PRED IF (X < ETA(1)) CL=23',
            S('CL'),
            sympy.Piecewise((23, sympy.Lt(S('X'), S('ETA(1)')))),
        ),
        (
            '$PK IF(AMT.GT.0) BTIME=TIME',
            S('BTIME'),
            Expr.forward('TIME', 'AMT>0'),
        ),
        (
            '$PRED IF (X.EQ.2.AND.Y.EQ.3) CL=23',
            S('CL'),
            sympy.Piecewise((23, sympy.And(sympy.Eq(S('X'), 2), sympy.Eq(S('Y'), 3)))),
        ),
        (
            '$PRED IF (X.EQ.2.OR.Y.EQ.3) CL=23',
            S('CL'),
            sympy.Piecewise((23, sympy.Or(sympy.Eq(S('X'), 2), sympy.Eq(S('Y'), 3)))),
        ),
        (
            '$PRED IF (.NOT.X.EQ.2) CL=25',
            S('CL'),
            sympy.Piecewise((25, sympy.Not(sympy.Eq(S('X'), 2)))),
        ),
        (
            '$PRED IF (Q.EQ.(R+C)/D) L=0',
            S('L'),
            sympy.Piecewise(
                (0, sympy.Eq(S('Q'), sympy.Mul(sympy.Add(S('R'), S('C')), 1 / S('D'))))
            ),
        ),
        (
            '$PRED IF (Q.EQ.R+C/D) L=0',
            S('L'),
            sympy.Piecewise((0, sympy.Eq(S('Q'), S('R') + S('C') / S('D')))),
        ),
        ('$PRED\nA_0(1) = 2', S('A_0(1)'), 2),
        (
            '$PRED\nIF(SPECIES.EQ.4)     IPRG = THETA(49)*EXP(ETA(11))',
            S('IPRG'),
            sympy.Piecewise((S('THETA(49)') * sympy.exp(S('ETA(11)')), sympy.Eq(S('SPECIES'), 4))),
        ),
        ('$PRED\nCL=-KA', S('CL'), -S('KA')),
        ('$PRED\nIFST=-KA', S('IFST'), -S('KA')),
        ('$PRED\nCL=-KA+2', S('CL'), -S('KA') + 2),
        ('$PRED\nCL=-2*X*2', S('CL'), -4 * S('X')),
        ('$PRED\nCL=-(-X)', S('CL'), S('X')),
        (
            '$PRED IF(ROUT.EQ.0.AND.DAYP.EQ.1.AND.PROT.EQ.1088) STRT=10881',
            S('STRT'),
            sympy.Piecewise(
                (
                    10881,
                    sympy.And(
                        sympy.Eq(S('ROUT'), 0), sympy.Eq(S('DAYP'), 1), sympy.Eq(S('PROT'), 1088)
                    ),
                ),
            ),
        ),
        ('$PRED X=A+B+C', S('X'), S('A') + S('B') + S('C')),
        ('$PRED X=A+B-C', S('X'), S('A') + S('B') - S('C')),
        ('$PRED X=A+B*C', S('X'), S('A') + S('B') * S('C')),
        ('$PRED X=A*B+C', S('X'), S('A') * S('B') + S('C')),
        ('$PRED\n"VERBATIM STUFF\nK=1', S('K'), 1),
        ('$PRED\n"VERBATIM STUFF\n"ON TWO LINES\nK=1', S('K'), 1),
        ('$PRED (CALLFL=0)\n\nK=1', S('K'), 1),
        ('$ERROR\nCL = 2', S('CL'), 2),
        ('$ERROR (ONLY OBSERVATION) \nCL = 2', S('CL'), 2),
        ('$ERROR\n (ONLY OBSERVATION) \nCL = 2', S('CL'), 2),
        ('$ERROR\n(ONLY OBSERVATION)\nCL = 2', S('CL'), 2),
        ('$PRED\nCL = 2\nEXIT\n', S('CL'), 2),
        ('$PRED\nCL = 2\nEXIT \n', S('CL'), 2),
        ('$PRED\nCL = 2\nEXIT 1 \n', S('CL'), 2),
        ('$PRED\nCL = 2\nEXIT 1 23 \n', S('CL'), 2),
        ('$PRED\nCL = 2\nCALL RANDOM (2, R)\n', S('CL'), 2),
        ('$PRED\nCL = 2\nCALL  RANDOM (2, R)\n', S('CL'), 2),
        ('$PRED CALLFL=0', S('CALLFL'), 0),
        ('$PRED\nCL = 2\n  RETURN  \n', S('CL'), 2),
        ('$PRED\nCOM(1) = 0\n', S('COM(1)'), 0),
        ('$PRED\nANYTHING(1) = 0\n', S('ANYTHING(1)'), 0),
        ('$PRED\nGROK(1,3) = 0\n', S('GROK(1,3)'), 0),
        ('$PRED\n X = &  \n  23\n', S('X'), 23),
        ('$PRED\nX=2\nDO WHILE (X < 23)\n CL = 2\nEND DO\n', S('X'), 2),
        (  # NOTE: Regression when we switched to LALR(1)
            '$PK\nALAG1 = EXP(PHI+KPLAG)/(1+EXP(PHI+KPLAG))',
            S('ALAG1'),
            sympy.exp(S('PHI') + S('KPLAG'))
            / (sympy.Integer(1) + sympy.exp(S('PHI') + S('KPLAG'))),
        ),
        (
            '$PK\nV=TVV*EXP(ETA2)',
            S('V'),
            S('TVV') * sympy.exp(S('ETA2')),
        ),
        (
            '$PK\nV=THETA\t \x00\t \x00\t \x00(1)',
            S('V'),
            S('THETA(1)'),
        ),
        (
            '$PK\nV=THETA \t & \t\t\x00\r\n & \t\t\x00\n  (1)',
            S('V'),
            S('THETA(1)'),
        ),
        ('$PK\nX(Z)=0', S('X(Z)'), 0),
        ('$PK\nX(  Z \x00)=0', S('X(Z)'), 0),
        ('$PK\n X \t \t \x00\x00\t \x00(\x00Z)=0', S('X(Z)'), 0),
        ('$PK\n X \t \t \x00& \t\t\x00\n \t \x00( -1 , 2 )=0', S('X(-1,2)'), 0),
        (
            '$PRED IF (NEWIND.NE.2) BCLC=CLC',
            S('BCLC'),
            Expr.first("CLC", "ID"),
        ),
        (
            '$PK IF (AMT.GT.0) PODO=AMT',
            S('PODO'),
            Expr.forward('AMT', 'AMT>0'),
        ),
    ],
)
def test_single_assignments(parser, buf, sym, expression):
    buf = _ensure_trailing_newline(buf)
    rec = parser.parse(buf).records[0]
    assert len(rec.statements) == 1
    statement = rec.statements[0]
    assert statement.symbol == sym
    assert statement.expression == expression


@pytest.mark.usefixtures('parser')
@pytest.mark.parametrize(
    'buf,symb_expr_arr',
    [
        (
            '$PRED\nIF (X.EQ.0) THEN\nY = 23\nZ = 9\nEND IF',
            [
                (S('Y'), sympy.Piecewise((23, sympy.Eq(S('X'), 0)))),
                (S('Z'), sympy.Piecewise((9, sympy.Eq(S('X'), 0)))),
            ],
        ),
        (
            '$PRED\nIF (FA1.EQ.0) IOV_1 = ETA(3)\nIF (FA1.EQ.1) IOV_1 = ETA(4)',
            [
                (S('IOV_1'), sympy.Piecewise((S('ETA(3)'), sympy.Eq(S('FA1'), 0)))),
                (
                    S('IOV_1'),
                    sympy.Piecewise((S('ETA(4)'), sympy.Eq(S('FA1'), 1)), (S('IOV_1'), True)),
                ),
            ],
        ),
        (
            '$PRED IF (B0.LT.3) THEN\nCL = THETA(1)\nELSE\nCL = 23\nENDIF',
            [(S('CL'), sympy.Piecewise((S('THETA(1)'), S('B0') < 3), (23, True)))],
        ),
        (
            '$PRED IF (B0.LT.3) THEN\nCL = THETA(1)\nKA = THETA(2)\n  ELSE  \nCL = 23\nKA=82\n'
            'END IF',
            [
                (S('CL'), sympy.Piecewise((S('THETA(1)'), S('B0') < 3), (23, True))),
                (S('KA'), sympy.Piecewise((S('THETA(2)'), S('B0') < 3), (82, True))),
            ],
        ),
        (
            '$PRED    IF (A>=.5) THEN   \n  VAR=1+2 \nELSE IF (B.EQ.23) THEN ; C\nVAR=9.25\n'
            'END IF  \n',
            [
                (
                    S('VAR'),
                    sympy.Piecewise((3, S('A') >= 0.5), (9.25, sympy.Eq(S('B'), 23))),
                )
            ],
        ),
        (
            '$PRED    if (a>=0.5) then   \n  var=1+2 \nELSE if (b.eq.23) then ; C\nVAR=9.25\n'
            'END IF  \n',
            [
                (
                    S('VAR'),
                    sympy.Piecewise((3, S('A') >= 0.5), (9.25, sympy.Eq(S('B'), 23))),
                )
            ],
        ),
        (
            '$PRED   IF (A>=0.5) THEN   \n  VAR1=1+2 \nELSE IF  (B.EQ.23)  THEN \nVAR2=9.25\n'
            'END IF  \n',
            [
                (S('VAR1'), sympy.Piecewise((3, S('A') >= 0.5))),
                (S('VAR2'), sympy.Piecewise((9.25, sympy.Eq(S('B'), 23)))),
            ],
        ),
        (
            '$PRED\nIF (X.EQ.0) THEN    ;   anything  \n  Y = 23\nZ = 9\nEND IF;AFTER',
            [
                (S('Y'), sympy.Piecewise((23, sympy.Eq(S('X'), 0)))),
                (S('Z'), sympy.Piecewise((9, sympy.Eq(S('X'), 0)))),
            ],
        ),
        (
            '$PRED\nIF (NEWIND.NE.2.OR.EVID.GE.3) THEN ; begin\nTNXD=TIME ; TIME\nENDIF',
            [
                (
                    S('TNXD'),
                    sympy.Piecewise(
                        (S('TIME'), sympy.Or(sympy.Ne(Expr.newind(), 2), sympy.Ge(S('EVID'), 3))),
                    ),
                )
            ],
        ),
        (
            '$PRED\nIF (NEWIND.NE.2.OR.EVID.GE.3.) THEN ; begin\nTNXD=TIME ; TIME\nENDIF',
            [
                (
                    S('TNXD'),
                    sympy.Piecewise(
                        (S('TIME'), sympy.Or(sympy.Ne(Expr.newind(), 2), sympy.Ge(S('EVID'), 3.0))),
                    ),
                )
            ],
        ),
        (
            '$PRED\nIF (NEWIND.NE.2..OR.EVID.GE.3) THEN ; begin\nTNXD=TIME ; TIME\nENDIF',
            [
                (
                    S('TNXD'),
                    sympy.Piecewise(
                        (S('TIME'), sympy.Or(sympy.Ne(Expr.newind(), 2.0), sympy.Ge(S('EVID'), 3))),
                    ),
                )
            ],
        ),
        (
            '$PRED IF (B0.LT.3) THEN\nCL = THETA(1)\nELSE;A close comment\nCL = 23\nENDIF',
            [(S('CL'), sympy.Piecewise((S('THETA(1)'), S('B0') < 3), (23, True)))],
        ),
        (
            '$PRED\nIF(MIXNUM.EQ.3) THEN\n    TVCL=THETA(1)       ; CL in population 1\nELSE\n    '
            'TVCL=THETA(2)       ; CL in population 2\nENDIF\n',
            [
                (
                    S('TVCL'),
                    sympy.Piecewise(
                        (S('THETA(1)'), sympy.Eq(S('MIXNUM'), 3)), (S('THETA(2)'), True)
                    ),
                )
            ],
        ),
        (
            '$PRED\nIF (X.EQ.0) THEN\nY = 23.\nZ = 9\nELSE\nEND IF',
            [
                (S('Y'), sympy.Piecewise((23.0, sympy.Eq(S('X'), 0)))),
                (S('Z'), sympy.Piecewise((9, sympy.Eq(S('X'), 0)))),
            ],
        ),
        (
            '$PRED\nIF (X.EQ.0) THEN\nY = 23\nZ = 9.\nELSE IF (X.EQ.23) THEN\nELSE\nEND IF',
            [
                (S('Y'), sympy.Piecewise((23, sympy.Eq(S('X'), 0)))),
                (S('Z'), sympy.Piecewise((9.0, sympy.Eq(S('X'), 0)))),
            ],
        ),
        (
            '$PRED\nIF (X.EQ.0) THEN\nY = 23\nZ = 9\nELSE IF (X.EQ.44) THEN\nEND IF',
            [
                (S('Y'), sympy.Piecewise((23, sympy.Eq(S('X'), 0)))),
                (S('Z'), sympy.Piecewise((9, sympy.Eq(S('X'), 0)))),
            ],
        ),
        (
            '$PRED\nIF (X.EQ.0) THEN\nELSE\nY = 23\nZ = 9\nEND IF',
            [
                (S('Y'), sympy.Piecewise((23, sympy.Ne(S('X'), 0)))),
                (S('Z'), sympy.Piecewise((9, sympy.Ne(S('X'), 0)))),
            ],
        ),
        (
            '$PRED\nIF (X.EQ.0) THEN\nY = 23\nZ = 9\nELSE\n CALL RANDOM(1,R)\nEND IF',
            [
                (S('Y'), sympy.Piecewise((23, sympy.Eq(S('X'), 0)))),
                (S('Z'), sympy.Piecewise((9, sympy.Eq(S('X'), 0)))),
            ],
        ),
        (
            '$PRED\nIF (X.EQ.0) THEN\nY = 23\nELSE\nY = 7\nZ = 9\nEND IF',
            [
                (S('Y'), sympy.Piecewise((23, sympy.Eq(S('X'), 0)), (7, True))),
                (S('Z'), 9),
            ],
        ),
    ],
)
def test_block_if(parser, buf, symb_expr_arr):
    buf = _ensure_trailing_newline(buf)
    rec = parser.parse(buf).records[0]
    assert len(rec.statements) == len(symb_expr_arr)
    for statement, (symb, expr) in zip(rec.statements, symb_expr_arr):
        assert statement.symbol == symb
        assert statement.expression == expr


@pytest.mark.parametrize(
    'buf',
    (
        '$PK IF (CL.EQ.0) EXIT 1',
        '$PK IF (CL.EQ.0) EXIT 1 24',
    ),
)
def test_exit(parser, buf):
    buf = _ensure_trailing_newline(buf)
    rec = parser.parse(buf).records[0]
    assert len(rec.statements) == 0


@pytest.mark.usefixtures('parser')
@pytest.mark.parametrize(
    'buf',
    [
        '$PRED\nY = THETA(1) + ETA(1) + EPS(1)',
        '$PRED IF (X.EQ.2) CL=23',
        '$PRED IF (X.NE.1.5) CL=THETA(1)',
        '$PRED IF (X.EQ.2+1) CL=23',
        '$PRED IF (X.EQ.2.AND.Y.EQ.3) CL=23',
        '$PRED IF (X.EQ.2.OR.Y.EQ.3) CL=23',
        '$PRED IF (.NOT.X.EQ.2) CL=25',
    ],
)
def test_grammar_repeats(parser, buf):  # Tests that there are no repeats due to parsing grammar
    buf = _ensure_trailing_newline(buf)
    rec = parser.parse(buf).records[0]
    tree_walk_gen = rec.root.tree_walk()
    parent = next(tree_walk_gen)

    for child in tree_walk_gen:
        assert parent != child
        parent = child


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
    assert rec.statements[0].expression == Expr.forward('TIME', 'AMT > 0')
    assert rec.statements[1].expression == S('TIME') - S('BTIME')
    assert rec.statements[2].expression == S('THETA(1)') * S('WGT')
    assert rec.statements[3].expression == S('THETA(2)') * S('WGT')
    assert rec.statements[4].expression == sympy.Piecewise(
        (S('TVV') * (1 + S('THETA(3)')), sympy.Lt(S('APGR'), 5)), (S('TVV'), True)
    )
    assert rec.statements[5].expression == S('TVCL') * sympy.exp(S('ETA(1)'))
    assert rec.statements[6].expression == S('TVV') * sympy.exp(S('ETA(2)'))
    assert rec.statements[7].expression == S('V')
    symbol_names = {s.name for s in rec.statements.free_symbols}
    assert symbol_names == {
        'AMT',
        'BTIME',
        'TIME',
        'TAD',
        'TVCL',
        'THETA(1)',
        'WGT',
        'TVV',
        'THETA(2)',
        'APGR',
        'THETA(3)',
        'CL',
        'ETA(1)',
        'V',
        'ETA(2)',
        'S1',
    }


@pytest.mark.usefixtures('parser')
@pytest.mark.parametrize(
    'buf_original,buf_new',
    [
        ('$PRED\nY = THETA(1) + ETA(1) + EPS(1)', '$PRED\nY = THETA(1) + ETA(1) + EPS(1)'),
        ('$PRED\nY = THETA(1) + ETA(1) + EPS(1) ;comment', '$PRED\nY = THETA(1) + ETA(1) + EPS(1)'),
        ('$PRED\nY = THETA(1) + ETA(1) + EPS(1)', '$PRED\nY=THETA(1)+ETA(1)+EPS(1)'),
    ],
)
def test_statements_setter_identical(parser, buf_original, buf_new):
    buf_original = _ensure_trailing_newline(buf_original)
    buf_new = _ensure_trailing_newline(buf_new)
    rec_original = parser.parse(buf_original).records[0]
    rec_new = parser.parse(buf_new).records[0]

    assert rec_original.statements == rec_new.statements


@pytest.mark.usefixtures('parser')
@pytest.mark.parametrize(
    'buf_original,buf_new',
    [
        (
            '$PRED\nY = THETA(1) + ETA(1) + EPS(1)\nCL = 2',
            '$PRED\nY = THETA(1) + ETA(1) + EPS(1)\n',
        ),
        ('$PRED\nY = THETA(1) + ETA(1) + EPS(1)\nCL = 2', '$PRED\nCL = 2'),
        ('$PRED\nY = A + B\nX = C - D\nZ = E * F', '$PRED\nY = A + B\nZ = E * F'),
    ],
)
def test_statements_setter_remove(parser, buf_original, buf_new):
    buf_original = _ensure_trailing_newline(buf_original)
    buf_new = _ensure_trailing_newline(buf_new)
    rec_original = parser.parse(buf_original).records[0]
    rec_new = parser.parse(buf_new).records[0]

    newrec = rec_original.update_statements(rec_new.statements)

    assert newrec.statements == rec_new.statements
    assert tuple(newrec.root.subtrees('statement')) == tuple(rec_new.root.subtrees('statement'))


@pytest.mark.usefixtures('parser')
@pytest.mark.parametrize(
    'buf_original,buf_new',
    [
        (
            '$PRED\nY = THETA(1) + ETA(1) + EPS(1)\n',
            '$PRED\nY = THETA(1) + ETA(1) + EPS(1)\nCL = 2\n',
        ),
        (
            '$PRED\nY = THETA(1) + ETA(1) + EPS(1)\n',
            '$PRED\nY = THETA(1) + ETA(1) + EPS(1)\nS1 = V\nCL = 2\n',
        ),
        (
            '$PRED\nY = THETA(1) + ETA(1) + EPS(1)\nS1 = V\n',
            '$PRED\nCL = 2\nY = THETA(1) + ETA(1) + EPS(1)\nS1 = V\n',
        ),
        (
            '$PRED\nY = THETA(1) + ETA(1) + EPS(1)\nCL = 2\n',
            '$PRED\nY = THETA(1) + ETA(1) + EPS(1)\nS1 = V\nCL = 2\n',
        ),
        (
            '$PRED\nY = THETA(1) + ETA(1) + EPS(1)\nCL = 2\n',
            '$PRED\nY = THETA(1) + ETA(1) + EPS(1)\nS1 = V\nTVCL = WGT + 2\nCL = 2\n',
        ),
        # (
        #    '$PRED\nY = THETA(1) + ETA(1) + EPS(1)\n',
        #    '$PRED\nY = THETA(1) + ETA(1) + EPS(1)\nIF (AMT.GT.0) BTIME = TIME\n',
        # ),
        # (
        #    '$PRED\nBTIME = 2\nIF (AMT.GT.0) BTIME = TIME\n',
        #    '$PRED\nCL = 2\nBTIME = 2\nIF (AMT.GT.0) BTIME = TIME\n',
        # ),
    ],
)
def test_statements_setter_add(parser, buf_original, buf_new):
    rec_original = parser.parse(buf_original).records[0]
    rec_new = parser.parse(buf_new).records[0]

    newrec = rec_original.update_statements(rec_new.statements)
    assert newrec.statements == rec_new.statements
    assert tuple(newrec.root.subtrees('statement')) == tuple(rec_new.root.subtrees('statement'))


@pytest.mark.usefixtures('parser')
@pytest.mark.parametrize(
    'buf_original,buf_new',
    [
        ('$PRED\nY = A + B\n', '$PRED\nZ = E*F\n'),
        ('$PRED\nY = A + B\nX = C - D\nZ = E * F\n', '$PRED\nY = A + B\nX = C\nZ = E * F\n'),
        ('$PRED\nY = A + B\nX = C - D\nZ = E * F\n', '$PRED\nY = A + B\nX = C - D\nZ = E\n'),
        ('$PRED\nY = A + B\nX = C - D\nZ = E * F\n', '$PRED\nY = A + B\nX = C\nZ = E * F\n'),
    ],
)
def test_statements_setter_change(parser, buf_original, buf_new):
    rec_original = parser.parse(buf_original).records[0]
    rec_new = parser.parse(buf_new).records[0]
    newrec = rec_original.update_statements(rec_new.statements)

    assert newrec.statements == rec_new.statements
    assert tuple(newrec.root.subtrees('statement')) == tuple(rec_new.root.subtrees('statement'))
    assert str(newrec) == buf_new


@pytest.mark.usefixtures('parser')
@pytest.mark.parametrize(
    'buf_original,sym,expression,buf_new',
    [
        (
            '$PRED\nY = THETA(1) + ETA(1) + EPS(1)\n',
            S('CL'),
            2,
            '$PRED\nY = THETA(1) + ETA(1) + EPS(1)\nCL = 2\n',
        ),
        (
            '$PRED\n"FIRST\n"!Fortran code goes here\n',
            S('V'),
            -S('CL'),
            '$PRED\nV = -CL\n"FIRST\n"!Fortran code goes here\n',
        ),
    ],
)
def test_statements_setter_add_from_sympy(parser, buf_original, sym, expression, buf_new):
    rec_original = parser.parse(buf_original).records[0]

    assignment = Assignment.create(sym, expression)
    statements = rec_original.statements + [assignment]
    newrec = rec_original.update_statements(statements)

    assert str(newrec) == buf_new


@pytest.mark.usefixtures('parser')
@pytest.mark.parametrize(
    'buf_original,assignment,buf_expected',
    [
        (
            '$PRED\nY = THETA(1) + ETA(1) + EPS(1)\n',
            Assignment.create(S('Z'), S('THETA(2)')),
            '$PRED\nY = THETA(1) + ETA(1) + EPS(1)\nZ = THETA(2)\n',
        ),
        (
            '$PRED\nY = THETA(1) + ETA(1) + EPS(1)\nCL = 1.3\n',
            Assignment.create(S('Z'), S('THETA(2)')),
            '$PRED\nY = THETA(1) + ETA(1) + EPS(1)\nCL = 1.3\nZ = THETA(2)\n',
        ),
        (
            '$PRED\nY = THETA(1) + ETA(1) + EPS(1)\n',
            Assignment.create(S('YWGT'), sympy.Piecewise((1, sympy.Eq(S('WGT'), S('NaN'))))),
            '$PRED\nY = THETA(1) + ETA(1) + EPS(1)\n' 'IF (NaN.EQ.WGT) YWGT = 1\n',
        ),
        (
            '$PRED\nY = THETA(1) + ETA(1) + EPS(1)\n',
            Assignment.create(S('Z'), PHI(S('A') + S('B'))),
            '$PRED\nY = THETA(1) + ETA(1) + EPS(1)\n' 'Z = PHI(A + B)\n',
        ),
    ],
)
def test_update(parser, buf_original, assignment, buf_expected):
    rec_original = parser.parse(buf_original).records[0]

    statements = rec_original.statements + assignment
    newrec = rec_original.update_statements(statements)

    assert str(newrec) == buf_expected


def test_nested_block_if(parser):
    code = (
        '\nIF (X.EQ.23) THEN\nIF (Y.EQ.0) THEN\nCL=1\nELSE\nCL=2\nENDIF\n' 'CL=5\nELSE\nCL=6\nENDIF'
    )
    code = _ensure_trailing_newline(code)
    rec = parser.parse('$PRED' + code).records[0]

    s = rec.statements
    rec = rec.update_statements(s)
    assert str(rec.root) == code


@pytest.mark.usefixtures('parser')
@pytest.mark.parametrize(
    'symbol, expression, buf_expected',
    [
        (
            S('BTIME'),
            sympy.Piecewise((S('TIME'), sympy.Gt(S('AMT'), 0))),
            'IF (AMT.GT.0) BTIME = TIME\n',
        ),
        # (S('CL'), sympy.Piecewise((23, sympy.Eq(S('X'), 2))), 'IF (X.EQ.2) CL = 23\n'),
        # (
        #     S('CLWGT'),
        #     sympy.Piecewise((23, sympy.Eq(S('X'), 1)), (0, sympy.Eq(S('X'), 0))),
        #     'IF (X.EQ.1) CLWGT = 23\nIF (X.EQ.0) CLWGT = 0\n',
        # ),
        # (
        #     S('X'),
        #     sympy.Piecewise((0, sympy.And(sympy.Ne(S('Y'), 1), sympy.Ne(S('K'), 2)))),
        #     'IF (K.NE.2.AND.Y.NE.1) X = 0\n',
        # ),
        # (
        #     S('X'),
        #     sympy.Piecewise((0, sympy.Ne(S('Y'), 1)), (1, True)),
        #     'IF (Y.NE.1) THEN\n    X = 0\nELSE\n    X = 1\nEND IF\n',
        # ),
        # (
        #     S('X'),
        #     sympy.Piecewise((0, sympy.Ne(S('Y'), 1)), (2, sympy.Eq(S('Y'), 15)), (1, True)),
        #     'IF (Y.NE.1) THEN\n    X = 0\nELSE IF (Y.EQ.15) THEN\n    X = 2\nELSE\n    X = 1\nEND IF\n',  # noqa: E501
        # ),
    ],
)
def test_translate_sympy_piecewise(parser, symbol, expression, buf_expected):
    buf_original = '$PRED\nY = THETA(1) + ETA(1) + EPS(1)\n'
    rec = parser.parse(buf_original).records[0]
    assert isinstance(rec, CodeRecord)
    s = Assignment.create(symbol, expression)
    statements = rec.statements + s
    newrec = rec.update_statements(statements)
    assert str(newrec) == buf_original + buf_expected


def test_empty_record(parser):
    rec = parser.parse('$PRED\n').records[0]
    assert len(rec.statements) == 0


def test_set_block_if(parser):
    rec = parser.parse('$PRED\n').records[0]
    z = S('z')
    s = Assignment.create(sympy.Symbol('X'), sympy.Piecewise((23, z < 12), (5, True)))
    statements = rec.statements + s
    rec = rec.update_statements(statements)
    correct = """$PRED
IF (Z.LT.12) THEN
    X = 23
ELSE
    X = 5
END IF
"""
    assert str(rec) == correct


def test_update_error_nodes_nested(parser):
    rec = parser.parse('$PRED\nY=X\n').records[0]
    y = Assignment.create(sympy.Symbol('Y_2'), sympy.Symbol('Z'))
    statements = rec.statements + y
    dvs = {sympy.Symbol('Y'): 1, sympy.Symbol('Y_2'): 2}
    rec = rec.update_statements(statements)
    rec = rec.update_extra_nodes(dvs, 'DVID')
    rec = rec.update_extra_nodes(dvs, 'DVID')
    y_str = 'IF (DVID.EQ.1) Y = Y'
    y2_str = 'IF (DVID.EQ.2) Y = Y_2'
    assert str(rec).split('\n').count(y_str) == 1
    assert str(rec).split('\n').count(y2_str) == 1
