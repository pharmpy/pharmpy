import pytest
import sympy

from pharmpy.model import ModelSyntaxError, Parameter, Parameters


def S(x):
    return sympy.Symbol(x)


@pytest.mark.usefixtures('parser')
@pytest.mark.parametrize(
    'buf,results',
    [
        ('$OMEGA 1', [([None], [1.0], False, False)]),
        ('$OMEGA   0.123 \n\n', [([None], [0.123], False, False)]),
        ('$OMEGA   (0 FIX) ; CL', [(['CL'], [0.0], True, False)]),
        (
            '$OMEGA DIAG(2) 1 2 FIX',
            [
                ([None], [1.0], False, False),
                ([None], [2.0], True, False),
            ],
        ),
        (
            '$OMEGA 1 2 3',
            [
                ([None], [1.0], False, False),
                ([None], [2.0], False, False),
                ([None], [3.0], False, False),
            ],
        ),
        ('$OMEGA 0.15 ;CL', [(['CL'], [0.15], False, False)]),
        (
            '$OMEGA 1 \n2 ; S   \n  3 ',
            [
                ([None], [1.0], False, False),
                (['S'], [2.0], False, False),
                ([None], [3.0], False, False),
            ],
        ),
        ('$OMEGA 2 SD', [([None], [4.0], False, False)]),
        ('$OMEGA ;CO\n (VAR 3)', [([None], [3.0], False, False)]),
        (
            '$OMEGA (1)x2',
            [
                ([None], [1.0], False, False),
                ([None], [1.0], False, False),
            ],
        ),
        (
            '$OMEGA BLOCK(2) 1 0.5 2',
            [
                ([None, None, None], [1.0, 0.5, 2.0], False, False),
            ],
        ),
        ('$OMEGA BLOCK(2) SAME', [(None, None, None, True)]),
        ('$OMEGA BLOCK SAME(3)', [(None, None, None, True)]),
        (
            '$OMEGA BLOCK(2) 1 (0.1)x2',
            [
                ([None, None, None], [1.0, 0.1, 0.1], False, False),
            ],
        ),
        (
            '$OMEGA BLOCK(2) CHOLESKY 0.8 -0.3 0.7',
            [
                ([None, None, None], [pytest.approx(0.64), -0.24, 0.58], False, False),
            ],
        ),
        (
            '$OMEGA BLOCK(2) SD 0.8 -0.394 0.762 CORR',
            [
                (
                    [None, None, None],
                    [pytest.approx(0.64), pytest.approx(-0.2401824), pytest.approx(0.580644)],
                    False,
                    False,
                ),
            ],
        ),
        (
            '$OMEGA BLOCK(1)   1.5',
            [
                ([None], [1.5], False, False),
            ],
        ),
        (
            '$OMEGA  0.0258583  ;      V2\n'
            ';$OMEGA BLOCK(1) 0.0075 FIX    ;.02 ; IOC\n'
            ';$OMEGA BLOCK(1) SAME\n',
            [
                (['V2'], [0.0258583], False, False),
            ],
        ),
        ('$OMEGA 1 ; IVCL', [(['IVCL'], [1.0], False, False)]),
        (
            '$OMEGA DIAG(2) 1 ; V1 df\n 2 FIX ; VA2 __12\n',
            [
                (['V1'], [1.0], False, False),
                (['VA2'], [2.0], True, False),
            ],
        ),
        (
            '$OMEGA BLOCK(2) 1 ;IV1\n 2 ;CORR\n 3 ;IV2',
            [
                (['IV1', 'CORR', 'IV2'], [1.0, 2.0, 3.0], False, False),
            ],
        ),
    ],
)
def test_parse(parser, buf, results):
    recs = parser.parse(buf)
    rec = recs.records[0]
    assert rec.parse() == results


@pytest.mark.usefixtures('parser')
@pytest.mark.parametrize(
    'buf, exc_msg',
    [
        ('$OMEGA 0 ', 'If initial estimate for OMEGA is'),
        ('$OMEGA DIAG(1) 1 SD VARIANCE', 'Initial estimate for OMEGA cannot'),
        ('$OMEGA SD BLOCK(2) 0.1 0.001 0.1 STANDARD', 'Cannot specify either option'),
        ('$OMEGA CHOLESKY BLOCK(2) 0.1 VAR 0.001 \n  0.1 ', 'Cannot specify either option'),
        ('$OMEGA BLOCK(3) 0.1 \n  0.1', 'Wrong number of inits in BLOCK'),
    ],
)
def test_errors(parser, buf, exc_msg):
    recs = parser.parse(buf)
    rec = recs.records[0]
    with pytest.raises(ModelSyntaxError, match=exc_msg):
        pset, _, _ = rec.parse()


@pytest.mark.usefixtures('parser')
@pytest.mark.parametrize(
    'buf, params, results',
    [
        ('$OMEGA 1', [Parameter('OMEGA_1_1', init=2.0, lower=0.0)], '$OMEGA 2'),
        ('$OMEGA 1 SD', [Parameter('OMEGA_1_1', init=4.0, lower=0.0)], '$OMEGA 2 SD'),
        (
            '$OMEGA (1)x3\n;FTOL',
            [
                Parameter('OMEGA_1_1', init=4.0, lower=0.0),
                Parameter('OMEGA_2_2', init=4.0, lower=0.0),
                Parameter('OMEGA_3_3', init=4.0, lower=0.0),
            ],
            '$OMEGA (4)x3\n;FTOL',
        ),
        (
            '$OMEGA (1)x2 2',
            [
                Parameter('OMEGA_1_1', init=1.0, lower=0.0),
                Parameter('OMEGA_2_2', init=2.0, lower=0.0),
                Parameter('OMEGA_3_3', init=0.5, lower=0.0),
            ],
            '$OMEGA (1) (2) 0.5',
        ),
        (
            '$OMEGA DIAG(2) (1 VAR) (SD 2)',
            [
                Parameter('OMEGA_1_1', init=1.5, lower=0.0),
                Parameter('OMEGA_2_2', init=16.0, lower=0.0),
            ],
            '$OMEGA DIAG(2) (1.5 VAR) (SD 4)',
        ),
        (
            '$OMEGA BLOCK(2) 1 2 4',
            [
                Parameter('OMEGA_1_1', init=7.0, lower=0.0),
                Parameter('OMEGA_2_1', init=0.5),
                Parameter('OMEGA_2_2', init=8.0, lower=0.0),
            ],
            '$OMEGA BLOCK(2) 7 0.5 8',
        ),
        (
            '$OMEGA BLOCK(2)\n SD 1 0.5 ;COM \n 1\n',
            [
                Parameter('OMEGA_1_1', init=4.0, lower=0.0),
                Parameter('OMEGA_2_1', init=0.25),
                Parameter('OMEGA_2_2', init=9.0, lower=0.0),
            ],
            '$OMEGA BLOCK(2)\n SD 2 0.25 ;COM \n 3\n',
        ),
        (
            '$OMEGA CORR BLOCK(2)  1 0.5 1\n',
            [
                Parameter('OMEGA_1_1', init=4.0, lower=0.0),
                Parameter('OMEGA_2_1', init=1.0),
                Parameter('OMEGA_2_2', init=4.0, lower=0.0),
            ],
            '$OMEGA CORR BLOCK(2)  4 0.25 4\n',
        ),
        (
            '$OMEGA CORR BLOCK(2)  1 0.5 SD 1\n',
            [
                Parameter('OMEGA_1_1', init=4.0, lower=0.0),
                Parameter('OMEGA_2_1', init=1.0),
                Parameter('OMEGA_2_2', init=4.0, lower=0.0),
            ],
            '$OMEGA CORR BLOCK(2)  2 0.25 SD 2\n',
        ),
        (
            '$OMEGA BLOCK(2) 1 0.1 1\n CHOLESKY',
            [
                Parameter('OMEGA_1_1', init=0.64, lower=0.0),
                Parameter('OMEGA_2_1', init=-0.24),
                Parameter('OMEGA_2_2', init=0.58, lower=0.0),
            ],
            '$OMEGA BLOCK(2) 0.8 -0.3 0.7\n CHOLESKY',
        ),
        (
            '$OMEGA BLOCK(2) (1)x3\n',
            [
                Parameter('OMEGA_1_1', init=0.64, lower=0.0),
                Parameter('OMEGA_2_1', init=-0.24),
                Parameter('OMEGA_2_2', init=0.58, lower=0.0),
            ],
            '$OMEGA BLOCK(2) 0.64 -0.24 0.58\n',
        ),
        (
            '$OMEGA BLOCK(3) 1 ;CL\n0.1 1; V\n0.1 0.1 1; KA',
            [
                Parameter('OMEGA_1_1', init=1.0, lower=0.0),
                Parameter('OMEGA_2_1', init=0.2),
                Parameter('OMEGA_2_2', init=3.0, lower=0.0),
                Parameter('OMEGA_3_1', init=0.4),
                Parameter('OMEGA_3_2', init=0.5),
                Parameter('OMEGA_3_3', init=6.0, lower=0.0),
            ],
            '$OMEGA BLOCK(3) 1 ;CL\n0.2 3; V\n0.4 0.5 6; KA',
        ),
        ('$OMEGA BLOCK(2) SAME', [], '$OMEGA BLOCK(2) SAME'),
        ('$OMEGA BLOCK SAME', [], '$OMEGA BLOCK SAME'),
        (
            '$OMEGA 2 FIX 4 (FIX 6)',
            [
                Parameter('OMEGA_1_1', init=2.0, lower=0.0, fix=False),
                Parameter('OMEGA_2_2', init=4.0, lower=0.0),
                Parameter('OMEGA_3_3', init=6.0, lower=0.0, fix=True),
            ],
            '$OMEGA 2 4 (FIX 6)',
        ),
        (
            '$OMEGA 2 4 6 ;STRAML',
            [
                Parameter('OMEGA_1_1', init=2.0, lower=0.0, fix=True),
                Parameter('OMEGA_2_2', init=4.0, lower=0.0),
                Parameter('OMEGA_3_3', init=6.0, lower=0.0),
            ],
            '$OMEGA 2 FIX 4 6 ;STRAML',
        ),
        (
            '$OMEGA 2 FIX 4 6 ;STRAML',
            [
                Parameter('OMEGA_1_1', init=2.0, lower=0.0, fix=False),
                Parameter('OMEGA_2_2', init=4.0, lower=0.0, fix=True),
                Parameter('OMEGA_3_3', init=23.0, lower=0.0, fix=True),
            ],
            '$OMEGA 2 4 FIX 23 FIX ;STRAML',
        ),
        (
            '$OMEGA BLOCK(2) 1 .1 2 ;CLERMT',
            [
                Parameter('OMEGA_1_1', init=1.0, lower=0.0, fix=True),
                Parameter('OMEGA_2_1', init=0.1, fix=True),
                Parameter('OMEGA_2_2', init=2.0, lower=0.0, fix=True),
            ],
            '$OMEGA BLOCK(2) FIX 1 .1 2 ;CLERMT',
        ),
        (
            '$OMEGA BLOCK(2) 1 .1 FIX 1 ;CLERMT',
            [
                Parameter('OMEGA_1_1', init=1.0, lower=0.0, fix=False),
                Parameter('OMEGA_2_1', init=0.1, fix=False),
                Parameter('OMEGA_2_2', init=1.0, lower=0.0, fix=False),
            ],
            '$OMEGA BLOCK(2) 1 .1 1 ;CLERMT',
        ),
    ],
)
def test_update(parser, buf, params, results):
    rec = parser.parse(buf).records[0]
    pset = Parameters(params)
    newrec = rec.update(pset)
    assert str(newrec) == results


@pytest.mark.usefixtures('parser')
@pytest.mark.parametrize(
    'buf, params',
    [
        (
            '$OMEGA BLOCK(2) 1 .1 2 ;CLERMT',
            [
                Parameter('OMEGA_1_1', init=1.0, lower=0.0, fix=True),
                Parameter('OMEGA_2_1', init=0.1),
                Parameter('OMEGA_2_2', init=2.0, lower=0.0, fix=True),
            ],
        ),
    ],
)
def test_update_error(parser, buf, params):
    rec = parser.parse(buf).records[0]
    pset = Parameters(params)
    with pytest.raises(ValueError):
        rec.update(pset)


@pytest.mark.parametrize(
    'buf,remove,result',
    [
        (
            '$OMEGA BLOCK(3) 1 ;CL\n0.1 1; V\n0.1 0.1 1; KA',
            [(0, 0)],
            '$OMEGA BLOCK(2)\n1.0\n0.1 1.0\n',
        ),
        (
            '$OMEGA BLOCK(4) 1 ;CL\n0.2 3; V\n0.4 0.5 6; KA\n7 8 9 10\n',
            [(0, 1), (0, 3)],
            '$OMEGA BLOCK(2)\n1.0\n0.4 6.0\n',
        ),
        ('$OMEGA 1 2 3\n', [(1, 0)], '$OMEGA 1 3\n'),
        ('$OMEGA BLOCK(2) 1 2 3 FIX\n', [(0, 0)], '$OMEGA BLOCK(1) FIX\n3.0\n'),
    ],
)
def test_remove(parser, buf, remove, result):
    rec = parser.parse(buf).records[0]
    newrec = rec.remove(remove)
    assert str(newrec) == result
