import pytest

from pharmpy.model import ModelSyntaxError, Parameter
from pharmpy.model.external.nonmem.parsing import parse_thetas
from pharmpy.modeling import set_lower_bounds, set_upper_bounds

INF = float("inf")


@pytest.mark.usefixtures('parser')
@pytest.mark.parametrize(
    'buf,expected',
    [
        ('$THETA 0 FIX', [(None, 0, -INF, INF, True)]),
        ('$THETA (0,1,INF)', [(None, 1, 0, INF, False)]),
        ('$THETA    12.3 \n\n', [(None, 12.3, -INF, INF, False)]),
        ('$THETAS  (0,0.00469) ; CL', [('CL', 0.00469, 0, INF, False)]),
        (
            '$THETA  (0,3) 2 FIXED (0,.6,1) 10 (-INF,-2.7,0)  (37 FIXED)\n 19 (0,1,2)x3',
            [
                (None, 3, 0, INF, False),
                (None, 2, -INF, INF, True),
                (None, 0.6, 0, 1, False),
                (None, 10, -INF, INF, False),
                (None, -2.7, -INF, 0, False),
                (None, 37, -INF, INF, True),
                (None, 19, -INF, INF, False),
                (None, 1, 0, 2, False),
                (None, 1, 0, 2, False),
                (None, 1, 0, 2, False),
            ],
        ),
        (
            '$THETA (0.00469555) FIX ; CL',
            [('CL', 0.00469555, -INF, INF, True)],
        ),
        (
            '$THETA\n ;; Model characteristics\n (0, 0.15, 0.6) ; Proportional\n',
            [("Proportional", 0.15, 0, 0.6, False)],
        ),
        ('$THETA\n;\n1;COMMENTING', [('COMMENTING', 1, -INF, INF, False)]),
        (
            '$THETA\n   ;CMT1;SAMELINE\n;ONLYCOMMENT\n\t;COMMENT2\n    1    \n',
            [(None, 1, -INF, INF, False)],
        ),
        (
            '$THETA\n ;; Model characteristics\n  (0, 0.15, 0.6) ; Proportional error (Drug123)\n'
            '  (0, 1, 10)     ; Additive error (Drug123)\n',
            [('Proportional', 0.15, 0, 0.6, False), ('Additive', 1, 0, 10, False)],
        ),
        ('$THETA  (FIX FIX 0.4) ; CL', [('CL', 0.4, -INF, INF, True)]),
        ('$THETA (FIX 1,   FIX FIX    1 FIX, 1  FIX) ; CL', [('CL', 1, 1, 1, True)]),
        (
            '$THETA\n(0,0.105,)   ; RUV_CVFPG\n',
            [
                ('RUV_CVFPG', 0.105, 0, INF, False),
            ],
        ),
        (
            '$THETA\n(0,0.105,)   ; RUV_CVFPG\n',
            [
                ('RUV_CVFPG', 0.105, 0, INF, False),
            ],
        ),
        (
            '$THETA  (0,3) ; CL\n 2 FIXED ; V\n',
            [
                ('CL', 3, 0, INF, False),
                ('V', 2, -INF, INF, True),
            ],
        ),
        ('$THETA (1,1,1)\n', [(None, 1.0, 1.0, 1.0, True)]),
        ('$THETA (0,1) FIX\n', [(None, 1.0, 0.0, INF, True)]),
        ('$THETA (1,1 FIX)\n', [(None, 1.0, 1.0, INF, True)]),
        ('$THETA (1,1,1 FIX)\n', [(None, 1.0, 1.0, 1.0, True)]),
        ('$THETA 20000000\n', [(None, 20000000, -INF, INF, False)]),
    ],
)
def test_parameters(parser, buf, expected):
    control_stream = parser.parse("$PROB\n" + buf)
    names, bounds, inits, fixs = parse_thetas(control_stream)

    correct_names = [name for name, _, _, _, _ in expected]
    assert names == correct_names

    correct_inits = [init for _, init, _, _, _ in expected]
    assert inits == correct_inits

    correct_lower = [lower for _, _, lower, _, _ in expected]
    lower = [b[0] for b in bounds]
    assert lower == correct_lower

    correct_upper = [upper for _, _, _, upper, _ in expected]
    upper = [b[1] for b in bounds]
    assert upper == correct_upper

    correct_fix = [fix for _, _, _, _, fix in expected]
    assert fixs == correct_fix


@pytest.mark.usefixtures('parser')
@pytest.mark.parametrize(
    'buf',
    [
        ('$THETA (-1000001,0)'),
        ('$THETA (0,1,2000000)'),
        ('$THETA (0,1 FIX)'),
        ('$THETA (0,1,1 FIX)'),
        ('$THETA 0'),
        ('$THETA -1000000'),
        ('$THETA (0,1000000)'),
        ('$THETA (1,1)'),
    ],
)
def test_bad_thetas(parser, buf):
    control_stream = parser.parse("$PROB\n" + buf)
    with pytest.raises(ModelSyntaxError):
        _ = parse_thetas(control_stream)


@pytest.mark.usefixtures('parser')
@pytest.mark.parametrize(
    'buf,params,expected',
    [
        ('$THETA 1', [Parameter('THETA_1', 41)], '$THETA 41'),
        ('$THETA 1 FIX', [Parameter('THETA_1', 1, fix=False)], '$THETA 1'),
        (
            '$THETA (2, 2, 2) FIX',
            [Parameter('THETA_1', 2, lower=2, upper=2, fix=False)],
            '$THETA (2, 2, 2)',
        ),
        (
            '$THETA (2, 2, 2 FIX)',
            [Parameter('THETA_1', 2, lower=2, upper=2, fix=False)],
            '$THETA (2, 2, 2)',
        ),
        (
            '$THETA (2, 2, 2)',
            [Parameter('THETA_1', 2, lower=2, upper=2, fix=True)],
            '$THETA (2, 2, 2) FIX',
        ),
        (
            '$THETA 1 2 3 ;CMT',
            [
                Parameter('THETA_1', 1, fix=True),
                Parameter('THETA_2', 2),
                Parameter('THETA_3', 3, fix=True),
            ],
            '$THETA 1 FIX 2 3 FIX ;CMT',
        ),
    ],
)
def test_update(parser, buf, params, expected):
    rec = parser.parse(buf).records[0]
    rec = rec.update(params)
    assert str(rec) == expected


@pytest.mark.usefixtures('parser')
@pytest.mark.parametrize(
    'buf,remove,expected',
    [
        ('$THETA 1 2 3 ;CMT', [0, 2], '$THETA  2  ;CMT'),
    ],
)
def test_remove_theta(parser, buf, remove, expected):
    rec = parser.parse(buf).records[0]
    rec = rec.remove(remove)
    assert str(rec) == expected


def test_update_lower_bounds(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem/pheno.mod')
    model = set_lower_bounds(model, {'TVCL': -1})
    model = set_upper_bounds(model, {'TVCL': 30})
    expected = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV
$SUBROUTINE ADVAN1 TRANS2

$PK
CL=THETA(1)*EXP(ETA(1))
V=THETA(2)*EXP(ETA(2))
S1=V

$ERROR
Y=F+F*EPS(1)

$THETA (-1,0.00469307,30) ; TVCL
$THETA (0,1.00916) ; TVV
$OMEGA 0.0309626  ; IVCL
$OMEGA 0.031128  ; IVV
$SIGMA 0.013241

$ESTIMATION METHOD=1 INTERACTION\n"""
    assert model.code == expected

    model = set_lower_bounds(model, {'TVCL': 0})
    expected = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV
$SUBROUTINE ADVAN1 TRANS2

$PK
CL=THETA(1)*EXP(ETA(1))
V=THETA(2)*EXP(ETA(2))
S1=V

$ERROR
Y=F+F*EPS(1)

$THETA (0,0.00469307,30) ; TVCL
$THETA (0,1.00916) ; TVV
$OMEGA 0.0309626  ; IVCL
$OMEGA 0.031128  ; IVV
$SIGMA 0.013241

$ESTIMATION METHOD=1 INTERACTION\n"""
    assert model.code == expected
