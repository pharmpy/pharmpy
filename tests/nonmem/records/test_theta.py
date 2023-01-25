import pytest

from pharmpy.model import Parameter, Parameters


@pytest.mark.usefixtures('parser')
@pytest.mark.parametrize(
    'buf,expected',
    [
        ('$THETA 0', [(None, 0, -1000000, 1000000, False)]),
        ('$THETA (0,1,INF)', [(None, 1, 0, 1000000, False)]),
        ('$THETA    12.3 \n\n', [(None, 12.3, -1000000, 1000000, False)]),
        ('$THETA  (0,0.00469) ; CL', [('CL', 0.00469, 0, 1000000, False)]),
        (
            '$THETA  (0,3) 2 FIXED (0,.6,1) 10 (-INF,-2.7,0)  (37 FIXED)\n 19 (0,1,2)x3',
            [
                (None, 3, 0, 1000000, False),
                (None, 2, -1000000, 1000000, True),
                (None, 0.6, 0, 1, False),
                (None, 10, -1000000, 1000000, False),
                (None, -2.7, -1000000, 0, False),
                (None, 37, -1000000, 1000000, True),
                (None, 19, -1000000, 1000000, False),
                (None, 1, 0, 2, False),
                (None, 1, 0, 2, False),
                (None, 1, 0, 2, False),
            ],
        ),
        (
            '$THETA (0.00469555) FIX ; CL',
            [('CL', 0.00469555, -1000000, 1000000, True)],
        ),
        (
            '$THETA\n ;; Model characteristics\n (0, 0.15, 0.6) ; Proportional\n',
            [("Proportional", 0.15, 0, 0.6, False)],
        ),
        ('$THETA\n;\n1;COMMENTING', [('COMMENTING', 1, -1000000, 1000000, False)]),
        (
            '$THETA\n   ;CMT1;SAMELINE\n;ONLYCOMMENT\n\t;COMMENT2\n    1    \n',
            [(None, 1, -1000000, 1000000, False)],
        ),
        (
            '$THETA\n ;; Model characteristics\n  (0, 0.15, 0.6) ; Proportional error (Drug123)\n'
            '  (0, 1, 10)     ; Additive error (Drug123)\n',
            [('Proportional', 0.15, 0, 0.6, False), ('Additive', 1, 0, 10, False)],
        ),
        ('$THETA  (FIX FIX 0.4) ; CL', [('CL', 0.4, -1000000, 1000000, True)]),
        ('$THETA (FIX 1,   FIX FIX    1 FIX, 1  FIX) ; CL', [('CL', 1, 1, 1, True)]),
        (
            '$THETA\n(0,0.105,)   ; RUV_CVFPG\n',
            [
                ('RUV_CVFPG', 0.105, 0, 1000000, False),
            ],
        ),
        (
            '$THETA\n(0,0.105,)   ; RUV_CVFPG\n',
            [
                ('RUV_CVFPG', 0.105, 0, 1000000, False),
            ],
        ),
        (
            '$THETA  (0,3) ; CL\n 2 FIXED ; V\n',
            [
                ('CL', 3, 0, 1000000, False),
                ('V', 2, -1000000, 1000000, True),
            ],
        ),
    ],
)
def test_parameters(parser, buf, expected):
    control_stream = parser.parse(buf)
    rec = control_stream.records[0]

    correct_names = [name for name, _, _, _, _ in expected]
    assert rec.comment_names == correct_names

    correct_inits = [init for _, init, _, _, _ in expected]
    assert rec.inits == correct_inits

    correct_lower = [lower for _, _, lower, _, _ in expected]
    assert [lower for lower, _ in rec.bounds] == correct_lower

    correct_upper = [upper for _, _, _, upper, _ in expected]
    assert [upper for _, upper in rec.bounds] == correct_upper

    correct_fix = [fix for _, _, _, _, fix in expected]
    assert rec.fixs == correct_fix


def test_update(parser):
    rec = parser.parse('$THETA 1').records[0]
    pset = Parameters((Parameter('THETA_1', 41),))
    rec = rec.update(pset)
    assert str(rec) == '$THETA 41'

    rec = parser.parse('$THETA 1 FIX').records[0]
    pset = Parameters((Parameter('THETA(1)', 1, fix=False),))
    rec = rec.update(pset)
    assert str(rec) == '$THETA 1'

    rec = parser.parse('$THETA (2, 2, 2)  FIX').records[0]
    pset = Parameters((Parameter('THETA(1)', 2, lower=2, upper=2, fix=False),))
    rec = rec.update(pset)
    assert str(rec) == '$THETA (2, 2, 2)'

    rec = parser.parse('$THETA (2, 2, 2 FIX)').records[0]
    pset = Parameters((Parameter('THETA(1)', 2, lower=2, upper=2, fix=False),))
    rec = rec.update(pset)
    assert str(rec) == '$THETA (2, 2, 2)'

    rec = parser.parse('$THETA (2, 2, 2)').records[0]
    pset = Parameters((Parameter('THETA(1)', 2, lower=2, upper=2, fix=True),))
    rec = rec.update(pset)
    assert str(rec) == '$THETA (2, 2, 2) FIX'

    rec = parser.parse('$THETA 1 2 3 ;CMT').records[0]
    pset = Parameters(
        (
            Parameter('THETA(1)', 1, fix=True),
            Parameter('THETA(2)', 2),
            Parameter('THETA(3)', 3, fix=True),
        )
    )
    rec = rec.update(pset)
    assert str(rec) == '$THETA 1 FIX 2 3 FIX ;CMT'


def test_remove_theta(parser):
    rec = parser.parse('$THETA 1 2 3 ;CMT').records[0]
    rec = rec.remove([0, 2])
    assert str(rec) == '$THETA  2  ;CMT'
