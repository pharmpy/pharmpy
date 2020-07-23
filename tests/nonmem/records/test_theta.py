import pytest


@pytest.mark.usefixtures('parser')
@pytest.mark.parametrize('buf,results', [
    ('$THETA 0', [('THETA(1)', 0, -1000000, 1000000, False)]),
    ('$THETA    12.3 \n\n', [('THETA(1)', 12.3, -1000000, 1000000, False)]),
    ('$THETA  (0,0.00469) ; CL', [('THETA(1)', 0.00469, 0, 1000000, False)]),
    ('$THETA  (0,3) 2 FIXED (0,.6,1) 10 (-INF,-2.7,0)  (37 FIXED)\n 19 (0,1,2)x3', [
        ('THETA(1)', 3, 0, 1000000, False),
        ('THETA(2)', 2, -1000000, 1000000, True),
        ('THETA(3)', .6, 0, 1, False),
        ('THETA(4)', 10, -1000000, 1000000, False),
        ('THETA(5)', -2.7, -1000000, 0, False),
        ('THETA(6)', 37, -1000000, 1000000, True),
        ('THETA(7)', 19, -1000000, 1000000, False),
        ('THETA(8)', 1, 0, 2, False),
        ('THETA(9)', 1, 0, 2, False),
        ('THETA(10)', 1, 0, 2, False),
        ]),
    ('$THETA (0.00469555) FIX ; CL', [('THETA(1)', 0.00469555, -1000000, 1000000, True)]),
    ('$THETA\n ;; Model characteristics\n (0, 0.15, 0.6) ; Proportional\n',
        [('THETA(1)', 0.15, 0, 0.6, False)]),
    ('$THETA\n;\n1;COMMENTING', [('THETA(1)', 1, -1000000, 1000000, False)]),
    ('$THETA\n   ;CMT1;SAMELINE\n;ONLYCOMMENT\n\t;COMMENT2\n    1    \n',
        [('THETA(1)', 1, -1000000, 1000000, False)]),
    ('$THETA\n ;; Model characteristics\n  (0, 0.15, 0.6) ; Proportional error (Drug123)\n'
     '  (0, 1, 10)     ; Additive error (Drug123)\n',
        [('THETA(1)', 0.15, 0, 0.6, False), ('THETA(2)', 1, 0, 10, False)]),
    ('$THETA  (FIX FIX 0.4) ; CL', [('THETA(1)', 0.4, -1000000, 1000000, True)]),
    ('$THETA (FIX 1,   FIX FIX    1 FIX, 1  FIX) ; CL', [('THETA(1)', 1, 1, 1, True)]),
])
def test_parameters(parser, buf, results):
    recs = parser.parse(buf)
    rec = recs.records[0]
    pset = rec.parameters(1)
    assert len(pset) == len(results)
    assert len(pset) == len(rec)
    for res in results:
        name = res[0]
        init = res[1]
        lower = res[2]
        upper = res[3]
        fix = res[4]
        param = pset[name]
        assert param.name == name
        assert param.init == init
        assert param.lower == lower
        assert param.upper == upper
        assert param.fix == fix


def test_theta_num(parser):
    rec = parser.parse('$THETA 1').records[0]
    pset = rec.parameters(2)
    assert len(pset) == 1
    assert pset['THETA(2)'].init == 1


def test_update(parser):
    rec = parser.parse('$THETA 1').records[0]
    pset = rec.parameters(1)
    pset['THETA(1)'].init = 41
    rec.update(pset, 1)
    assert str(rec) == '$THETA 41'

    rec = parser.parse('$THETA 1 FIX').records[0]
    pset = rec.parameters(1)
    pset['THETA(1)'].fix = False
    rec.update(pset, 1)
    assert str(rec) == '$THETA 1'

    rec = parser.parse('$THETA (2, 2, 2)  FIX').records[0]
    pset = rec.parameters(1)
    pset['THETA(1)'].fix = False
    rec.update(pset, 1)
    assert str(rec) == '$THETA (2, 2, 2)'

    rec = parser.parse('$THETA (2, 2, 2 FIX)').records[0]
    pset = rec.parameters(1)
    pset['THETA(1)'].fix = False
    rec.update(pset, 1)
    assert str(rec) == '$THETA (2, 2, 2)'

    rec = parser.parse('$THETA (2, 2, 2)').records[0]
    pset = rec.parameters(1)
    pset['THETA(1)'].fix = True
    rec.update(pset, 1)
    assert str(rec) == '$THETA (2, 2, 2) FIX'

    rec = parser.parse('$THETA 1 2 3 ;CMT').records[0]
    pset = rec.parameters(1)
    pset['THETA(1)'].fix = True
    pset['THETA(3)'].fix = True
    rec.update(pset, 1)
    assert str(rec) == '$THETA 1 FIX 2 3 FIX ;CMT'
