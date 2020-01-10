import pytest
import sympy


@pytest.mark.usefixtures('parser')
@pytest.mark.parametrize('buf,results', [
    ('$OMEGA 1', [('OMEGA(1,1)', 1, 0, sympy.oo, False)]),
    ('$OMEGA   0.123 \n\n', [('OMEGA(1,1)', 0.123, 0, sympy.oo, False)]),
    ('$OMEGA   (0 FIX) ; CL', [('OMEGA(1,1)', 0, 0, 0, True)]),
    ('$OMEGA DIAG(2) 1 2 FIX', [
        ('OMEGA(1,1)', 1, 0, sympy.oo, False),
        ('OMEGA(2,2)', 2, 2, 2, True),
        ]),
    ('$OMEGA 1 2 3', [
        ('OMEGA(1,1)', 1, 0, sympy.oo, False),
        ('OMEGA(2,2)', 2, 0, sympy.oo, False),
        ('OMEGA(3,3)', 3, 0, sympy.oo, False),
        ]),
    ('$OMEGA 0.15 ;CL', [('OMEGA(1,1)', 0.15, 0, sympy.oo, False)]),
    ('$OMEGA 1 \n2 ; S   \n  3 ', [
        ('OMEGA(1,1)', 1, 0, sympy.oo, False),
        ('OMEGA(2,2)', 2, 0, sympy.oo, False),
        ('OMEGA(3,3)', 3, 0, sympy.oo, False),
        ]),
    ('$OMEGA 2 SD', [('OMEGA(1,1)', 4, 0, sympy.oo, False)]),
    ('$OMEGA ;CO\n (VAR 3)', [('OMEGA(1,1)', 3, 0, sympy.oo, False)]),
    ('$OMEGA (1)x2', [
        ('OMEGA(1,1)', 1, 0, sympy.oo, False),
        ('OMEGA(2,2)', 1, 0, sympy.oo, False),
        ]),
    ('$OMEGA BLOCK(2) 1 0.5 2', [
        ('OMEGA(1,1)', 1, 0, sympy.oo, False),
        ('OMEGA(2,1)', 0.5, -sympy.oo, sympy.oo, False),
        ('OMEGA(2,2)', 2, 0, sympy.oo, False),
        ]),
#    ('$OMEGA 0 ', [('OMEGA(1,1)', 0, 0, sympy.oo, False)]),
#    ('$OMEGA BLOCK(2) SAME', []),
#    ('$OMEGA BLOCK SAME(3)', []),
])
def test_parameters(parser, buf, results):
    recs = parser.parse(buf)
    rec = recs.records[0]
    pset = rec.parameters(1)
    assert len(pset) == len(results)
    for res in results:
        name = res[0]
        init = res[1]
        lower = res[2]
        upper = res[3]
        fix = res[4]
        param = pset[name]
        assert param.symbol.name == name
        assert param.init == init
        assert param.lower == lower
        assert param.upper == upper
        assert param.fix == fix


def test_parameters_offseted(parser):
    rec = parser.parse("$OMEGA 1").records[0]
    pset = rec.parameters(3)
    assert pset['OMEGA(3,3)'].init == 1
