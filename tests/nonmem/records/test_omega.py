import pytest
import sympy

from pharmpy.model import ModelFormatError


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
    ('$OMEGA BLOCK(2) SAME', []),
    ('$OMEGA BLOCK SAME(3)', []),
    ('$OMEGA BLOCK(2) 1 (0.1)x2', [
        ('OMEGA(1,1)', 1, 0, sympy.oo, False),
        ('OMEGA(2,1)', 0.1, -sympy.oo, sympy.oo, False),
        ('OMEGA(2,2)', 0.1, 0, sympy.oo, False),
        ]),
    ('$OMEGA BLOCK(2) CHOLESKY 0.8 -0.3 0.7', [
        ('OMEGA(1,1)', 0.64, 0, sympy.oo, False),
        ('OMEGA(2,1)', -0.24, -sympy.oo, sympy.oo, False),
        ('OMEGA(2,2)', 0.58, 0, sympy.oo, False),
        ]),
    ('$OMEGA BLOCK(2) SD 0.8 -0.394 0.762 CORR', [
        ('OMEGA(1,1)', 0.64, 0, sympy.oo, False),
        ('OMEGA(2,1)', -0.2401824, -sympy.oo, sympy.oo, False),
        ('OMEGA(2,2)', 0.580644, 0, sympy.oo, False),
        ]),
])
def test_parameters(parser, buf, results):
    recs = parser.parse(buf)
    rec = recs.records[0]
    pset, _ = rec.parameters(1)
    assert len(pset) == len(results)
    for res in results:
        name = res[0]
        init = res[1]
        lower = res[2]
        upper = res[3]
        fix = res[4]
        param = pset[name]
        assert param.symbol.name == name
        assert pytest.approx(param.init, 0.00000000000001) == init
        assert param.lower == lower
        assert param.upper == upper
        assert param.fix == fix


@pytest.mark.usefixtures('parser')
@pytest.mark.parametrize('buf', [
    ('$OMEGA 0 '),
    ('$OMEGA DIAG(1) 1 SD VARIANCE'),
    ('$OMEGA SD BLOCK(2) 0.1 0.001 0.1 STANDARD'),
    ('$OMEGA CHOLESKY BLOCK(2) 0.1 VAR 0.001 \n  0.1 '),
])
def test_errors(parser, buf):
    recs = parser.parse(buf)
    rec = recs.records[0]
    with pytest.raises(ModelFormatError):
        pset, _ = rec.parameters(1)


def test_parameters_offseted(parser):
    rec = parser.parse("$OMEGA 1").records[0]
    pset, _ = rec.parameters(3)
    assert pset['OMEGA(3,3)'].init == 1


def test_update(parser):
    rec = parser.parse('$OMEGA 1').records[0]
    pset, _ = rec.parameters(1)
    pset['OMEGA(1,1)'].init = 2
    rec.update(pset, 1)
    assert str(rec) == '$OMEGA 2.0'

    rec = parser.parse('$OMEGA 1 SD').records[0]
    pset, _ = rec.parameters(1)
    pset['OMEGA(1,1)'].init = 4
    rec.update(pset, 1)
    assert str(rec) == '$OMEGA 2.0 SD'

    rec = parser.parse('$OMEGA (1)x3\n;FTOL').records[0]
    pset, _ = rec.parameters(1)
    pset['OMEGA(1,1)'].init = 4
    pset['OMEGA(2,2)'].init = 4
    pset['OMEGA(3,3)'].init = 4
    rec.update(pset, 1)
    assert str(rec) == '$OMEGA (4.0)x3\n;FTOL'

    rec = parser.parse('$OMEGA (1)x2 2').records[0]
    pset, _ = rec.parameters(1)
    pset['OMEGA(1,1)'].init = 1
    pset['OMEGA(2,2)'].init = 2
    pset['OMEGA(3,3)'].init = 0.5
    rec.update(pset, 1)
    assert str(rec) == '$OMEGA (1.0) (2.0) 0.5'

    rec = parser.parse("$OMEGA DIAG(2) (1 VAR) (SD 2)").records[0]
    pset, _ = rec.parameters(1)
    pset['OMEGA(1,1)'].init = 1.5
    pset['OMEGA(2,2)'].init = 16
    rec.update(pset, 1)
    assert str(rec) == '$OMEGA DIAG(2) (1.5 VAR) (SD 4.0)'

    rec = parser.parse("$OMEGA BLOCK(2) 1 2 4").records[0]
    pset, _ = rec.parameters(1)
    pset['OMEGA(1,1)'].init = 7
    pset['OMEGA(2,1)'].init = 0.5
    pset['OMEGA(2,2)'].init = 8
    rec.update(pset, 1)
    assert str(rec) == '$OMEGA BLOCK(2) 7.0 0.5 8.0'

    rec = parser.parse("$OMEGA BLOCK(2)\n SD 1 0.5 ;COM \n 1\n").records[0]
    pset, _ = rec.parameters(1)
    pset['OMEGA(1,1)'].init = 4
    pset['OMEGA(2,1)'].init = 0.25
    pset['OMEGA(2,2)'].init = 9
    rec.update(pset, 1)
    assert str(rec) == '$OMEGA BLOCK(2)\n SD 2.0 0.25 ;COM \n 3.0\n'

    rec = parser.parse("$OMEGA CORR BLOCK(2)  1 0.5 1\n").records[0]
    pset, _ = rec.parameters(1)
    pset['OMEGA(1,1)'].init = 4
    pset['OMEGA(2,1)'].init = 1
    pset['OMEGA(2,2)'].init = 4
    rec.update(pset, 1)
    assert str(rec) == '$OMEGA CORR BLOCK(2)  4.0 0.25 4.0\n'

    rec = parser.parse("$OMEGA CORR BLOCK(2)  1 0.5 SD 1\n").records[0]
    pset, _ = rec.parameters(1)
    pset['OMEGA(1,1)'].init = 4
    pset['OMEGA(2,1)'].init = 1
    pset['OMEGA(2,2)'].init = 4
    rec.update(pset, 1)
    assert str(rec) == '$OMEGA CORR BLOCK(2)  2.0 0.25 SD 2.0\n'

    rec = parser.parse("$OMEGA CORR BLOCK(2)  1 0.5 SD 1\n").records[0]
    pset, _ = rec.parameters(1)
    pset['OMEGA(1,1)'].init = 1
    pset['OMEGA(2,1)'].init = 4
    pset['OMEGA(2,2)'].init = 1
    with pytest.raises(ValueError):
        rec.update(pset, 1)

    rec = parser.parse("$OMEGA BLOCK(2) 1 0.1 1\n CHOLESKY").records[0]
    pset, _ = rec.parameters(1)
    pset['OMEGA(1,1)'].init = 0.64
    pset['OMEGA(2,1)'].init = -0.24
    pset['OMEGA(2,2)'].init = 0.58
    rec.update(pset, 1)
    assert str(rec) == '$OMEGA BLOCK(2) 0.8 -0.3 0.7\n CHOLESKY'

    rec = parser.parse("$OMEGA BLOCK(3) 1 ;CL\n0.1 1; V\n0.1 0.1 1; KA").records[0]
    pset, _ = rec.parameters(1)
    pset['OMEGA(1,1)'].init = 1
    pset['OMEGA(2,1)'].init = 0.2
    pset['OMEGA(2,2)'].init = 3
    pset['OMEGA(3,1)'].init = 0.4
    pset['OMEGA(3,2)'].init = 0.5
    pset['OMEGA(3,3)'].init = 6
    rec.update(pset, 1)
    assert str(rec) == '$OMEGA BLOCK(3) 1.0 ;CL\n0.2 3.0; V\n0.4 0.5 6.0; KA'
