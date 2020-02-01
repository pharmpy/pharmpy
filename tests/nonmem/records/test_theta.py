import pytest


@pytest.mark.usefixtures('parser')
@pytest.mark.parametrize('buf,results', [
    ('$THETA 0', [('THETA(1)', 0, -1000000, 1000000, False)]),
    ('$THETA    12.3 \n\n', [('THETA(1)', 12.3, -1000000, 1000000, False)]),
    ('$THETA  (0,0.00469) ; CL', [('THETA(1)', 0.00469, 0, 1000000, False)]),
    ('$THETA  (0,3) 2 FIXED (0,.6,1) 10 (-INF,-2.7,0)  (37 FIXED)\n 19 (0,1,2)x3', [
        ('THETA(1)', 3, 0, 1000000, False),
        ('THETA(2)', 2, 2, 2, True),
        ('THETA(3)', .6, 0, 1, False),
        ('THETA(4)', 10, -1000000, 1000000, False),
        ('THETA(5)', -2.7, -1000000, 0, False),
        ('THETA(6)', 37, 37, 37, True),
        ('THETA(7)', 19, -1000000, 1000000, False),
        ('THETA(8)', 1, 0, 2, False),
        ('THETA(9)', 1, 0, 2, False),
        ('THETA(10)', 1, 0, 2, False),
        ])
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
