# -*- encoding: utf-8 -*-

import pytest

@pytest.mark.usefixtures('parser')
@pytest.mark.parametrize('buf,results', [
    ('$OMEGA 1', [('OMEGA(1,1)', 1, 0, 1000000, False)]),
#    ('$THETA    12.3 \n\n', [('THETA(1)', 12.3, -1000000, 1000000, False)]),
#    ('$THETA  (0,0.00469) ; CL', [('THETA(1)', 0.00469, 0, 1000000, False)]),
#    ('$THETA  (0,3) 2 FIXED (0,.6,1) 10 (-INF,-2.7,0)  (37 FIXED)\n 19 (0,1,2)x3', [
#        ('THETA(1)', 3, 0, 1000000, False),
#        ('THETA(2)', 2, 2, 2, True),
#        ('THETA(3)', .6, 0, 1, False),
#        ('THETA(4)', 10, -1000000, 1000000, False),
#        ('THETA(5)', -2.7, -1000000, 0, False),
#        ('THETA(6)', 37, 37, 37, True),
#        ('THETA(7)', 19, -1000000, 1000000, False),
#        ('THETA(8)', 1, 0, 2, False),
#        ('THETA(9)', 1, 0, 2, False),
#        ('THETA(10)', 1, 0, 2, False),
#        ])
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

"""
@pytest.mark.usefixtures('create_record')
@pytest.mark.parametrize('buf,params', [
    ('OMEGA 0', [
        Var(0),
    ]),
    ('OMEGA   0.123 \n\n', [
        Var(0.123),
    ]),
    ('OMEGA  (0 FIX) ; CL', [
        Var(0, fix=True),
    ]),
    ('OMEGA DIAG(2) 1 2 FIX', [
        Var(1), Var(2, fix=True),
    ]),
    ('OMEGA BLOCK(2) 1 0.5 2', [
        Var(1),
        Covar(0.5), Var(2),
    ])
])
def test_create(create_record, buf, params):
    rec = create_record(buf)
    assert rec.name == 'OMEGA'
    assert rec.matrix.params == params


def test_create_replicate(create_record):
    single = create_record('OMEGA 2 2 2 2 (0.1) (0.1) (0.1)'
                           '       (0.5 FIXED) (0.5 FIXED)')
    multi = create_record('OMEGA (2)x4 (0.1)x3 (0.5 FIXED)x2')
    assert single.matrix == multi.matrix
"""
