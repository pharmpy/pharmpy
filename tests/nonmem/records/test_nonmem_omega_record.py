# -*- encoding: utf-8 -*-

import pytest

from pharmpy.parameters import Covar
from pharmpy.parameters import Var


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
