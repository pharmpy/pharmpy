# -*- encoding: utf-8 -*-

import numpy as np
from numpy.testing import assert_array_equal


import pytest


@pytest.mark.usefixtures('create_record')
@pytest.mark.parametrize('buf,matrix', [
    ('OMEGA 0', np.diagflat((0,))),
    ('OMEGA   0.123 \n\n', np.diagflat((0.123,))),
    ('OMEGA  (0 FIX) ; CL', np.diagflat((0,))),
    ('OMEGA DIAG(2) 1 2 FIX', np.diagflat((1, 2))),
    ('OMEGA BLOCK(2) 1 0.5 2', np.array(((1, 0.5), (0.5, 2)))),
])
def test_create(create_record, buf, matrix):
    rec = create_record(buf)
    assert rec.name == 'OMEGA'
    assert_array_equal(rec.block[0], matrix)
