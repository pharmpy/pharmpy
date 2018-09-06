# -*- encoding: utf-8 -*-

import numpy as np
from numpy.testing import assert_array_equal
import pytest

from pysn.parameter_model import Scalar


@pytest.mark.usefixtures('create_record')
@pytest.mark.parametrize('buf,matrix', [
    ('OMEGA 0', np.array((
        (Scalar(0),),
    ))),
    ('OMEGA   0.123 \n\n', np.array((
        (Scalar(0.123),),
    ))),
    ('OMEGA  (0 FIX) ; CL', np.array((
        (Scalar(0, True),),
    ))),
    ('OMEGA DIAG(2) 1 2 FIX', np.array((
        (Scalar(1), Scalar(0, True)),
        (Scalar(0, True), Scalar(2, True)),
    ))),
    ('OMEGA BLOCK(2) 1 0.5 2', np.array((
        (Scalar(1), Scalar(0.5)),
        (Scalar(0.5), Scalar(2)),
    ))),
])
def test_create(create_record, buf, matrix):
    rec = create_record(buf)
    assert rec.name == 'OMEGA'
    assert_array_equal(rec.block, matrix)


def test_create_replicate(create_record):
    single = create_record('OMEGA 2 2 2 2 (0.1) (0.1) (0.1)'
                           '       (0.5 FIXED) (0.5 FIXED)')
    multi = create_record('OMEGA (2)x4 (0.1)x3 (0.5 FIXED)x2')
    assert_array_equal(single.block, multi.block)
