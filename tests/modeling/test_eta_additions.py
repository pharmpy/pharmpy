from operator import add, mul

import pytest
import sympy
from sympy import Symbol as S

from pharmpy.modeling import add_iov, remove_iov
from pharmpy.modeling.eta_additions import EtaAddition


@pytest.mark.parametrize(
    'addition,expression',
    [
        (EtaAddition.exponential(add), S('CL') + sympy.exp(S('eta_new'))),
        (EtaAddition.exponential(mul), S('CL') * sympy.exp(S('eta_new'))),
    ],
)
def test_apply(addition, expression):
    addition.apply(original='CL', eta='eta_new')

    assert addition.template == expression


@pytest.mark.parametrize(
    'eta_iov_1,eta_iov_2',
    [
        ('ETA_IOV_1_1', 'ETA_IOV_2_1'),
        ('ETA_IOV_1_1', 'ETA_IOV_3_1'),
        ('ETA_IOV_2_1', 'ETA_IOV_3_1'),
        ('ETA_IOV_2_1', 'ETA_IOV_1_1'),
        ('ETA_IOV_3_1', 'ETA_IOV_1_1'),
        ('ETA_IOV_3_1', 'ETA_IOV_2_1'),
    ],
)
def test_regression_code_record(load_model_for_test, testdata, eta_iov_1, eta_iov_2):
    """This is a regression test for NONMEM CodeRecord statements property round-trip
    serialization.
    See https://github.com/pharmpy/pharmpy/pull/771
    """

    model_no_iov = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox2.mod')
    model = add_iov(model_no_iov, occ="VISI")

    # remove the first IOV, can reproduce the same issue
    model_r1 = remove_iov(model, to_remove=[eta_iov_1])

    # remove the second IOV, can reproduce the same issue
    model_r1r2 = remove_iov(model_r1, to_remove=[eta_iov_2])

    # remove the first and second IOV
    model_r12 = remove_iov(model, to_remove=[eta_iov_1, eta_iov_2])

    assert model_r12 == model_r1r2
    assert model_r12.model_code == model_r1r2.model_code

    model_r2r1 = remove_iov(model, to_remove=[eta_iov_2])
    model_r2r1 = remove_iov(model_r2r1, to_remove=[eta_iov_1])

    assert model_r1r2 == model_r2r1
    assert model_r1r2.model_code == model_r2r1.model_code
