from operator import add, mul

import pytest
import sympy

from pharmpy.modeling import add_iov, read_model, remove_iov
from pharmpy.modeling.eta_additions import EtaAddition
from pharmpy.symbols import symbol as S


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
def test_regression_code_record(testdata, eta_iov_1, eta_iov_2):
    """This is a regression test for NONMEM CodeRecord statements property round-trip
    serialization.
    See https://github.com/pharmpy/pharmpy/pull/771
    """

    model_no_iov = read_model(testdata / 'nonmem' / 'models' / 'mox2.mod')
    model = model_no_iov.copy()
    add_iov(model, occ="VISI")
    model.model_code  # this triggers AST update

    # remove the first IOV, can reproduce the same issue
    model_r1 = model.copy()
    remove_iov(model_r1, to_remove=[eta_iov_1])
    model_r1.model_code  # this triggers AST update

    # remove the second IOV, can reproduce the same issue
    model_r1r2 = model_r1.copy()
    remove_iov(model_r1r2, to_remove=[eta_iov_2])
    model_r1r2.model_code  # this triggers AST update

    # remove the first and second IOV
    model_r12 = model.copy()
    remove_iov(model_r12, to_remove=[eta_iov_1, eta_iov_2])
    model_r12.model_code  # this triggers AST update

    assert model_r12 == model_r1r2
    assert model_r12.model_code == model_r1r2.model_code

    model_r2r1 = model.copy()
    remove_iov(model_r2r1, to_remove=[eta_iov_2])
    model_r2r1.model_code  # this triggers AST update
    remove_iov(model_r2r1, to_remove=[eta_iov_1])
    model_r2r1.model_code  # this triggers AST update

    assert model_r1r2 == model_r2r1
    assert model_r1r2.model_code == model_r2r1.model_code
