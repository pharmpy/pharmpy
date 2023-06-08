import os
import re
import shutil
from operator import add, mul

import pytest
from sympy import Symbol as S

from pharmpy.deps import sympy
from pharmpy.internals.fs.cwd import chdir
from pharmpy.model import NormalDistribution
from pharmpy.modeling import (
    add_iiv,
    add_iov,
    add_peripheral_compartment,
    add_pk_iiv,
    create_joint_distribution,
    remove_iiv,
    remove_iov,
    set_first_order_absorption,
    set_transit_compartments,
    set_zero_order_elimination,
    split_joint_distribution,
    transform_etas_boxcox,
    transform_etas_john_draper,
    transform_etas_tdist,
    update_initial_individual_estimates,
)
from pharmpy.modeling.parameter_variability import (
    EtaAddition,
    EtaTransformation,
    _choose_cov_param_init,
)
from pharmpy.tools import read_modelfit_results


@pytest.mark.parametrize(
    'parameter, expression, operation, eta_name, buf_new, no_of_omega_recs',
    [
        ('S1', 'exp', '+', None, 'V=TVV*EXP(ETA(2))\nS1 = V + EXP(ETA_S1)', 2),
        ('S1', 'exp', '*', None, 'V=TVV*EXP(ETA(2))\nS1 = V*EXP(ETA_S1)', 2),
        ('V', 'exp', '+', None, 'V = TVV*EXP(ETA(2)) + EXP(ETA_V)\nS1=V', 2),
        ('S1', 'add', None, None, 'V=TVV*EXP(ETA(2))\nS1 = V + ETA_S1', 2),
        ('S1', 'prop', None, None, 'V=TVV*EXP(ETA(2))\nS1 = ETA_S1*V', 2),
        ('S1', 'log', None, None, 'V=TVV*EXP(ETA(2))\nS1 = V*EXP(ETA_S1)/(EXP(ETA_S1) + 1)', 2),
        ('S1', 'eta_new', '+', None, 'V=TVV*EXP(ETA(2))\nS1 = V + ETA_S1', 2),
        ('S1', 'eta_new**2', '+', None, 'V=TVV*EXP(ETA(2))\nS1 = V + ETA_S1**2', 2),
        ('S1', 'exp', '+', 'ETA(3)', 'V=TVV*EXP(ETA(2))\nS1 = V + EXP(ETA(3))', 2),
        (
            ['V', 'S1'],
            'exp',
            '+',
            None,
            'V = TVV*EXP(ETA(2)) + EXP(ETA_V)\nS1 = V + EXP(ETA_S1)',
            3,
        ),
        (
            ['V', 'S1'],
            'exp',
            '+',
            ['new_eta1', 'new_eta2'],
            'V = TVV*EXP(ETA(2)) + EXP(NEW_ETA1)\nS1 = V + EXP(NEW_ETA2)',
            3,
        ),
    ],
)
def test_add_iiv(
    load_model_for_test,
    pheno_path,
    parameter,
    expression,
    operation,
    eta_name,
    buf_new,
    no_of_omega_recs,
):
    model = load_model_for_test(pheno_path)

    model = add_iiv(
        model,
        list_of_parameters=parameter,
        expression=expression,
        operation=operation,
        eta_names=eta_name,
    )

    etas = model.random_variables.etas.names

    assert eta_name is None or set(eta_name).intersection(etas) or eta_name in etas

    rec_ref = (
        f'$PK\n'
        f'IF(AMT.GT.0) BTIME=TIME\n'
        f'TAD=TIME-BTIME\n'
        f'TVCL=THETA(1)*WGT\n'
        f'TVV=THETA(2)*WGT\n'
        f'IF(APGR.LT.5) TVV=TVV*(1+THETA(3))\n'
        f'CL=TVCL*EXP(ETA(1))\n'
        f'{buf_new}\n\n'
    )

    assert str(model.internals.control_stream.get_pred_pk_record()) == rec_ref

    omega_rec = model.internals.control_stream.get_records('OMEGA')

    assert len(omega_rec) == no_of_omega_recs
    assert '$OMEGA  0.09 ; IIV_' in str(omega_rec[-1])


def test_add_iiv_missing_param(load_model_for_test, pheno_path):
    model = load_model_for_test(pheno_path)
    with pytest.raises(ValueError):
        add_iiv(model, 'non_existing_param', 'add')


@pytest.mark.parametrize(
    'path, occ, etas, eta_names, pk_start_ref, pk_end_ref, omega_ref, distribution',
    [
        (
            'nonmem/pheno_real.mod',
            'FA1',
            ['ETA_1'],
            None,
            'IOV_1 = 0\n'
            'IF (FA1.EQ.0) IOV_1 = ETA_IOV_1_1\n'
            'IF (FA1.EQ.1) IOV_1 = ETA_IOV_1_2\n'
            'ETAI1 = IOV_1 + ETA(1)\n',
            'CL = TVCL*EXP(ETAI1)\n' 'V=TVV*EXP(ETA(2))\n' 'S1=V\n',
            '$OMEGA  BLOCK(1)\n' '0.00309626 ; OMEGA_IOV_1\n' '$OMEGA  BLOCK(1) SAME\n',
            'disjoint',
        ),
        (
            'nonmem/pheno_real.mod',
            'FA1',
            'ETA_1',
            None,
            'IOV_1 = 0\n'
            'IF (FA1.EQ.0) IOV_1 = ETA_IOV_1_1\n'
            'IF (FA1.EQ.1) IOV_1 = ETA_IOV_1_2\n'
            'ETAI1 = IOV_1 + ETA(1)\n',
            'CL = TVCL*EXP(ETAI1)\n' 'V=TVV*EXP(ETA(2))\n' 'S1=V\n',
            '$OMEGA  BLOCK(1)\n' '0.00309626 ; OMEGA_IOV_1\n' '$OMEGA  BLOCK(1) SAME\n',
            'disjoint',
        ),
        (
            'nonmem/pheno_real.mod',
            'FA1',
            'ETA_1',
            None,
            'IOV_1 = 0\n'
            'IF (FA1.EQ.0) IOV_1 = ETA_IOV_1_1\n'
            'IF (FA1.EQ.1) IOV_1 = ETA_IOV_1_2\n'
            'ETAI1 = IOV_1 + ETA(1)\n',
            'CL = TVCL*EXP(ETAI1)\n' 'V=TVV*EXP(ETA(2))\n' 'S1=V\n',
            '$OMEGA  BLOCK(1)\n' '0.00309626 ; OMEGA_IOV_1\n' '$OMEGA  BLOCK(1) SAME\n',
            'joint',
        ),
        (
            'nonmem/pheno_real.mod',
            'FA1',
            ['CL', 'ETA_1'],
            None,
            'IOV_1 = 0\n'
            'IF (FA1.EQ.0) IOV_1 = ETA_IOV_1_1\n'
            'IF (FA1.EQ.1) IOV_1 = ETA_IOV_1_2\n'
            'ETAI1 = IOV_1 + ETA(1)\n',
            'CL = TVCL*EXP(ETAI1)\n' 'V=TVV*EXP(ETA(2))\n' 'S1=V\n',
            '$OMEGA  BLOCK(1)\n' '0.00309626 ; OMEGA_IOV_1\n' '$OMEGA  BLOCK(1) SAME\n',
            'joint',
        ),
        (
            'nonmem/pheno_real.mod',
            'FA1',
            ['ETA_1', 'CL'],
            None,
            'IOV_1 = 0\n'
            'IF (FA1.EQ.0) IOV_1 = ETA_IOV_1_1\n'
            'IF (FA1.EQ.1) IOV_1 = ETA_IOV_1_2\n'
            'ETAI1 = IOV_1 + ETA(1)\n',
            'CL = TVCL*EXP(ETAI1)\n' 'V=TVV*EXP(ETA(2))\n' 'S1=V\n',
            '$OMEGA  BLOCK(1)\n' '0.00309626 ; OMEGA_IOV_1\n' '$OMEGA  BLOCK(1) SAME\n',
            'joint',
        ),
        (
            'nonmem/pheno_real.mod',
            'FA1',
            [['CL', 'ETA_1']],
            None,
            'IOV_1 = 0\n'
            'IF (FA1.EQ.0) IOV_1 = ETA_IOV_1_1\n'
            'IF (FA1.EQ.1) IOV_1 = ETA_IOV_1_2\n'
            'ETAI1 = IOV_1 + ETA(1)\n',
            'CL = TVCL*EXP(ETAI1)\n' 'V=TVV*EXP(ETA(2))\n' 'S1=V\n',
            '$OMEGA  BLOCK(1)\n' '0.00309626 ; OMEGA_IOV_1\n' '$OMEGA  BLOCK(1) SAME\n',
            'explicit',
        ),
        (
            'nonmem/pheno_real.mod',
            'FA1',
            [['ETA_1', 'CL']],
            None,
            'IOV_1 = 0\n'
            'IF (FA1.EQ.0) IOV_1 = ETA_IOV_1_1\n'
            'IF (FA1.EQ.1) IOV_1 = ETA_IOV_1_2\n'
            'ETAI1 = IOV_1 + ETA(1)\n',
            'CL = TVCL*EXP(ETAI1)\n' 'V=TVV*EXP(ETA(2))\n' 'S1=V\n',
            '$OMEGA  BLOCK(1)\n' '0.00309626 ; OMEGA_IOV_1\n' '$OMEGA  BLOCK(1) SAME\n',
            'explicit',
        ),
        (
            'nonmem/pheno_real.mod',
            'FA1',
            None,
            None,
            'IOV_1 = 0\n'
            'IF (FA1.EQ.0) IOV_1 = ETA_IOV_1_1\n'
            'IF (FA1.EQ.1) IOV_1 = ETA_IOV_1_2\n'
            'IOV_2 = 0\n'
            'IF (FA1.EQ.0) IOV_2 = ETA_IOV_2_1\n'
            'IF (FA1.EQ.1) IOV_2 = ETA_IOV_2_2\n'
            'ETAI1 = IOV_1 + ETA(1)\n'
            'ETAI2 = IOV_2 + ETA(2)\n',
            'CL = TVCL*EXP(ETAI1)\n' 'V = TVV*EXP(ETAI2)\n' 'S1=V\n',
            '$OMEGA  BLOCK(1)\n'
            '0.00309626 ; OMEGA_IOV_1\n'
            '$OMEGA  BLOCK(1) SAME\n'
            '$OMEGA  BLOCK(1)\n'
            '0.0031128 ; OMEGA_IOV_2\n'
            '$OMEGA  BLOCK(1) SAME\n',
            'disjoint',
        ),
        (
            'nonmem/pheno_real.mod',
            'FA1',
            None,
            None,
            'IOV_1 = 0\n'
            'IF (FA1.EQ.0) IOV_1 = ETA_IOV_1_1\n'
            'IF (FA1.EQ.1) IOV_1 = ETA_IOV_1_2\n'
            'IOV_2 = 0\n'
            'IF (FA1.EQ.0) IOV_2 = ETA_IOV_2_1\n'
            'IF (FA1.EQ.1) IOV_2 = ETA_IOV_2_2\n'
            'ETAI1 = IOV_1 + ETA(1)\n'
            'ETAI2 = IOV_2 + ETA(2)\n',
            'CL = TVCL*EXP(ETAI1)\n' 'V = TVV*EXP(ETAI2)\n' 'S1=V\n',
            '$OMEGA BLOCK(2)\n'
            '0.00309626\t; OMEGA_IOV_1\n'
            '0.001\t; OMEGA_IOV_1_2\n'
            '0.0031128\t; OMEGA_IOV_2\n'
            '$OMEGA BLOCK(2) SAME\n',
            'joint',
        ),
        (
            'nonmem/pheno_real.mod',
            'FA1',
            None,
            None,
            'IOV_1 = 0\n'
            'IF (FA1.EQ.0) IOV_1 = ETA_IOV_1_1\n'
            'IF (FA1.EQ.1) IOV_1 = ETA_IOV_1_2\n'
            'IOV_2 = 0\n'
            'IF (FA1.EQ.0) IOV_2 = ETA_IOV_2_1\n'
            'IF (FA1.EQ.1) IOV_2 = ETA_IOV_2_2\n'
            'ETAI1 = IOV_1 + ETA(1)\n'
            'ETAI2 = IOV_2 + ETA(2)\n',
            'CL = TVCL*EXP(ETAI1)\n' 'V = TVV*EXP(ETAI2)\n' 'S1=V\n',
            '$OMEGA  BLOCK(1)\n'
            '0.00309626 ; OMEGA_IOV_1\n'
            '$OMEGA  BLOCK(1) SAME\n'
            '$OMEGA  BLOCK(1)\n'
            '0.0031128 ; OMEGA_IOV_2\n'
            '$OMEGA  BLOCK(1) SAME\n',
            'same-as-iiv',
        ),
        (
            'nonmem/pheno_real.mod',
            'FA1',
            ['ETA_2'],
            None,
            'IOV_1 = 0\n'
            'IF (FA1.EQ.0) IOV_1 = ETA_IOV_1_1\n'
            'IF (FA1.EQ.1) IOV_1 = ETA_IOV_1_2\n'
            'ETAI1 = IOV_1 + ETA(2)\n',
            'CL=TVCL*EXP(ETA(1))\n' 'V = TVV*EXP(ETAI1)\n' 'S1=V\n',
            '$OMEGA  BLOCK(1)\n' '0.0031128 ; OMEGA_IOV_1\n' '$OMEGA  BLOCK(1) SAME\n',
            'same-as-iiv',
        ),
        (
            'nonmem/pheno_real.mod',
            'FA1',
            ['CL'],
            None,
            'IOV_1 = 0\n'
            'IF (FA1.EQ.0) IOV_1 = ETA_IOV_1_1\n'
            'IF (FA1.EQ.1) IOV_1 = ETA_IOV_1_2\n'
            'ETAI1 = IOV_1 + ETA(1)\n',
            'CL = TVCL*EXP(ETAI1)\n' 'V=TVV*EXP(ETA(2))\n' 'S1=V\n',
            '$OMEGA  BLOCK(1)\n' '0.00309626 ; OMEGA_IOV_1\n' '$OMEGA  BLOCK(1) SAME\n',
            'disjoint',
        ),
        (
            'nonmem/pheno_real.mod',
            'FA1',
            ['CL', 'ETA_2'],
            None,
            'IOV_1 = 0\n'
            'IF (FA1.EQ.0) IOV_1 = ETA_IOV_1_1\n'
            'IF (FA1.EQ.1) IOV_1 = ETA_IOV_1_2\n'
            'IOV_2 = 0\n'
            'IF (FA1.EQ.0) IOV_2 = ETA_IOV_2_1\n'
            'IF (FA1.EQ.1) IOV_2 = ETA_IOV_2_2\n'
            'ETAI1 = IOV_1 + ETA(1)\n'
            'ETAI2 = IOV_2 + ETA(2)\n',
            'CL = TVCL*EXP(ETAI1)\n' 'V = TVV*EXP(ETAI2)\n' 'S1=V\n',
            '$OMEGA  BLOCK(1)\n'
            '0.00309626 ; OMEGA_IOV_1\n'
            '$OMEGA  BLOCK(1) SAME\n'
            '$OMEGA  BLOCK(1)\n'
            '0.0031128 ; OMEGA_IOV_2\n'
            '$OMEGA  BLOCK(1) SAME\n',
            'disjoint',
        ),
        (
            'nonmem/pheno_real.mod',
            'FA1',
            ['CL', 'ETA_2'],
            None,
            'IOV_1 = 0\n'
            'IF (FA1.EQ.0) IOV_1 = ETA_IOV_1_1\n'
            'IF (FA1.EQ.1) IOV_1 = ETA_IOV_1_2\n'
            'IOV_2 = 0\n'
            'IF (FA1.EQ.0) IOV_2 = ETA_IOV_2_1\n'
            'IF (FA1.EQ.1) IOV_2 = ETA_IOV_2_2\n'
            'ETAI1 = IOV_1 + ETA(1)\n'
            'ETAI2 = IOV_2 + ETA(2)\n',
            'CL = TVCL*EXP(ETAI1)\n' 'V = TVV*EXP(ETAI2)\n' 'S1=V\n',
            '$OMEGA BLOCK(2)\n'
            '0.00309626\t; OMEGA_IOV_1\n'
            '0.001\t; OMEGA_IOV_1_2\n'
            '0.0031128\t; OMEGA_IOV_2\n'
            '$OMEGA BLOCK(2) SAME\n',
            'joint',
        ),
        (
            'nonmem/pheno_real.mod',
            'FA1',
            ['ETA_1'],
            ['ETA_3', 'ETA_4'],
            'IOV_1 = 0\n'
            'IF (FA1.EQ.0) IOV_1 = ETA(3)\n'
            'IF (FA1.EQ.1) IOV_1 = ETA(4)\n'
            'ETAI1 = IOV_1 + ETA(1)\n',
            'CL = TVCL*EXP(ETAI1)\n' 'V=TVV*EXP(ETA(2))\n' 'S1=V\n',
            '$OMEGA  BLOCK(1)\n' '0.00309626 ; OMEGA_IOV_1\n' '$OMEGA  BLOCK(1) SAME\n',
            'disjoint',
        ),
        (
            'nonmem/pheno_real.mod',
            'FA1',
            ['ETA_1'],
            ['ETA_3', 'ETA_4'],
            'IOV_1 = 0\n'
            'IF (FA1.EQ.0) IOV_1 = ETA(3)\n'
            'IF (FA1.EQ.1) IOV_1 = ETA(4)\n'
            'ETAI1 = IOV_1 + ETA(1)\n',
            'CL = TVCL*EXP(ETAI1)\n' 'V=TVV*EXP(ETA(2))\n' 'S1=V\n',
            '$OMEGA  BLOCK(1)\n' '0.00309626 ; OMEGA_IOV_1\n' '$OMEGA  BLOCK(1) SAME\n',
            'joint',
        ),
        (
            'nonmem/pheno_block.mod',
            'FA1',
            ['ETA_CL'],
            None,
            'IOV_1 = 0\n'
            'IF (FA1.EQ.0) IOV_1 = ETA_IOV_1_1\n'
            'IF (FA1.EQ.1) IOV_1 = ETA_IOV_1_2\n'
            'ETAI1 = IOV_1 + ETA_CL\n',
            'CL = THETA(1)*EXP(ETAI1)\n'
            'V=THETA(2)*EXP(ETA_V)\n'
            'S1=V+ETA_S1\n'
            'MAT=THETA(3)*EXP(ETA_MAT)\n'
            'Q=THETA(4)*EXP(ETA_Q)\n',
            '$OMEGA  BLOCK(1)\n' '0.00309626 ; OMEGA_IOV_1\n' '$OMEGA  BLOCK(1) SAME\n',
            'disjoint',
        ),
        (
            'nonmem/pheno_block.mod',
            'FA1',
            None,
            None,
            'IOV_1 = 0\n'
            'IF (FA1.EQ.0) IOV_1 = ETA_IOV_1_1\n'
            'IF (FA1.EQ.1) IOV_1 = ETA_IOV_1_2\n'
            'IOV_2 = 0\n'
            'IF (FA1.EQ.0) IOV_2 = ETA_IOV_2_1\n'
            'IF (FA1.EQ.1) IOV_2 = ETA_IOV_2_2\n'
            'IOV_3 = 0\n'
            'IF (FA1.EQ.0) IOV_3 = ETA_IOV_3_1\n'
            'IF (FA1.EQ.1) IOV_3 = ETA_IOV_3_2\n'
            'IOV_4 = 0\n'
            'IF (FA1.EQ.0) IOV_4 = ETA_IOV_4_1\n'
            'IF (FA1.EQ.1) IOV_4 = ETA_IOV_4_2\n'
            'IOV_5 = 0\n'
            'IF (FA1.EQ.0) IOV_5 = ETA_IOV_5_1\n'
            'IF (FA1.EQ.1) IOV_5 = ETA_IOV_5_2\n'
            'ETAI1 = IOV_1 + ETA_CL\n'
            'ETAI2 = IOV_2 + ETA_V\n'
            'ETAI3 = IOV_3 + ETA_S1\n'
            'ETAI4 = IOV_4 + ETA_MAT\n'
            'ETAI5 = IOV_5 + ETA_Q\n',
            'CL = THETA(1)*EXP(ETAI1)\n'
            'V = THETA(2)*EXP(ETAI2)\n'
            'S1 = ETAI3 + V\n'
            'MAT = THETA(3)*EXP(ETAI4)\n'
            'Q = THETA(4)*EXP(ETAI5)\n',
            '$OMEGA  BLOCK(1)\n'
            '0.00309626 ; OMEGA_IOV_1\n'
            '$OMEGA  BLOCK(1) SAME\n'
            '$OMEGA  BLOCK(1)\n'
            '0.0031128 ; OMEGA_IOV_2\n'
            '$OMEGA  BLOCK(1) SAME\n'
            '$OMEGA  BLOCK(1)\n'
            '0.010000000000000002 ; OMEGA_IOV_3\n'
            '$OMEGA  BLOCK(1) SAME\n'
            '$OMEGA BLOCK(2)\n'
            '0.00309626\t; OMEGA_IOV_4\n'
            '5E-05\t; OMEGA_IOV_4_5\n'
            '0.0031128\t; OMEGA_IOV_5\n'
            '$OMEGA BLOCK(2) SAME\n',
            'same-as-iiv',
        ),
    ],
)
def test_add_iov(
    load_model_for_test,
    testdata,
    path,
    occ,
    etas,
    eta_names,
    pk_start_ref,
    pk_end_ref,
    omega_ref,
    distribution,
):
    model = load_model_for_test(testdata / path)
    model = add_iov(model, occ, etas, eta_names, distribution=distribution)

    model_etas = set(model.random_variables.etas.names)
    assert eta_names is None or model_etas.issuperset(eta_names)

    pk_rec = str(model.internals.control_stream.get_pred_pk_record())

    expected_pk_rec_start = f'$PK\n{pk_start_ref}'
    expected_pk_rec_end = f'{pk_end_ref}\n'

    assert pk_rec[: len(expected_pk_rec_start)] == expected_pk_rec_start
    assert pk_rec[-len(expected_pk_rec_end) :] == expected_pk_rec_end

    rec_omega = ''.join(str(rec) for rec in model.internals.control_stream.get_records('OMEGA'))

    assert rec_omega[-len(omega_ref) :] == omega_ref

    if eta_names:
        assert len(model.internals.control_stream.get_records('ABBREVIATED')) == 0
    else:
        assert len(model.internals.control_stream.get_records('ABBREVIATED')) > 0


def test_add_iov_compose(load_model_for_test, pheno_path):
    model1 = load_model_for_test(pheno_path)
    model1 = add_iov(model1, 'FA1', ['ETA_1', 'ETA_2'])

    model2 = load_model_for_test(pheno_path)
    model2 = add_iov(model2, 'FA1', 'ETA_1')
    model2 = add_iov(model2, 'FA1', 'ETA_2')

    assert set(model1.random_variables.etas.names) == set(model2.random_variables.etas.names)
    # FIXME find better way to assert models are equivalent
    assert sorted(str(model1.internals.control_stream.get_pred_pk_record()).split('\n')) == sorted(
        str(model2.internals.control_stream.get_pred_pk_record()).split('\n')
    )

    rec_omega_1 = list(str(rec) for rec in model1.internals.control_stream.get_records('OMEGA'))
    rec_omega_2 = list(str(rec) for rec in model2.internals.control_stream.get_records('OMEGA'))

    assert rec_omega_1 == rec_omega_2


def test_add_iov_only_one_level(load_model_for_test, pheno_path):
    model = load_model_for_test(pheno_path)
    df = model.dataset.copy()
    df['FA1'] = 1
    model = model.replace(dataset=df)

    with pytest.raises(ValueError, match='Only one value in FA1 column.'):
        add_iov(model, 'FA1', ['ETA_1'])


@pytest.mark.parametrize(
    'occ, params, new_eta_names, distribution, error, message',
    (
        (
            'FA1',
            ['ETA_1', 'CL'],
            None,
            'disjoint',
            ValueError,
            'ETA_1 was given twice.',
        ),
        (
            'FA1',
            ['CL', 'ETA_1'],
            None,
            'disjoint',
            ValueError,
            'ETA_1 was given twice.',
        ),
        (
            'FA1',
            [['ETA_1'], ['CL']],
            None,
            'explicit',
            ValueError,
            'ETA_1 was given twice.',
        ),
        (
            'FA1',
            [['CL'], ['ETA_1']],
            None,
            'explicit',
            ValueError,
            'ETA_1 was given twice.',
        ),
        (
            'FA1',
            ['ETA_1'],
            None,
            'abracadabra',
            ValueError,
            '"abracadabra" is not a valid value for distribution',
        ),
        (
            'FA1',
            ['ETA_1'],
            None,
            'explicit',
            ValueError,
            'distribution == "explicit" requires parameters to be given as lists of lists',
        ),
        (
            'FA1',
            [['ETA_2'], 'ETA_1'],
            None,
            'explicit',
            ValueError,
            'distribution == "explicit" requires parameters to be given as lists of lists',
        ),
        (
            'FA1',
            [['ETA_1']],
            None,
            'joint',
            ValueError,
            'distribution != "explicit" requires parameters to be given as lists of strings',
        ),
        (
            'FA1',
            [['ETA_1'], [2, 'ETA_2']],
            None,
            'explicit',
            ValueError,
            'not all parameters are string',
        ),
        (
            'FA1',
            [['ETA_1', 'ETA_2']],
            ['A', 'B', 'C', 'D', 'E'],
            'explicit',
            ValueError,
            'Number of given eta names is incorrect, need 4 names.',
        ),
    ),
)
def test_add_iov_raises(
    load_model_for_test, pheno_path, occ, params, new_eta_names, distribution, error, message
):
    model = load_model_for_test(pheno_path)
    with pytest.raises(error, match=re.escape(message)):
        add_iov(model, occ, params, eta_names=new_eta_names, distribution=distribution)


@pytest.mark.parametrize(
    'addition,expression',
    [
        (EtaAddition.exponential(add), S('CL') + sympy.exp(S('eta_new'))),
        (EtaAddition.exponential(mul), S('CL') * sympy.exp(S('eta_new'))),
    ],
)
def test_add_iiv_apply(addition, expression):
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
def test_add_iov_regression_code_record(load_model_for_test, testdata, eta_iov_1, eta_iov_2):
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


def test_add_pk_iiv_1(load_model_for_test, pheno_path):
    model = load_model_for_test(pheno_path)
    model = set_zero_order_elimination(model)
    model = add_pk_iiv(model)
    iivs = set(model.random_variables.iiv.names)
    assert iivs == {'ETA_1', 'ETA_2', 'ETA_KM'}
    model = add_peripheral_compartment(model)
    model = add_pk_iiv(model)
    iivs = set(model.random_variables.iiv.names)
    assert iivs == {'ETA_1', 'ETA_2', 'ETA_KM', 'ETA_VP1', 'ETA_QP1'}


def test_add_pk_iiv_2(load_model_for_test, pheno_path):
    model = load_model_for_test(pheno_path)
    model = set_zero_order_elimination(model)
    model = add_peripheral_compartment(model)
    model = add_pk_iiv(model)
    iivs = set(model.random_variables.iiv.names)
    assert iivs == {'ETA_1', 'ETA_2', 'ETA_KM', 'ETA_VP1', 'ETA_QP1'}


def test_add_pk_iiv_nested_params(load_model_for_test, pheno_path):
    model = load_model_for_test(pheno_path)
    model = set_transit_compartments(model, 3)
    model = add_pk_iiv(model)
    iivs = set(model.random_variables.iiv.names)
    assert iivs == {'ETA_1', 'ETA_2', 'ETA_MDT'}

    model = load_model_for_test(pheno_path)
    model = set_first_order_absorption(model)
    model = add_pk_iiv(model)
    iivs = set(model.random_variables.iiv.names)
    assert iivs == {'ETA_1', 'ETA_2', 'ETA_MAT'}

    model = load_model_for_test(pheno_path)
    model = set_transit_compartments(model, 3)
    model = add_pk_iiv(model, initial_estimate=0.01)
    assert model.parameters['IIV_MDT'].init == 0.01


@pytest.mark.parametrize(
    'etas, pk_ref, omega_ref',
    [
        (
            ['ETA_CL'],
            '$PK\n'
            'CL = THETA(1)\n'
            'V=THETA(2)*EXP(ETA_V)\n'
            'S1=V+ETA_S1\n'
            'MAT=THETA(3)*EXP(ETA_MAT)\n'
            'Q=THETA(4)*EXP(ETA_Q)\n\n',
            '$OMEGA 0.031128  ; IVV\n'
            '$OMEGA 0.1\n'
            '$OMEGA BLOCK(2)\n'
            '0.0309626\n'
            '0.0005 0.031128\n',
        ),
        (
            ['ETA_CL', 'ETA_V'],
            '$PK\n'
            'CL = THETA(1)\n'
            'V = THETA(2)\n'
            'S1=V+ETA_S1\n'
            'MAT=THETA(3)*EXP(ETA_MAT)\n'
            'Q=THETA(4)*EXP(ETA_Q)\n\n',
            '$OMEGA 0.1\n' '$OMEGA BLOCK(2)\n' '0.0309626\n' '0.0005 0.031128\n',
        ),
        (
            ['ETA_CL', 'ETA_MAT'],
            '$PK\n'
            'CL = THETA(1)\n'
            'V=THETA(2)*EXP(ETA_V)\n'
            'S1=V+ETA_S1\n'
            'MAT = THETA(3)\n'
            'Q=THETA(4)*EXP(ETA_Q)\n\n',
            '$OMEGA 0.031128  ; IVV\n' '$OMEGA 0.1\n' '$OMEGA  0.031128 ; OMEGA_5_5\n',
        ),
        (
            ['ETA_MAT', 'ETA_Q'],
            '$PK\n'
            'CL=THETA(1)*EXP(ETA_CL)\n'
            'V=THETA(2)*EXP(ETA_V)\n'
            'S1=V+ETA_S1\n'
            'MAT = THETA(3)\n'
            'Q = THETA(4)\n\n',
            '$OMEGA DIAGONAL(2)\n' '0.0309626  ; IVCL\n' '0.031128  ; IVV\n' '$OMEGA 0.1\n',
        ),
        (
            None,
            '$PK\n'
            'DUMMYETA = ETA(1)\n'
            'CL = THETA(1)\n'
            'V = THETA(2)\n'
            'S1 = V\n'
            'MAT = THETA(3)\n'
            'Q = THETA(4)\n\n',
            '$OMEGA  0 FIX ; DUMMYOMEGA\n',
        ),
        (
            ['CL'],
            '$PK\n'
            'CL = THETA(1)\n'
            'V=THETA(2)*EXP(ETA_V)\n'
            'S1=V+ETA_S1\n'
            'MAT=THETA(3)*EXP(ETA_MAT)\n'
            'Q=THETA(4)*EXP(ETA_Q)\n\n',
            '$OMEGA 0.031128  ; IVV\n'
            '$OMEGA 0.1\n'
            '$OMEGA BLOCK(2)\n'
            '0.0309626\n'
            '0.0005 0.031128\n',
        ),
        (
            'ETA_CL',
            '$PK\n'
            'CL = THETA(1)\n'
            'V=THETA(2)*EXP(ETA_V)\n'
            'S1=V+ETA_S1\n'
            'MAT=THETA(3)*EXP(ETA_MAT)\n'
            'Q=THETA(4)*EXP(ETA_Q)\n\n',
            '$OMEGA 0.031128  ; IVV\n'
            '$OMEGA 0.1\n'
            '$OMEGA BLOCK(2)\n'
            '0.0309626\n'
            '0.0005 0.031128\n',
        ),
    ],
)
def test_remove_iiv(load_model_for_test, testdata, etas, pk_ref, omega_ref):
    model = load_model_for_test(testdata / 'nonmem/pheno_block.mod')
    model = remove_iiv(model, etas)

    assert str(model.internals.control_stream.get_pred_pk_record()) == pk_ref

    rec_omega = ''.join(str(rec) for rec in model.internals.control_stream.get_records('OMEGA'))

    assert rec_omega == omega_ref


def test_remove_iov(create_model_for_test, load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem/pheno_block.mod')

    model_str = model.model_code
    model_with_iov = model_str.replace(
        '$OMEGA DIAGONAL(2)\n' '0.0309626  ; IVCL\n' '0.031128  ; IVV',
        '$OMEGA BLOCK(1)\n0.1\n$OMEGA BLOCK(1) SAME\n',
    )

    model = create_model_for_test(model_with_iov)

    model = remove_iov(model)

    assert (
        str(model.internals.control_stream.get_pred_pk_record()) == '$PK\n'
        'CL = THETA(1)\n'
        'V = THETA(2)\n'
        'S1=V+ETA_S1\n'
        'MAT=THETA(3)*EXP(ETA_MAT)\n'
        'Q=THETA(4)*EXP(ETA_Q)\n\n'
    )
    rec_omega = ''.join(str(rec) for rec in model.internals.control_stream.get_records('OMEGA'))

    assert rec_omega == '$OMEGA 0.1\n' '$OMEGA BLOCK(2)\n' '0.0309626\n' '0.0005 0.031128\n'


def test_remove_iov_no_iovs(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem/pheno_block.mod')

    with pytest.warns(UserWarning):
        remove_iov(model)


def test_remove_iov_github_issues_538_and_561_1(load_model_for_test, testdata):
    m = load_model_for_test(testdata / 'nonmem' / 'models' / 'fviii6.mod')

    m = remove_iov(m)

    assert not m.random_variables.iov


def test_remove_iov_github_issues_538_and_561_2(load_model_for_test, testdata):
    m = load_model_for_test(testdata / 'nonmem' / 'models' / 'fviii6.mod')

    m = remove_iov(m, 'ETA_4')

    assert set(m.random_variables.iov.names) == {
        'ETA_12',
        'ETA_13',
        'ETA_14',
        'ETA_15',
        'ETA_16',
        'ETA_17',
        'ETA_18',
        'ETA_19',
    }


def test_remove_iov_diagonal(create_model_for_test):
    model = create_model_for_test(
        '''$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN1 TRANS1
$PK
K=THETA(1)*EXP(ETA(1))+ETA(2)+ETA(3)+ETA(4)+ETA(5)+ETA(6)+ETA(7)
$ERROR
Y=F+F*EPS(1)
$THETA 0.1
$OMEGA DIAGONAL(2)
0.015
0.02
$OMEGA BLOCK(1)
0.6
$OMEGA BLOCK(1) SAME
$OMEGA 0.1
$OMEGA BLOCK(1)
0.01
$OMEGA BLOCK(1) SAME
$SIGMA 0.013241
$ESTIMATION METHOD=1 INTERACTION
'''
    )

    model = remove_iov(model)

    assert (
        '''$OMEGA DIAGONAL(2)
0.015
0.02
$OMEGA 0.1'''
        in model.model_code
    )


@pytest.mark.parametrize(
    ('distribution', 'occ', 'to_remove', 'cases', 'rest', 'abbr_ref'),
    (
        (
            'disjoint',
            'VISI',
            None,
            'IOV_1 = 0\n'
            'IF (VISI.EQ.3.OR.VISI.EQ.8) IOV_1 = 0\n'
            'IOV_2 = 0\n'
            'IF (VISI.EQ.3.OR.VISI.EQ.8) IOV_2 = 0\n'
            'IOV_3 = 0\n'
            'IF (VISI.EQ.3.OR.VISI.EQ.8) IOV_3 = 0\n',
            (),
            '',
        ),
        (
            'joint',
            'VISI',
            None,
            'IOV_1 = 0\n'
            'IF (VISI.EQ.3.OR.VISI.EQ.8) IOV_1 = 0\n'
            'IOV_2 = 0\n'
            'IF (VISI.EQ.3.OR.VISI.EQ.8) IOV_2 = 0\n'
            'IOV_3 = 0\n'
            'IF (VISI.EQ.3.OR.VISI.EQ.8) IOV_3 = 0\n',
            (),
            '',
        ),
        (
            'disjoint',
            'VISI',
            'ETA_IOV_1_1',
            'IOV_1 = 0\n'
            'IF (VISI.EQ.3.OR.VISI.EQ.8) IOV_1 = 0\n'
            'IOV_2 = 0\n'
            'IF (VISI.EQ.3) IOV_2 = ETA_IOV_2_1\n'
            'IF (VISI.EQ.8) IOV_2 = ETA_IOV_2_2\n'
            'IOV_3 = 0\n'
            'IF (VISI.EQ.3) IOV_3 = ETA_IOV_3_1\n'
            'IF (VISI.EQ.8) IOV_3 = ETA_IOV_3_2\n',
            ('ETA_IOV_2_1', 'ETA_IOV_2_2', 'ETA_IOV_3_1', 'ETA_IOV_3_2'),
            '$ABBR REPLACE ETA_IOV_2_1=ETA(4)\n'
            '$ABBR REPLACE ETA_IOV_2_2=ETA(5)\n'
            '$ABBR REPLACE ETA_IOV_3_1=ETA(6)\n'
            '$ABBR REPLACE ETA_IOV_3_2=ETA(7)\n',
        ),
        (
            'joint',
            'VISI',
            'ETA_IOV_1_1',
            'IOV_1 = 0\n'
            'IF (VISI.EQ.3.OR.VISI.EQ.8) IOV_1 = 0\n'
            'IOV_2 = 0\n'
            'IF (VISI.EQ.3) IOV_2 = ETA_IOV_2_1\n'
            'IF (VISI.EQ.8) IOV_2 = ETA_IOV_2_2\n'
            'IOV_3 = 0\n'
            'IF (VISI.EQ.3) IOV_3 = ETA_IOV_3_1\n'
            'IF (VISI.EQ.8) IOV_3 = ETA_IOV_3_2\n',
            ('ETA_IOV_2_1', 'ETA_IOV_2_2', 'ETA_IOV_3_1', 'ETA_IOV_3_2'),
            '$ABBR REPLACE ETA_IOV_2_1=ETA(4)\n'
            '$ABBR REPLACE ETA_IOV_3_1=ETA(5)\n'
            '$ABBR REPLACE ETA_IOV_2_2=ETA(6)\n'
            '$ABBR REPLACE ETA_IOV_3_2=ETA(7)\n',
        ),
        (
            'disjoint',
            'VISI',
            ['ETA_IOV_1_1', 'ETA_IOV_1_2'],
            'IOV_1 = 0\n'
            'IF (VISI.EQ.3.OR.VISI.EQ.8) IOV_1 = 0\n'
            'IOV_2 = 0\n'
            'IF (VISI.EQ.3) IOV_2 = ETA_IOV_2_1\n'
            'IF (VISI.EQ.8) IOV_2 = ETA_IOV_2_2\n'
            'IOV_3 = 0\n'
            'IF (VISI.EQ.3) IOV_3 = ETA_IOV_3_1\n'
            'IF (VISI.EQ.8) IOV_3 = ETA_IOV_3_2\n',
            ('ETA_IOV_2_1', 'ETA_IOV_2_2', 'ETA_IOV_3_1', 'ETA_IOV_3_2'),
            '$ABBR REPLACE ETA_IOV_2_1=ETA(4)\n'
            '$ABBR REPLACE ETA_IOV_2_2=ETA(5)\n'
            '$ABBR REPLACE ETA_IOV_3_1=ETA(6)\n'
            '$ABBR REPLACE ETA_IOV_3_2=ETA(7)\n',
        ),
        (
            'joint',
            'VISI',
            ['ETA_IOV_1_1', 'ETA_IOV_1_2'],
            'IOV_1 = 0\n'
            'IF (VISI.EQ.3.OR.VISI.EQ.8) IOV_1 = 0\n'
            'IOV_2 = 0\n'
            'IF (VISI.EQ.3) IOV_2 = ETA_IOV_2_1\n'
            'IF (VISI.EQ.8) IOV_2 = ETA_IOV_2_2\n'
            'IOV_3 = 0\n'
            'IF (VISI.EQ.3) IOV_3 = ETA_IOV_3_1\n'
            'IF (VISI.EQ.8) IOV_3 = ETA_IOV_3_2\n',
            ('ETA_IOV_2_1', 'ETA_IOV_2_2', 'ETA_IOV_3_1', 'ETA_IOV_3_2'),
            '$ABBR REPLACE ETA_IOV_2_1=ETA(4)\n'
            '$ABBR REPLACE ETA_IOV_3_1=ETA(5)\n'
            '$ABBR REPLACE ETA_IOV_2_2=ETA(6)\n'
            '$ABBR REPLACE ETA_IOV_3_2=ETA(7)\n',
        ),
        (
            'disjoint',
            'VISI',
            ['ETA_IOV_1_1', 'ETA_IOV_1_2', 'ETA_IOV_2_1'],
            'IOV_1 = 0\n'
            'IF (VISI.EQ.3.OR.VISI.EQ.8) IOV_1 = 0\n'
            'IOV_2 = 0\n'
            'IF (VISI.EQ.3.OR.VISI.EQ.8) IOV_2 = 0\n'
            'IOV_3 = 0\n'
            'IF (VISI.EQ.3) IOV_3 = ETA_IOV_3_1\n'
            'IF (VISI.EQ.8) IOV_3 = ETA_IOV_3_2\n',
            ('ETA_IOV_3_1', 'ETA_IOV_3_2'),
            '$ABBR REPLACE ETA_IOV_3_1=ETA(4)\n' '$ABBR REPLACE ETA_IOV_3_2=ETA(5)\n',
        ),
        (
            'joint',
            'VISI',
            ['ETA_IOV_1_1', 'ETA_IOV_1_2', 'ETA_IOV_2_1'],
            'IOV_1 = 0\n'
            'IF (VISI.EQ.3.OR.VISI.EQ.8) IOV_1 = 0\n'
            'IOV_2 = 0\n'
            'IF (VISI.EQ.3.OR.VISI.EQ.8) IOV_2 = 0\n'
            'IOV_3 = 0\n'
            'IF (VISI.EQ.3) IOV_3 = ETA_IOV_3_1\n'
            'IF (VISI.EQ.8) IOV_3 = ETA_IOV_3_2\n',
            ('ETA_IOV_3_1', 'ETA_IOV_3_2'),
            '$ABBR REPLACE ETA_IOV_3_1=ETA(4)\n' '$ABBR REPLACE ETA_IOV_3_2=ETA(5)\n',
        ),
    ),
    ids=repr,
)
def test_remove_iov_with_options(
    tmp_path, load_model_for_test, testdata, distribution, occ, to_remove, cases, rest, abbr_ref
):
    with chdir(tmp_path):
        shutil.copy2(testdata / 'nonmem' / 'models' / 'mox2.mod', tmp_path)
        shutil.copy2(testdata / 'nonmem' / 'models' / 'mox_simulated_normal.csv', tmp_path)
        model = load_model_for_test('mox2.mod')
        model = model.replace(
            datainfo=model.datainfo.replace(path=tmp_path / 'mox_simulated_normal.csv')
        )

        start_model = add_iov(model, occ=occ, distribution=distribution)

        model_with_some_iovs_removed = remove_iov(start_model, to_remove=to_remove)

        assert cases in model_with_some_iovs_removed.model_code
        assert set(model_with_some_iovs_removed.random_variables.iov.names) == set(rest)

        rec_abbr = ''.join(
            str(rec)
            for rec in model_with_some_iovs_removed.internals.control_stream.get_records(
                'ABBREVIATED'
            )
        )
        assert rec_abbr == abbr_ref


@pytest.mark.parametrize(
    'etas, etab, buf_new',
    [
        (
            ['ETA_1'],
            'ETAB1 = (EXP(ETA(1))**THETA(4) - 1)/THETA(4)',
            'CL = TVCL*EXP(ETAB1)\nV=TVV*EXP(ETA(2))',
        ),
        (
            ['ETA_1', 'ETA_2'],
            'ETAB1 = (EXP(ETA(1))**THETA(4) - 1)/THETA(4)\n'
            'ETAB2 = (EXP(ETA(2))**THETA(5) - 1)/THETA(5)',
            'CL = TVCL*EXP(ETAB1)\nV = TVV*EXP(ETAB2)',
        ),
        (
            None,
            'ETAB1 = (EXP(ETA(1))**THETA(4) - 1)/THETA(4)\n'
            'ETAB2 = (EXP(ETA(2))**THETA(5) - 1)/THETA(5)',
            'CL = TVCL*EXP(ETAB1)\nV = TVV*EXP(ETAB2)',
        ),
        (
            'ETA_1',
            'ETAB1 = (EXP(ETA(1))**THETA(4) - 1)/THETA(4)',
            'CL = TVCL*EXP(ETAB1)\nV=TVV*EXP(ETA(2))',
        ),
    ],
)
def test_transform_etas_boxcox(load_model_for_test, pheno_path, etas, etab, buf_new):
    model = load_model_for_test(pheno_path)

    model = transform_etas_boxcox(model, etas)

    rec_ref = (
        f'$PK\n'
        f'{etab}\n'
        f'IF(AMT.GT.0) BTIME=TIME\n'
        f'TAD=TIME-BTIME\n'
        f'TVCL=THETA(1)*WGT\n'
        f'TVV=THETA(2)*WGT\n'
        f'IF(APGR.LT.5) TVV=TVV*(1+THETA(3))\n'
        f'{buf_new}\n'
        f'S1=V\n\n'
    )

    assert str(model.internals.control_stream.get_pred_pk_record()) == rec_ref
    assert model.parameters['lambda1'].init == 0.01


def test_transform_etas_tdist(load_model_for_test, pheno_path):
    model = load_model_for_test(pheno_path)

    model = transform_etas_tdist(model, ['ETA_1'])

    symbol = 'ETAT1'

    eta = 'ETA(1)'
    theta = 'THETA(4)'

    num_1 = f'{eta}**2 + 1'
    denom_1 = f'4*{theta}'

    num_2 = f'5*{eta}**4 + 16*{eta}**2 + 3'
    denom_2 = f'96*{theta}**2'

    num_3 = f'3*{eta}**6 + 17*{eta}**2 + 19*{eta}**4 - 15'
    denom_3 = f'384*{theta}**3'

    expression = (
        f'ETA(1)*(({num_1})/({denom_1}) + ({num_2})/({denom_2}) + ({num_3})/({denom_3}) + 1)'
    )

    rec_ref = (
        f'$PK\n'
        f'{symbol} = {expression}\n'
        f'IF(AMT.GT.0) BTIME=TIME\n'
        f'TAD=TIME-BTIME\n'
        f'TVCL=THETA(1)*WGT\n'
        f'TVV=THETA(2)*WGT\n'
        f'IF(APGR.LT.5) TVV=TVV*(1+THETA(3))\n'
        f'CL = TVCL*EXP(ETAT1)\n'
        f'V=TVV*EXP(ETA(2))\n'
        f'S1=V\n\n'
    )

    assert str(model.internals.control_stream.get_pred_pk_record()) == rec_ref
    assert model.parameters['df1'].init == 80


@pytest.mark.parametrize(
    'etas, etad, buf_new',
    [
        (
            ['ETA_1'],
            'ETAD1 = ((ABS(ETA(1)) + 1)**THETA(4) - 1)*ABS(ETA(1))/(ETA(1)*THETA(4))',
            'CL = TVCL*EXP(ETAD1)\nV=TVV*EXP(ETA(2))',
        ),
        (
            'ETA_1',
            'ETAD1 = ((ABS(ETA(1)) + 1)**THETA(4) - 1)*ABS(ETA(1))/(ETA(1)*THETA(4))',
            'CL = TVCL*EXP(ETAD1)\nV=TVV*EXP(ETA(2))',
        ),
    ],
)
def test_transform_etas_john_draper(load_model_for_test, pheno_path, etas, etad, buf_new):
    model = load_model_for_test(pheno_path)

    model = transform_etas_john_draper(model, etas)

    rec_ref = (
        f'$PK\n'
        f'{etad}\n'
        f'IF(AMT.GT.0) BTIME=TIME\n'
        f'TAD=TIME-BTIME\n'
        f'TVCL=THETA(1)*WGT\n'
        f'TVV=THETA(2)*WGT\n'
        f'IF(APGR.LT.5) TVV=TVV*(1+THETA(3))\n'
        f'{buf_new}\n'
        f'S1=V\n\n'
    )

    assert str(model.internals.control_stream.get_pred_pk_record()) == rec_ref


@pytest.mark.parametrize(
    'eta_trans,symbol,expression',
    [
        (
            EtaTransformation.boxcox(2),
            S('ETAB(2)'),
            ((sympy.exp(S('ETA(2)')) ** S('BOXCOX2') - 1) / S('BOXCOX2')),
        ),
    ],
)
def test_transform_etas_apply(eta_trans, symbol, expression):
    etas = {'eta1': S('ETA(1)'), 'eta2': S('ETA(2)'), 'etab1': S('ETAB(1)'), 'etab2': S('ETAB(2)')}

    thetas = {'theta1': 'BOXCOX1', 'theta2': 'BOXCOX2'}

    eta_trans.apply(etas, thetas)
    assert eta_trans.assignments[1].symbol == symbol
    assert eta_trans.assignments[1].expression == expression


@pytest.mark.parametrize(
    'etas, abbr_ref, omega_ref',
    [
        (
            ['ETA_CL', 'ETA_V'],
            '$ABBR REPLACE ETA_CL=ETA(1)\n'
            '$ABBR REPLACE ETA_V=ETA(2)\n'
            '$ABBR REPLACE ETA_S1=ETA(3)\n'
            '$ABBR REPLACE ETA_MAT=ETA(4)\n'
            '$ABBR REPLACE ETA_Q=ETA(5)\n',
            '$OMEGA BLOCK(2)\n'
            '0.0309626\t; IVCL\n'
            '0.0031045\t; IIV_CL_IIV_V\n'
            '0.031128\t; IVV\n'
            '$OMEGA 0.1\n'
            '$OMEGA BLOCK(2)\n'
            '0.0309626\n'
            '0.0005 0.031128\n',
        ),
        (
            ['ETA_CL', 'ETA_S1'],
            '$ABBR REPLACE ETA_CL=ETA(1)\n'
            '$ABBR REPLACE ETA_S1=ETA(2)\n'
            '$ABBR REPLACE ETA_V=ETA(3)\n'
            '$ABBR REPLACE ETA_MAT=ETA(4)\n'
            '$ABBR REPLACE ETA_Q=ETA(5)\n',
            '$OMEGA BLOCK(2)\n'
            '0.0309626\t; IVCL\n'
            '0.0055644\t; IIV_CL_IIV_S1\n'
            '0.1\t; OMEGA_3_3\n'
            '$OMEGA 0.031128  ; IVV\n'
            '$OMEGA BLOCK(2)\n'
            '0.0309626\n'
            '0.0005 0.031128\n',
        ),
        (
            ['ETA_V', 'ETA_S1'],
            '$ABBR REPLACE ETA_CL=ETA(1)\n'
            '$ABBR REPLACE ETA_V=ETA(2)\n'
            '$ABBR REPLACE ETA_S1=ETA(3)\n'
            '$ABBR REPLACE ETA_MAT=ETA(4)\n'
            '$ABBR REPLACE ETA_Q=ETA(5)\n',
            '$OMEGA 0.0309626  ; IVCL\n'
            '$OMEGA BLOCK(2)\n'
            '0.031128\t; IVV\n'
            '0.0055792\t; IIV_V_IIV_S1\n'
            '0.1\n'
            '$OMEGA BLOCK(2)\n'
            '0.0309626\n'
            '0.0005 0.031128\n',
        ),
        (
            ['ETA_CL', 'ETA_V', 'ETA_MAT'],
            '$ABBR REPLACE ETA_CL=ETA(1)\n'
            '$ABBR REPLACE ETA_V=ETA(2)\n'
            '$ABBR REPLACE ETA_MAT=ETA(3)\n'
            '$ABBR REPLACE ETA_S1=ETA(4)\n'
            '$ABBR REPLACE ETA_Q=ETA(5)\n',
            '$OMEGA BLOCK(3)\n'
            '0.0309626\t; IVCL\n'
            '0.0031045\t; IIV_CL_IIV_V\n'
            '0.031128\t; IVV\n'
            '0.0030963\t; IIV_CL_IIV_MAT\n'
            '0.0031045\t; IIV_V_IIV_MAT\n'
            '0.0309626\t; OMEGA_4_4\n'
            '$OMEGA 0.1\n'
            '$OMEGA  0.031128\n',
        ),
        (
            ['ETA_V', 'ETA_S1', 'ETA_MAT'],
            '$ABBR REPLACE ETA_CL=ETA(1)\n'
            '$ABBR REPLACE ETA_V=ETA(2)\n'
            '$ABBR REPLACE ETA_S1=ETA(3)\n'
            '$ABBR REPLACE ETA_MAT=ETA(4)\n'
            '$ABBR REPLACE ETA_Q=ETA(5)\n',
            '$OMEGA 0.0309626  ; IVCL\n'
            '$OMEGA BLOCK(3)\n'
            '0.031128\t; IVV\n'
            '0.0055792\t; IIV_V_IIV_S1\n'
            '0.1\n'
            '0.0031045\t; IIV_V_IIV_MAT\n'
            '0.0055644\t; IIV_S1_IIV_MAT\n'
            '0.0309626\n'
            '$OMEGA  0.031128\n',
        ),
        (
            ['ETA_S1', 'ETA_MAT', 'ETA_Q'],
            '$ABBR REPLACE ETA_CL=ETA(1)\n'
            '$ABBR REPLACE ETA_V=ETA(2)\n'
            '$ABBR REPLACE ETA_S1=ETA(3)\n'
            '$ABBR REPLACE ETA_MAT=ETA(4)\n'
            '$ABBR REPLACE ETA_Q=ETA(5)\n',
            '$OMEGA DIAGONAL(2)\n'
            '0.0309626  ; IVCL\n'
            '0.031128  ; IVV\n'
            '$OMEGA BLOCK(3)\n'
            '0.1\n'
            '0.0055644\t; IIV_S1_IIV_MAT\n'
            '0.0309626\n'
            '0.0055792\t; IIV_S1_IIV_Q\n'
            '0.0005\n'
            '0.031128\n',
        ),
        (
            None,
            '$ABBR REPLACE ETA_CL=ETA(1)\n'
            '$ABBR REPLACE ETA_V=ETA(2)\n'
            '$ABBR REPLACE ETA_S1=ETA(3)\n'
            '$ABBR REPLACE ETA_MAT=ETA(4)\n'
            '$ABBR REPLACE ETA_Q=ETA(5)\n',
            '$OMEGA BLOCK(5)\n'
            '0.0309626\t; IVCL\n'
            '0.0031045\t; IIV_CL_IIV_V\n'
            '0.031128\t; IVV\n'
            '0.0055644\t; IIV_CL_IIV_S1\n'
            '0.0055792\t; IIV_V_IIV_S1\n'
            '0.1\n'
            '0.0030963\t; IIV_CL_IIV_MAT\n'
            '0.0031045\t; IIV_V_IIV_MAT\n'
            '0.0055644\t; IIV_S1_IIV_MAT\n'
            '0.0309626\n'
            '0.0031045\t; IIV_CL_IIV_Q\n'
            '0.0031128\t; IIV_V_IIV_Q\n'
            '0.0055792\t; IIV_S1_IIV_Q\n'
            '0.0005\n'
            '0.031128\n',
        ),
    ],
)
def test_create_joint_distribution_plain(load_model_for_test, testdata, etas, abbr_ref, omega_ref):
    model_start = load_model_for_test(testdata / 'nonmem/pheno_block.mod')

    model = create_joint_distribution(model_start, etas, individual_estimates=None)

    rec_abbr = ''.join(
        str(rec) for rec in model.internals.control_stream.get_records('ABBREVIATED')
    )
    assert rec_abbr == abbr_ref

    rec_pk = str(model.internals.control_stream.get_pred_pk_record())
    pk_ref = str(model_start.internals.control_stream.get_pred_pk_record())
    assert rec_pk == pk_ref

    rec_omega = ''.join(str(rec) for rec in model.internals.control_stream.get_records('OMEGA'))
    assert rec_omega == omega_ref


@pytest.mark.parametrize(
    'etas, abbr_ref, omega_ref',
    [
        (
            (['ETA_CL', 'ETA_V'], ['ETA_CL', 'ETA_S1']),
            '$ABBR REPLACE ETA_CL=ETA(1)\n'
            '$ABBR REPLACE ETA_S1=ETA(2)\n'
            '$ABBR REPLACE ETA_V=ETA(3)\n'
            '$ABBR REPLACE ETA_MAT=ETA(4)\n'
            '$ABBR REPLACE ETA_Q=ETA(5)\n',
            '$OMEGA BLOCK(2)\n'
            '0.0309626\t; IVCL\n'
            '0.0055644\t; IIV_CL_IIV_S1\n'
            '0.1\t; OMEGA_3_3\n'
            '$OMEGA  0.031128 ; IVV\n'
            '$OMEGA BLOCK(2)\n'
            '0.0309626\n'
            '0.0005 0.031128\n',
        ),
        (
            (None, ['ETA_CL', 'ETA_V']),
            '$ABBR REPLACE ETA_CL=ETA(1)\n'
            '$ABBR REPLACE ETA_V=ETA(2)\n'
            '$ABBR REPLACE ETA_S1=ETA(3)\n'
            '$ABBR REPLACE ETA_MAT=ETA(4)\n'
            '$ABBR REPLACE ETA_Q=ETA(5)\n',
            '$OMEGA BLOCK(2)\n'
            '0.0309626\t; IVCL\n'
            '0.0031045\t; IIV_CL_IIV_V\n'
            '0.031128\t; IVV\n'
            '$OMEGA BLOCK(3)\n'
            '0.1\n'
            '0.0055644\t; IIV_S1_IIV_MAT\n'
            '0.0309626\n'
            '0.0055792\t; IIV_S1_IIV_Q\n'
            '0.0005\n'
            '0.031128\n',
        ),
        (
            (['ETA_CL', 'ETA_V'], None),
            '$ABBR REPLACE ETA_CL=ETA(1)\n'
            '$ABBR REPLACE ETA_V=ETA(2)\n'
            '$ABBR REPLACE ETA_S1=ETA(3)\n'
            '$ABBR REPLACE ETA_MAT=ETA(4)\n'
            '$ABBR REPLACE ETA_Q=ETA(5)\n',
            '$OMEGA BLOCK(5)\n'
            '0.0309626\t; IVCL\n'
            '0.0031045\t; IIV_CL_IIV_V\n'
            '0.031128\t; IVV\n'
            '0.0055644\t; IIV_CL_IIV_S1\n'
            '0.0055792\t; IIV_V_IIV_S1\n'
            '0.1\n'
            '0.0030963\t; IIV_CL_IIV_MAT\n'
            '0.0031045\t; IIV_V_IIV_MAT\n'
            '0.0055644\t; IIV_S1_IIV_MAT\n'
            '0.0309626\n'
            '0.0031045\t; IIV_CL_IIV_Q\n'
            '0.0031128\t; IIV_V_IIV_Q\n'
            '0.0055792\t; IIV_S1_IIV_Q\n'
            '0.0005\n'
            '0.031128\n',
        ),
    ],
)
def test_create_joint_distribution_nested(load_model_for_test, testdata, etas, abbr_ref, omega_ref):
    model_start = load_model_for_test(testdata / 'nonmem/pheno_block.mod')

    model = create_joint_distribution(model_start, etas[0], individual_estimates=None)
    model = create_joint_distribution(model, etas[1], individual_estimates=None)

    rec_abbr = ''.join(
        str(rec) for rec in model.internals.control_stream.get_records('ABBREVIATED')
    )
    assert rec_abbr == abbr_ref

    rec_pk = str(model.internals.control_stream.get_pred_pk_record())
    pk_ref = str(model_start.internals.control_stream.get_pred_pk_record())
    assert rec_pk == pk_ref

    rec_omega = ''.join(str(rec) for rec in model.internals.control_stream.get_records('OMEGA'))

    assert rec_omega == omega_ref


@pytest.mark.parametrize(
    'etas, abbr_ref, omega_ref',
    [
        (
            ['ETA_CL'],
            '$ABBR REPLACE ETA_CL=ETA(1)\n'
            '$ABBR REPLACE ETA_V=ETA(2)\n'
            '$ABBR REPLACE ETA_S1=ETA(3)\n'
            '$ABBR REPLACE ETA_MAT=ETA(4)\n'
            '$ABBR REPLACE ETA_Q=ETA(5)\n',
            '$OMEGA  0.0309626 ; IVCL\n'
            '$OMEGA BLOCK(4)\n'
            '0.031128\t; IVV\n'
            '0.0055792\t; IIV_V_IIV_S1\n'
            '0.1\n'
            '0.0031045\t; IIV_V_IIV_MAT\n'
            '0.0055644\t; IIV_S1_IIV_MAT\n'
            '0.0309626\n'
            '0.0031128\t; IIV_V_IIV_Q\n'
            '0.0055792\t; IIV_S1_IIV_Q\n'
            '0.0005\n'
            '0.031128\n',
        ),
        (
            ['ETA_CL', 'ETA_V'],
            '$ABBR REPLACE ETA_CL=ETA(1)\n'
            '$ABBR REPLACE ETA_V=ETA(2)\n'
            '$ABBR REPLACE ETA_S1=ETA(3)\n'
            '$ABBR REPLACE ETA_MAT=ETA(4)\n'
            '$ABBR REPLACE ETA_Q=ETA(5)\n',
            '$OMEGA  0.0309626 ; IVCL\n'
            '$OMEGA  0.031128 ; IVV\n'
            '$OMEGA BLOCK(3)\n'
            '0.1\n'
            '0.0055644\t; IIV_S1_IIV_MAT\n'
            '0.0309626\n'
            '0.0055792\t; IIV_S1_IIV_Q\n'
            '0.0005\n'
            '0.031128\n',
        ),
        (
            ['ETA_CL', 'ETA_S1'],
            '$ABBR REPLACE ETA_CL=ETA(1)\n'
            '$ABBR REPLACE ETA_S1=ETA(2)\n'
            '$ABBR REPLACE ETA_V=ETA(3)\n'
            '$ABBR REPLACE ETA_MAT=ETA(4)\n'
            '$ABBR REPLACE ETA_Q=ETA(5)\n',
            '$OMEGA  0.0309626 ; IVCL\n'
            '$OMEGA  0.1 ; OMEGA_3_3\n'
            '$OMEGA BLOCK(3)\n'
            '0.031128\t; IVV\n'
            '0.0031045\t; IIV_V_IIV_MAT\n'
            '0.0309626\n'
            '0.0031128\t; IIV_V_IIV_Q\n'
            '0.0005\n'
            '0.031128\n',
        ),
        (
            None,
            '$ABBR REPLACE ETA_CL=ETA(1)\n'
            '$ABBR REPLACE ETA_V=ETA(2)\n'
            '$ABBR REPLACE ETA_S1=ETA(3)\n'
            '$ABBR REPLACE ETA_MAT=ETA(4)\n'
            '$ABBR REPLACE ETA_Q=ETA(5)\n',
            '$OMEGA  0.0309626 ; IVCL\n'
            '$OMEGA  0.031128 ; IVV\n'
            '$OMEGA  0.1\n'
            '$OMEGA  0.0309626\n'
            '$OMEGA  0.031128\n',
        ),
        (
            'ETA_CL',
            '$ABBR REPLACE ETA_CL=ETA(1)\n'
            '$ABBR REPLACE ETA_V=ETA(2)\n'
            '$ABBR REPLACE ETA_S1=ETA(3)\n'
            '$ABBR REPLACE ETA_MAT=ETA(4)\n'
            '$ABBR REPLACE ETA_Q=ETA(5)\n',
            '$OMEGA  0.0309626 ; IVCL\n'
            '$OMEGA BLOCK(4)\n'
            '0.031128\t; IVV\n'
            '0.0055792\t; IIV_V_IIV_S1\n'
            '0.1\n'
            '0.0031045\t; IIV_V_IIV_MAT\n'
            '0.0055644\t; IIV_S1_IIV_MAT\n'
            '0.0309626\n'
            '0.0031128\t; IIV_V_IIV_Q\n'
            '0.0055792\t; IIV_S1_IIV_Q\n'
            '0.0005\n'
            '0.031128\n',
        ),
    ],
)
def test_split_joint_distribution(load_model_for_test, testdata, etas, abbr_ref, omega_ref):
    model_start = load_model_for_test(testdata / 'nonmem/pheno_block.mod')
    model = create_joint_distribution(model_start)

    model = split_joint_distribution(model, etas)

    rec_abbr = ''.join(
        str(rec) for rec in model.internals.control_stream.get_records('ABBREVIATED')
    )
    assert rec_abbr == abbr_ref

    rec_pk = str(model.internals.control_stream.get_pred_pk_record())
    pk_ref = str(model_start.internals.control_stream.get_pred_pk_record())
    assert rec_pk == pk_ref

    rec_omega = ''.join(str(rec) for rec in model.internals.control_stream.get_records('OMEGA'))

    assert rec_omega == omega_ref


@pytest.mark.parametrize(
    'rvs, exception_msg',
    [
        (['ETA_3', 'NON_EXISTENT_RV'], r'.*non-existing.*'),
        (['ETA_3', 'ETA_6'], r'.*ETA_6.*'),
        (['ETA_1'], 'At least two random variables are needed'),
    ],
)
def test_create_joint_distribution_incorrect_params(
    load_model_for_test, testdata, rvs, exception_msg
):
    model = load_model_for_test(
        testdata / 'nonmem' / 'modelfit_results' / 'onePROB' / 'multEST' / 'noSIM' / 'withBayes.mod'
    )

    with pytest.raises(Exception, match=exception_msg):
        create_joint_distribution(model, rvs, individual_estimates=None)


def test_create_joint_distribution_choose_param_init(load_model_for_test, pheno_path):
    model = load_model_for_test(pheno_path)
    res = read_modelfit_results(pheno_path)
    params = (model.parameters['IVCL'], model.parameters['IVV'])
    rvs = model.random_variables.etas
    init = _choose_cov_param_init(model, res.individual_estimates, rvs, *params)
    assert init == 0.0118179

    model = load_model_for_test(pheno_path)
    init = _choose_cov_param_init(model, None, rvs, *params)
    assert init == 0.0031045

    model = load_model_for_test(pheno_path)
    rv_new = NormalDistribution.create('ETA_3', 'IIV', 0, S('OMEGA_3_3'))
    rvs += rv_new
    ie = res.individual_estimates.copy()
    ie['ETA_3'] = ie['ETA_1']
    init = _choose_cov_param_init(model, res.individual_estimates, rvs, *params)
    assert init == 0.0118179

    # If one eta doesn't have individual estimates
    model = load_model_for_test(pheno_path)
    model = add_iiv(model, 'S1', 'add')
    params = (model.parameters['IVCL'], model.parameters['IIV_S1'])
    rvs = model.random_variables[('ETA_1', 'ETA_S1')]
    init = _choose_cov_param_init(model, res.individual_estimates, rvs, *params)
    assert init == 0.0052789

    # If the standard deviation in individual estimates of one eta is 0
    model = load_model_for_test(pheno_path)
    ie = res.individual_estimates.copy()
    ie['ETA_1'] = 0
    params = (model.parameters['IVCL'], model.parameters['IVV'])
    rvs = model.random_variables[('ETA_1', 'ETA_2')]
    with pytest.warns(UserWarning, match='Correlation of individual estimates'):
        init = _choose_cov_param_init(model, ie, rvs, *params)
        assert init == 0.0031045


def test_create_joint_distribution_choose_param_init_fo(create_model_for_test):
    model = create_model_for_test(
        '''$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV
$SUBROUTINE ADVAN1 TRANS2

$PK
CL=THETA(1)*EXP(ETA(1))
V=THETA(2)*EXP(ETA(1))
S1=V*EXP(ETA(2))

$ERROR
Y=F+F*EPS(1)

$THETA (0,0.00469307) ; TVCL
$THETA (0,1.00916) ; TVV
$OMEGA 0.1  ; IVCL
$OMEGA 0.1  ; IVV
$SIGMA 0.013241

$ESTIMATION METHOD=0
'''
    )
    params = (model.parameters['IVCL'], model.parameters['IVV'])
    rvs = model.random_variables.etas
    init = _choose_cov_param_init(model, None, rvs, *params)

    assert init == 0.01


def test_create_joint_distribution_names(create_model_for_test):
    model = create_model_for_test(
        '''$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV
$SUBROUTINE ADVAN1 TRANS2

$PK
CL=THETA(1)*EXP(ETA(1))
V=THETA(2)*EXP(ETA(1))
S1=V*EXP(ETA(2))

$ERROR
Y=F+F*EPS(1)

$THETA (0,0.00469307) ; TVCL
$THETA (0,1.00916) ; TVV
$OMEGA 0.0309626  ; IVCL
$OMEGA 0.031128  ; IVV
$SIGMA 0.013241

$ESTIMATION METHOD=1 INTERACTION
'''
    )
    model = create_joint_distribution(
        model,
        model.random_variables.names,
        individual_estimates=model.modelfit_results.individual_estimates
        if model.modelfit_results is not None
        else None,
    )
    assert 'IIV_CL_V_IIV_S1' in model.parameters.names


@pytest.mark.parametrize(
    'etas_file, force, file_exists',
    [('', False, False), ('', True, True), ('$ETAS FILE=run1.phi', False, True)],
)
def test_update_initial_individual_estimates(
    load_model_for_test, testdata, etas_file, force, file_exists, tmp_path
):
    shutil.copy(testdata / 'nonmem/pheno.mod', tmp_path / 'run1.mod')
    shutil.copy(testdata / 'nonmem/pheno.phi', tmp_path / 'run1.phi')
    shutil.copy(testdata / 'nonmem/pheno.ext', tmp_path / 'run1.ext')
    shutil.copy(testdata / 'nonmem/pheno.dta', tmp_path / 'pheno.dta')

    with chdir(tmp_path):
        with open('run1.mod', 'a') as f:
            f.write(etas_file)

        model = load_model_for_test('run1.mod')
        res = read_modelfit_results('run1.mod')
        model = update_initial_individual_estimates(model, res.individual_estimates, force=force)
        model = model.write_files()

        assert ('$ETAS FILE=run1_input.phi' in model.model_code) is file_exists
        assert (os.path.isfile('run1_input.phi')) is file_exists


def test_nested_iiv_transformations(load_model_for_test, pheno_path):
    model = load_model_for_test(pheno_path)
    res = read_modelfit_results(pheno_path)

    model = create_joint_distribution(model, individual_estimates=res.individual_estimates)

    assert 'IIV_CL_IIV_V' in model.model_code

    model = load_model_for_test(pheno_path)

    model = remove_iiv(model, 'CL')

    assert '0.031128' in model.model_code
    assert '0.0309626' not in model.model_code

    model = load_model_for_test(pheno_path)

    model = remove_iiv(model, 'V')

    assert '0.0309626' in model.model_code
    assert '0.031128' not in model.model_code
