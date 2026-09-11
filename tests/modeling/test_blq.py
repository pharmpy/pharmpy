import pytest

from pharmpy.basic import Expr
from pharmpy.deps import numpy as np
from pharmpy.deps import pandas as pd
from pharmpy.model.model import update_datainfo
from pharmpy.modeling import (
    create_datainfo,
    create_joint_distribution,
    remove_error_model,
    set_additive_error_model,
    set_combined_error_model,
    set_iiv_on_ruv,
    set_power_on_ruv,
    set_proportional_error_model,
    transform_blq,
)
from pharmpy.modeling.blq import has_blq_transformation


@pytest.fixture(scope='module')
def model_with_blq_column(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    df = model.dataset.copy()
    df['BLQ'] = df.index < len(df) // 2
    di = create_datainfo(df)
    return model.replace(dataset=df, datainfo=di)


@pytest.fixture(scope='module')
def model_with_lloq_column(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    df = model.dataset.copy()
    df['LLOQ'] = 20.0
    di = create_datainfo(df)
    return model.replace(dataset=df, datainfo=di)


def test_transform_blq_m1(load_model_for_test, testdata, model_with_blq_column):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    assert len(model.dataset) == 744
    model = transform_blq(model, method='m1', lloq=20)
    assert len(model.dataset) == 703

    model = transform_blq(model_with_blq_column, method='m1')
    assert len(model.dataset) == 671


def test_transform_blq_m5(
    load_model_for_test, testdata, model_with_blq_column, model_with_lloq_column
):
    def get_lowest_dv(df):
        return df[df['DV'] != 0]['DV'].min()

    model = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    assert len(model.dataset) == 744
    assert get_lowest_dv(model.dataset) == 6.7
    model = transform_blq(model, method='m5', lloq=20)
    assert len(model.dataset) == 744
    assert get_lowest_dv(model.dataset) == 10.0

    model = transform_blq(model_with_blq_column, method='m5', lloq=20)
    assert len(model.dataset) == 744
    assert get_lowest_dv(model.dataset) == 10.0

    model = transform_blq(model_with_lloq_column, method='m5')
    assert len(model.dataset) == 744
    assert get_lowest_dv(model.dataset) == 10.0


@pytest.mark.parametrize(
    'blq_indicator_col, no_of_records',
    [
        (
            False,
            7,
        ),
        (
            True,
            6,
        ),
    ],
)
def test_transform_blq_m6(load_model_for_test, testdata, blq_indicator_col, no_of_records):
    data = {
        'ID': [1, 1, 1, 1, 2, 2, 2, 2],
        'MDV': [1, 1, 0, 0, 1, 1, 0, 0],
        'DV': [0, 1, 2, 3, 4, 5, 6, 7],
    }
    if blq_indicator_col:
        data['BLQ'] = [0, 0, 1, 1, 0, 0, 1, 1]

    df = pd.DataFrame(data)
    di = create_datainfo(df)

    model = load_model_for_test(testdata / 'nonmem' / 'minimal.mod')
    model = model.replace(dataset=df, datainfo=di)
    assert len(model.dataset) == 8
    model = transform_blq(model, method='m6', lloq=4)
    assert len(model.dataset) == no_of_records


def test_transform_blq_m6_lloq(load_model_for_test, testdata):
    data = {
        'ID': [1, 1, 1, 1, 2, 2, 2, 2],
        'MDV': [1, 1, 0, 0, 1, 1, 0, 0],
        'LLOQ': [4, 4, 4, 4, 4, 4, 4, 4],
        'DV': [0, 1, 2, 3, 4, 5, 6, 7],
    }
    df = pd.DataFrame(data)
    di = create_datainfo(df)

    model = load_model_for_test(testdata / 'nonmem' / 'minimal.mod')
    model = model.replace(dataset=df, datainfo=di)
    assert len(model.dataset) == 8
    model = transform_blq(model, method='m6')
    assert len(model.dataset) == 7


def test_transform_blq_m7(load_model_for_test, testdata, model_with_blq_column):
    def get_lowest_dv(df):
        return df[df['DV'] != 0]['DV'].min()

    model = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    assert len(model.dataset) == 744
    assert get_lowest_dv(model.dataset) == 6.7
    model = transform_blq(model, method='m7', lloq=20)
    assert len(model.dataset) == 744
    assert get_lowest_dv(model.dataset) > 20.0

    model = transform_blq(model_with_blq_column, method='m7')
    assert len(model.dataset) == 744
    assert get_lowest_dv(model.dataset) == 12.7


@pytest.mark.parametrize(
    'method, error_func, sd_ref, y_ref',
    [
        (
            'm4',
            set_additive_error_model,
            'SD = SQRT(SIGMA(1,1))',
            ('Y = F + EPS(1)', 'Y = (CUMD - CUMDZ)/(1 - CUMDZ)'),
        ),
        (
            'm4',
            set_proportional_error_model,
            'SD = SQRT(SIGMA(1,1))*ABS(F)',
            ('Y = F + EPS(1)*F', 'Y = (CUMD - CUMDZ)/(1 - CUMDZ)'),
        ),
        (
            'm4',
            set_combined_error_model,
            'SD = SQRT(F**2*SIGMA(1,1) + SIGMA(2,2))',
            ('Y = F + EPS(1)*F + EPS(2)', 'Y = (CUMD - CUMDZ)/(1 - CUMDZ)'),
        ),
        (
            'm4',
            set_power_on_ruv,
            'SD = SQRT(SIGMA(1,1))*SQRT(F**(2*THETA(3)))',
            ('Y = F + EPS(1)*F**THETA(3)', 'Y = (CUMD - CUMDZ)/(1 - CUMDZ)'),
        ),
        (
            'm3',
            set_additive_error_model,
            'SD = SQRT(SIGMA(1,1))',
            ('Y = F + EPS(1)', 'Y = PHI((-F + LLOQ)/SD)'),
        ),
    ],
)
def test_transform_blq(load_model_for_test, testdata, method, error_func, sd_ref, y_ref):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    model = error_func(model)

    model = transform_blq(model, method=method, lloq=0.1)

    assert sd_ref in model.code
    assert all(statement in model.code for statement in y_ref)

    assert all(est.laplace for est in model.execution_steps)


def test_transform_blq_raises(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    with pytest.raises(ValueError):
        transform_blq(model, method='m1')


@pytest.mark.parametrize(
    'method, error_func_before, error_func_after, args, sd_ref, y_ref',
    [
        (
            'm4',
            set_additive_error_model,
            set_proportional_error_model,
            {'zero_protection': False},
            'SD = SQRT(SIGMA(1,1))*ABS(F)',
            ('Y = F + EPS(1)*F', 'Y = (CUMD - CUMDZ)/(1 - CUMDZ)'),
        ),
        (
            'm4',
            set_additive_error_model,
            set_proportional_error_model,
            {},
            'SD = SQRT(SIGMA(1,1))*ABS(IPREDADJ)',
            ('Y = F + EPS(1)*IPREDADJ', 'Y = (CUMD - CUMDZ)/(1 - CUMDZ)'),
        ),
        (
            'm4',
            set_additive_error_model,
            set_combined_error_model,
            {},
            'SD = SQRT(F**2*SIGMA(1,1) + SIGMA(2,2))',
            ('Y = F + EPS(1)*F + EPS(2)', 'Y = (CUMD - CUMDZ)/(1 - CUMDZ)'),
        ),
        (
            'm4',
            set_proportional_error_model,
            set_combined_error_model,
            {},
            'SD = SQRT(F**2*SIGMA(1,1) + SIGMA(2,2))',
            ('Y = F + EPS(1)*F + EPS(2)', 'Y = (CUMD - CUMDZ)/(1 - CUMDZ)'),
        ),
        (
            'm4',
            set_additive_error_model,
            set_power_on_ruv,
            {},
            'SD = SQRT(F**(2*THETA(3))*SIGMA(1,1))',
            ('Y = F + EPS(1)*F**THETA(3)', 'Y = (CUMD - CUMDZ)/(1 - CUMDZ)'),
        ),
        (
            'm4',
            set_proportional_error_model,
            set_power_on_ruv,
            {},
            'SD = SQRT(IPREDADJ**(2*THETA(3))*SIGMA(1,1))',
            ('Y = F + EPS(1)*IPREDADJ**THETA(3)', 'Y = (CUMD - CUMDZ)/(1 - CUMDZ)'),
        ),
        (
            'm3',
            set_proportional_error_model,
            set_additive_error_model,
            {},
            'SD = SQRT(SIGMA(1,1))',
            ('Y = F + EPS(1)', 'Y = PHI((-F + LLOQ)/SD)'),
        ),
    ],
)
def test_update_blq_transformation(
    load_model_for_test, testdata, method, error_func_before, error_func_after, args, sd_ref, y_ref
):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    model = remove_error_model(model)
    model = error_func_before(model)

    model = transform_blq(model, method=method, lloq=0.1)

    model = error_func_after(model, **args)

    assert sd_ref in model.code
    assert all(statement in model.code for statement in y_ref)

    assert all(est.laplace for est in model.execution_steps)


@pytest.mark.parametrize(
    'method, error_func',
    [
        ('m4', set_additive_error_model),
        ('m4', set_proportional_error_model),
        ('m4', set_combined_error_model),
        ('m4', set_power_on_ruv),
        ('m3', set_additive_error_model),
    ],
)
def test_has_blq_transformation(load_model_for_test, testdata, method, error_func):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    model = error_func(model)

    assert not has_blq_transformation(model, Expr.symbol('Y'))

    model = transform_blq(model, method=method, lloq=0.1)

    assert has_blq_transformation(model, Expr.symbol('Y'))


def test_transform_blq_invalid_input_model(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    model = set_combined_error_model(model)
    model = create_joint_distribution(model, model.random_variables.epsilons.names)
    with pytest.raises(ValueError, match='Invalid input model: covariance between epsilons'):
        transform_blq(model, method='m4', lloq=0.1)

    model = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    model = set_iiv_on_ruv(model)
    with pytest.raises(ValueError, match='Invalid input model: error model not supported'):
        transform_blq(model, method='m4', lloq=0.1)


def test_transform_blq_different_lloq(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    model_float = transform_blq(model, lloq=0.1)

    assert 'DV.GE.LLOQ' in model_float.code

    df_blq = model.dataset
    df_blq['BLQ'] = np.random.randint(0, 2, df_blq.shape[0])
    di_blq = update_datainfo(model.datainfo, df_blq)
    blq_var = di_blq['BLQ'].variable.replace(type='blq')
    blq_col = di_blq['BLQ'].replace(variable_mapping=blq_var)
    di_blq = di_blq.set_column(blq_col)
    model_blq = model.replace(dataset=df_blq, datainfo=di_blq)

    with pytest.raises(ValueError):
        transform_blq(model_blq)

    df_lloq = model.dataset
    df_lloq['LLOQ'] = np.random.random(df_lloq.shape[0])
    di_lloq = update_datainfo(model.datainfo, df_lloq)
    lloq_var = di_lloq['LLOQ'].variable.replace(type='lloq')
    lloq_col = di_lloq['LLOQ'].replace(variable_mapping=lloq_var)
    di_lloq = di_lloq.set_column(lloq_col)
    model_lloq = model.replace(dataset=df_lloq, datainfo=di_lloq)

    model_lloq_col = transform_blq(model_lloq)

    assert 'DV.GE.LLOQ' in model_lloq_col.code
    assert 'LLOQ = ' not in model_lloq_col.code

    model_float_with_blq_col = transform_blq(model_blq, lloq=0.1)

    assert 'BLQ.EQ.0' in model_float_with_blq_col.code
    assert 'LLOQ = ' in model_float_with_blq_col.code


def test_has_blq_transformation_blq_col(model_with_blq_column):
    model = transform_blq(model_with_blq_column, method='m3', lloq=20.0)
    assert has_blq_transformation(model, Expr.symbol('Y'))


def test_has_blq_transformation_lloq_col(model_with_lloq_column):
    model = transform_blq(model_with_lloq_column, method='m3')
    assert has_blq_transformation(model, Expr.symbol('Y'))


def test_transform_blq_raises_no_y(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')

    with pytest.raises(ValueError):
        has_blq_transformation(model, Expr.symbol('X'))
