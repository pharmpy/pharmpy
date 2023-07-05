import pytest

from pharmpy.deps import numpy as np
from pharmpy.model.model import update_datainfo
from pharmpy.modeling import (
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

    assert sd_ref in model.model_code
    assert all(statement in model.model_code for statement in y_ref)

    assert all(est.laplace for est in model.estimation_steps)


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

    assert sd_ref in model.model_code
    assert all(statement in model.model_code for statement in y_ref)

    assert all(est.laplace for est in model.estimation_steps)


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

    assert not has_blq_transformation(model)

    model = transform_blq(model, method=method, lloq=0.1)

    assert has_blq_transformation(model)


def test_transform_blq_invalid_input_model(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    model = set_combined_error_model(model)
    model = create_joint_distribution(model, model.random_variables.epsilons.names)
    with pytest.raises(ValueError, match='Invalid input model: covariance between epsilons'):
        transform_blq(model, method='m4')

    model = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    model = set_iiv_on_ruv(model)
    with pytest.raises(ValueError, match='Invalid input model: error model not supported'):
        transform_blq(model, method='m4', lloq=0.1)


def test_transform_blq_different_lloq(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    model_float = transform_blq(model, lloq=0.1)

    assert 'DV.GE.LLOQ' in model_float.model_code

    df_blq = model.dataset
    df_blq['BLQ'] = np.random.randint(0, 2, df_blq.shape[0])
    di_blq = update_datainfo(model.datainfo, df_blq)
    blq_col = di_blq['BLQ'].replace(type='blqdv')
    di_blq = di_blq.set_column(blq_col)
    model_blq = model.replace(dataset=df_blq, datainfo=di_blq)

    model_blq_col = transform_blq(model_blq)

    assert 'BLQ.EQ.0' in model_blq_col.model_code

    df_lloq = model.dataset
    df_lloq['LLOQ'] = np.random.random(df_lloq.shape[0])
    di_lloq = update_datainfo(model.datainfo, df_lloq)
    lloq_col = di_lloq['LLOQ'].replace(type='lloq')
    di_lloq = di_lloq.set_column(lloq_col)
    model_lloq = model.replace(dataset=df_lloq, datainfo=di_lloq)

    model_lloq_col = transform_blq(model_lloq)

    assert 'DV.GE.LLOQ' in model_lloq_col.model_code
    assert 'LLOQ = ' not in model_lloq_col.model_code

    model_float_with_blq_col = transform_blq(model_blq, lloq=0.1)

    assert 'DV.GE.LLOQ' in model_float_with_blq_col.model_code
    assert 'LLOQ = ' in model_float_with_blq_col.model_code
