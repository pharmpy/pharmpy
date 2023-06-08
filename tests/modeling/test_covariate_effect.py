import re

import pytest

from pharmpy.deps import numpy as np
from pharmpy.deps import sympy
from pharmpy.modeling.covariate_effect import (
    CovariateEffect,
    _choose_param_inits,
    add_covariate_effect,
    has_covariate_effect,
    remove_covariate_effect,
)

from ..lib import diff


def S(x: str):
    return sympy.Symbol(x)


def test_nan_add_covariate_effect(load_model_for_test, pheno_path):
    model = load_model_for_test(pheno_path)
    data = model.dataset.copy()

    new_col = [np.nan] * 10 + ([1.0] * (len(data.index) - 10))

    data['new_col'] = new_col
    model = model.replace(dataset=data)

    model = add_covariate_effect(model, 'CL', 'new_col', 'cat')

    assert not re.search('NaN', model.model_code)
    assert re.search(r'NEW_COL\.EQ\.-99', model.model_code)


def test_nested_add_covariate_effect(load_model_for_test, testdata):
    model_path = testdata / 'nonmem' / 'pheno.mod'
    model = load_model_for_test(model_path)

    model = add_covariate_effect(model, 'CL', 'WGT', 'exp')
    assert has_covariate_effect(model, 'CL', 'WGT')

    with pytest.warns(UserWarning, match='Covariate effect of WGT on CL already exists'):
        add_covariate_effect(model, 'CL', 'WGT', 'exp')

    assert has_covariate_effect(model, 'CL', 'WGT')

    model = load_model_for_test(model_path)

    model = add_covariate_effect(model, 'CL', 'WGT', 'exp')
    assert has_covariate_effect(model, 'CL', 'WGT')
    assert not has_covariate_effect(model, 'CL', 'APGR')
    model = add_covariate_effect(model, 'CL', 'APGR', 'exp')
    assert has_covariate_effect(model, 'CL', 'WGT')
    assert has_covariate_effect(model, 'CL', 'APGR')

    assert 'CL = CL*CLAPGR*CLWGT' in model.model_code
    assert 'CL = CL*CLWGT' not in model.model_code


@pytest.mark.parametrize(
    ('model_path', 'effects', 'expected', 'allow_nested'),
    [
        (
            ('nonmem', 'pheno.mod'),
            [('CL', 'WGT', 'exp', '*')],
            '@@ -1,0 +2 @@\n'
            '+WGT_MEDIAN = 1.30000\n'
            '@@ -2,0 +4,2 @@\n'
            '+CLWGT = EXP(THETA(3)*(WGT - WGT_MEDIAN))\n'
            '+CL = CL*CLWGT\n',
            False,
        ),
        (
            ('nonmem', 'pheno.mod'),
            [('CL', 'WGT', 'exp', '+')],
            '@@ -1,0 +2 @@\n'
            '+WGT_MEDIAN = 1.30000\n'
            '@@ -2,0 +4,2 @@\n'
            '+CLWGT = EXP(THETA(3)*(WGT - WGT_MEDIAN))\n'
            '+CL = CL + CLWGT\n',
            False,
        ),
        (
            ('nonmem', 'pheno.mod'),
            [('CL', 'APGR', 'cat', '*')],
            '@@ -2,0 +3,22 @@\n'
            '+IF (APGR.EQ.7.0) THEN\n'
            '+    CLAPGR = 1\n'
            '+ELSE IF (APGR.EQ.1.0) THEN\n'
            '+    CLAPGR = THETA(3) + 1\n'
            '+ELSE IF (APGR.EQ.2.0) THEN\n'
            '+    CLAPGR = THETA(4) + 1\n'
            '+ELSE IF (APGR.EQ.3.0) THEN\n'
            '+    CLAPGR = THETA(5) + 1\n'
            '+ELSE IF (APGR.EQ.4.0) THEN\n'
            '+    CLAPGR = THETA(6) + 1\n'
            '+ELSE IF (APGR.EQ.5.0) THEN\n'
            '+    CLAPGR = THETA(7) + 1\n'
            '+ELSE IF (APGR.EQ.6.0) THEN\n'
            '+    CLAPGR = THETA(8) + 1\n'
            '+ELSE IF (APGR.EQ.8.0) THEN\n'
            '+    CLAPGR = THETA(9) + 1\n'
            '+ELSE IF (APGR.EQ.9.0) THEN\n'
            '+    CLAPGR = THETA(10) + 1\n'
            '+ELSE IF (APGR.EQ.10.0) THEN\n'
            '+    CLAPGR = THETA(11) + 1\n'
            '+END IF\n'
            '+CL = CL*CLAPGR\n',
            False,
        ),
        (
            ('nonmem', 'pheno_real.mod'),
            [('CL', 'WGT', 'exp', '*')],
            '@@ -1,0 +2 @@\n'
            '+WGT_MEDIAN = 1.30000\n'
            '@@ -7,0 +9,2 @@\n'
            '+CLWGT = EXP(THETA(4)*(WGT - WGT_MEDIAN))\n'
            '+CL = CL*CLWGT\n',
            True,
        ),
        (
            ('nonmem', 'pheno_real.mod'),
            [('CL', 'WGT', 'exp', '+')],
            '@@ -1,0 +2 @@\n'
            '+WGT_MEDIAN = 1.30000\n'
            '@@ -7,0 +9,2 @@\n'
            '+CLWGT = EXP(THETA(4)*(WGT - WGT_MEDIAN))\n'
            '+CL = CL + CLWGT\n',
            True,
        ),
        (
            ('nonmem', 'pheno_real.mod'),
            [('CL', 'WGT', 'pow', '*')],
            '@@ -1,0 +2 @@\n'
            '+WGT_MEDIAN = 1.30000\n'
            '@@ -7,0 +9,2 @@\n'
            '+CLWGT = (WGT/WGT_MEDIAN)**THETA(4)\n'
            '+CL = CL*CLWGT\n',
            True,
        ),
        (
            ('nonmem', 'pheno_real.mod'),
            [('CL', 'WGT', 'lin', '*')],
            '@@ -1,0 +2 @@\n'
            '+WGT_MEDIAN = 1.30000\n'
            '@@ -7,0 +9,2 @@\n'
            '+CLWGT = THETA(4)*(WGT - WGT_MEDIAN) + 1\n'
            '+CL = CL*CLWGT\n',
            True,
        ),
        (
            ('nonmem', 'pheno_real.mod'),
            [('CL', 'APGR', 'cat', '*')],
            '@@ -7,0 +8,22 @@\n'
            '+IF (APGR.EQ.7.0) THEN\n'
            '+    CLAPGR = 1\n'
            '+ELSE IF (APGR.EQ.1.0) THEN\n'
            '+    CLAPGR = THETA(4) + 1\n'
            '+ELSE IF (APGR.EQ.2.0) THEN\n'
            '+    CLAPGR = THETA(5) + 1\n'
            '+ELSE IF (APGR.EQ.3.0) THEN\n'
            '+    CLAPGR = THETA(6) + 1\n'
            '+ELSE IF (APGR.EQ.4.0) THEN\n'
            '+    CLAPGR = THETA(7) + 1\n'
            '+ELSE IF (APGR.EQ.5.0) THEN\n'
            '+    CLAPGR = THETA(8) + 1\n'
            '+ELSE IF (APGR.EQ.6.0) THEN\n'
            '+    CLAPGR = THETA(9) + 1\n'
            '+ELSE IF (APGR.EQ.8.0) THEN\n'
            '+    CLAPGR = THETA(10) + 1\n'
            '+ELSE IF (APGR.EQ.9.0) THEN\n'
            '+    CLAPGR = THETA(11) + 1\n'
            '+ELSE IF (APGR.EQ.10.0) THEN\n'
            '+    CLAPGR = THETA(12) + 1\n'
            '+END IF\n'
            '+CL = CL*CLAPGR\n',
            True,
        ),
        (
            ('nonmem', 'pheno_real.mod'),
            [('CL', 'WGT', 'piece_lin', '*')],
            '@@ -1,0 +2 @@\n'
            '+WGT_MEDIAN = 1.30000\n'
            '@@ -7,0 +9,6 @@\n'
            '+IF (WGT.LE.WGT_MEDIAN) THEN\n'
            '+    CLWGT = THETA(4)*(WGT - WGT_MEDIAN) + 1\n'
            '+ELSE\n'
            '+    CLWGT = THETA(5)*(WGT - WGT_MEDIAN) + 1\n'
            '+END IF\n'
            '+CL = CL*CLWGT\n',
            True,
        ),
        (
            ('nonmem', 'pheno_real.mod'),
            [('CL', 'WGT', 'theta - cov + median', '*')],
            '@@ -1,0 +2 @@\n'
            '+WGT_MEDIAN = 1.30000\n'
            '@@ -7,0 +9,2 @@\n'
            '+CLWGT = THETA(4) - WGT + WGT_MEDIAN\n'
            '+CL = CL*CLWGT\n',
            True,
        ),
        (
            ('nonmem', 'pheno_real.mod'),
            [('CL', 'WGT', 'theta - cov + std', '*')],
            '@@ -1,0 +2 @@\n'
            '+WGT_STD = 0.704565\n'
            '@@ -7,0 +9,2 @@\n'
            '+CLWGT = THETA(4) - WGT + WGT_STD\n'
            '+CL = CL*CLWGT\n',
            True,
        ),
        (
            ('nonmem', 'pheno_real.mod'),
            [('CL', 'WGT', 'theta1 * (cov/median)**theta2', '*')],
            '@@ -1,0 +2 @@\n'
            '+WGT_MEDIAN = 1.30000\n'
            '@@ -7,0 +9,2 @@\n'
            '+CLWGT = THETA(4)*(WGT/WGT_MEDIAN)**THETA(5)\n'
            '+CL = CL*CLWGT\n',
            True,
        ),
        (
            ('nonmem', 'pheno_real.mod'),
            [('CL', 'WGT', '((cov/std) - median) * theta', '*')],
            '@@ -1,0 +2,2 @@\n'
            '+WGT_MEDIAN = 1.30000\n'
            '+WGT_STD = 0.704565\n'
            '@@ -7,0 +10,2 @@\n'
            '+CLWGT = THETA(4)*(WGT/WGT_STD - WGT_MEDIAN)\n'
            '+CL = CL*CLWGT\n',
            True,
        ),
        (
            ('nonmem', 'pheno_real.mod'),
            [
                ('CL', 'WGT', 'exp', '+'),
                ('V', 'WGT', 'exp', '+'),
            ],
            '@@ -1,0 +2 @@\n'
            '+WGT_MEDIAN = 1.30000\n'
            '@@ -7,0 +9,2 @@\n'
            '+CLWGT = EXP(THETA(4)*(WGT - WGT_MEDIAN))\n'
            '+CL = CL + CLWGT\n'
            '@@ -8,0 +12,2 @@\n'
            '+VWGT = EXP(THETA(5)*(WGT - WGT_MEDIAN))\n'
            '+V = V + VWGT\n',
            True,
        ),
        (
            ('nonmem', 'pheno_real.mod'),
            [
                ('CL', 'WGT', 'exp', '*'),
                ('V', 'WGT', 'exp', '*'),
            ],
            '@@ -1,0 +2 @@\n'
            '+WGT_MEDIAN = 1.30000\n'
            '@@ -7,0 +9,2 @@\n'
            '+CLWGT = EXP(THETA(4)*(WGT - WGT_MEDIAN))\n'
            '+CL = CL*CLWGT\n'
            '@@ -8,0 +12,2 @@\n'
            '+VWGT = EXP(THETA(5)*(WGT - WGT_MEDIAN))\n'
            '+V = V*VWGT\n',
            True,
        ),
        (
            ('nonmem', 'models', 'mox2.mod'),
            [
                ('V', 'WT', 'exp', '*'),
                ('V', 'AGE', 'exp', '*'),
            ],
            '@@ -1,0 +2,2 @@\n'
            '+AGE_MEDIAN = 66.0000\n'
            '+WT_MEDIAN = 78.0000\n'
            '@@ -6,0 +9,3 @@\n'
            '+VWT = EXP(THETA(4)*(WT - WT_MEDIAN))\n'
            '+VAGE = EXP(THETA(5)*(AGE - AGE_MEDIAN))\n'
            '+V = V*VAGE*VWT\n',
            False,
        ),
        (
            ('nonmem', 'models', 'mox2.mod'),
            [
                ('V', 'WT', 'exp', '*'),
                ('V', 'AGE', 'exp', '*'),
                ('V', 'CLCR', 'exp', '*'),
            ],
            '@@ -1,0 +2,3 @@\n'
            '+CLCR_MEDIAN = 65.0000\n'
            '+AGE_MEDIAN = 66.0000\n'
            '+WT_MEDIAN = 78.0000\n'
            '@@ -6,0 +10,4 @@\n'
            '+VWT = EXP(THETA(4)*(WT - WT_MEDIAN))\n'
            '+VAGE = EXP(THETA(5)*(AGE - AGE_MEDIAN))\n'
            '+VCLCR = EXP(THETA(6)*(CLCR - CLCR_MEDIAN))\n'
            '+V = V*VAGE*VCLCR*VWT\n',
            False,
        ),
        (
            ('nonmem', 'models', 'mox2.mod'),
            [
                ('CL', 'CLCR', 'exp', '*'),
                ('V', 'CLCR', 'exp', '*'),
            ],
            '@@ -1,0 +2 @@\n'
            '+CLCR_MEDIAN = 65.0000\n'
            '@@ -2,0 +4,2 @@\n'
            '+CLCLCR = EXP(THETA(4)*(CLCR - CLCR_MEDIAN))\n'
            '+CL = CL*CLCLCR\n'
            '@@ -6,0 +10,2 @@\n'
            '+VCLCR = EXP(THETA(5)*(CLCR - CLCR_MEDIAN))\n'
            '+V = V*VCLCR\n',
            False,
        ),
        (
            ('nonmem', 'models', 'mox2.mod'),
            [
                ('V', 'CLCR', 'exp', '*'),
                ('CL', 'CLCR', 'exp', '*'),
            ],
            '@@ -1,0 +2 @@\n'
            '+CLCR_MEDIAN = 65.0000\n'
            '@@ -2,0 +4,2 @@\n'
            '+CLCLCR = EXP(THETA(5)*(CLCR - CLCR_MEDIAN))\n'
            '+CL = CL*CLCLCR\n'
            '@@ -6,0 +10,2 @@\n'
            '+VCLCR = EXP(THETA(4)*(CLCR - CLCR_MEDIAN))\n'
            '+V = V*VCLCR\n',
            False,
        ),
        (
            ('nonmem', 'models', 'mox2.mod'),
            [('CL', 'WT', '((cov/std) - median) * theta', '*')],
            '@@ -1,0 +2,2 @@\n'
            '+WT_MEDIAN = 78.0000\n'
            '+WT_STD = 15.6125\n'
            '@@ -2,0 +5,2 @@\n'
            '+CLWT = THETA(4)*(WT/WT_STD - WT_MEDIAN)\n'
            '+CL = CL*CLWT\n',
            False,
        ),
        (
            ('nonmem', 'models', 'mox2.mod'),
            [('MAT', 'SEX', 'cat', '*')],
            '@@ -4,0 +5,6 @@\n'
            '+IF (SEX.EQ.1) THEN\n'
            '+    MATSEX = 1\n'
            '+ELSE IF (SEX.EQ.2) THEN\n'
            '+    MATSEX = THETA(4) + 1\n'
            '+END IF\n'
            '+MAT = MAT*MATSEX\n',
            False,
        ),
        (
            ('nonmem', 'models', 'mox2.mod'),
            [('V', 'WT', 'lin', '+')],
            '@@ -1,0 +2 @@\n'
            '+WT_MEDIAN = 78.0000\n'
            '@@ -6,0 +8,2 @@\n'
            '+VWT = THETA(4)*(WT - WT_MEDIAN) + 1\n'
            '+V = V + VWT\n',
            False,
        ),
        (
            ('nonmem', 'models', 'mox2.mod'),
            [('V', 'WT', 'pow', '*')],
            '@@ -1,0 +2 @@\n'
            '+WT_MEDIAN = 78.0000\n'
            '@@ -6,0 +8,2 @@\n'
            '+VWT = (WT/WT_MEDIAN)**THETA(4)\n'
            '+V = V*VWT\n',
            False,
        ),
        (
            ('nonmem', 'models', 'mox2.mod'),
            [('V', 'WT', 'piece_lin', '+')],
            '@@ -1,0 +2 @@\n'
            '+WT_MEDIAN = 78.0000\n'
            '@@ -6,0 +8,6 @@\n'
            '+IF (WT.LE.WT_MEDIAN) THEN\n'
            '+    VWT = THETA(4)*(WT - WT_MEDIAN) + 1\n'
            '+ELSE\n'
            '+    VWT = THETA(5)*(WT - WT_MEDIAN) + 1\n'
            '+END IF\n'
            '+V = V + VWT\n',
            False,
        ),
    ],
    ids=repr,
)
def test_add_covariate_effect(
    load_model_for_test, testdata, model_path, effects, expected, allow_nested
):
    model = load_model_for_test(testdata.joinpath(*model_path))
    error_record_before = ''.join(map(str, model.internals.control_stream.get_records('ERROR')))

    if not allow_nested:
        for effect in effects:
            assert not has_covariate_effect(model, effect[0], effect[1])

    for effect in effects:
        model = add_covariate_effect(model, *effect, allow_nested=allow_nested)

    for effect in effects:
        assert has_covariate_effect(model, effect[0], effect[1])

    error_record_after = ''.join(map(str, model.internals.control_stream.get_records('ERROR')))

    original_model = load_model_for_test(testdata.joinpath(*model_path))
    assert (
        diff(
            original_model.internals.control_stream.get_pred_pk_record(),
            model.internals.control_stream.get_pred_pk_record(),
        )
        == expected
    )
    assert error_record_after == error_record_before

    for effect in effects:
        assert f'POP_{effect[0]}{effect[1]}' in model.model_code

    if not allow_nested:
        for effect in effects:
            model = remove_covariate_effect(model, effect[0], effect[1])

        for effect in effects:
            assert not has_covariate_effect(model, effect[0], effect[1])

        assert (
            diff(
                original_model.internals.control_stream.get_pred_pk_record(),
                model.internals.control_stream.get_pred_pk_record(),
            )
            == ''
        )

        for effect in effects:
            assert f'POP_{effect[0]}{effect[1]}' not in model.model_code


@pytest.mark.parametrize(
    'cov_eff,symbol,expression',
    [
        (
            CovariateEffect.exponential(),
            S('CLWGT'),
            sympy.exp(S('COVEFF1') * (S('WGT') - S('WGT_MEDIAN'))),
        ),
        (CovariateEffect.power(), S('CLWGT'), (S('WGT') / S('WGT_MEDIAN')) ** S('COVEFF1')),
        (CovariateEffect.linear(), S('CLWGT'), 1 + S('COVEFF1') * (S('WGT') - S('WGT_MEDIAN'))),
    ],
)
def test_apply(cov_eff, symbol, expression):
    cov_eff.apply(
        parameter='CL',
        covariate='WGT',
        thetas={'theta': 'COVEFF1'},
        statistics={'mean': 1, 'median': 1, 'std': 1},
    )

    assert cov_eff.template.symbol == symbol
    assert cov_eff.template.expression == expression


@pytest.mark.parametrize(
    'cov_eff, init, lower, upper', [('exp', 0.001, -0.8696, 0.8696), ('pow', 0.001, -100, 100000)]
)
def test_choose_param_inits(pheno_path, load_model_for_test, cov_eff, init, lower, upper):
    model = load_model_for_test(pheno_path)

    inits = _choose_param_inits(cov_eff, model, 'WGT')

    assert inits['init'] == init
    assert inits['lower'] == lower
    assert inits['upper'] == upper


@pytest.mark.parametrize(
    ('model_path', 'effect', 'has'),
    [
        (
            ('nonmem', 'models', 'fviii6.mod'),
            ('CL', 'PREP'),
            True,
        ),
        (
            ('nonmem', 'models', 'fviii6.mod'),
            ('V1', 'PREP'),
            True,
        ),
        (
            ('nonmem', 'models', 'fviii6.mod'),
            ('V2', 'PREP'),
            False,
        ),
        (
            ('nonmem', 'models', 'fviii6.mod'),
            ('Q', 'PREP'),
            False,
        ),
        (
            ('nonmem', 'models', 'fviii6.mod'),
            ('CL', 'AGE'),
            True,
        ),
        (
            ('nonmem', 'models', 'fviii6.mod'),
            ('V1', 'AGE'),
            False,
        ),
        (
            ('nonmem', 'models', 'fviii6.mod'),
            ('V2', 'AGE'),
            False,
        ),
        (
            ('nonmem', 'models', 'fviii6.mod'),
            ('Q', 'AGE'),
            False,
        ),
        (
            ('nonmem', 'models', 'fviii6.mod'),
            ('CL', 'OCC'),
            True,
        ),
        (
            ('nonmem', 'models', 'fviii6.mod'),
            ('V1', 'OCC'),
            True,
        ),
        (
            ('nonmem', 'models', 'fviii6.mod'),
            ('V2', 'OCC'),
            False,
        ),
        (
            ('nonmem', 'models', 'fviii6.mod'),
            ('Q', 'OCC'),
            False,
        ),
        (
            ('nonmem', 'pheno_real.mod'),
            ('CL', 'WGT'),
            True,
        ),
        (
            ('nonmem', 'pheno_real.mod'),
            ('V', 'WGT'),
            True,
        ),
        (
            ('nonmem', 'pheno_real.mod'),
            ('CL', 'APGR'),
            False,
        ),
        (
            ('nonmem', 'pheno_real.mod'),
            ('V', 'APGR'),
            True,
        ),
        (
            ('nonmem', 'pheno.mod'),
            ('CL', 'WGT'),
            False,
        ),
        (
            ('nonmem', 'pheno.mod'),
            ('V', 'WGT'),
            False,
        ),
        (
            ('nonmem', 'pheno.mod'),
            ('CL', 'APGR'),
            False,
        ),
        (
            ('nonmem', 'pheno.mod'),
            ('V', 'APGR'),
            False,
        ),
    ],
    ids=repr,
)
def test_has_covariate_effect(load_model_for_test, testdata, model_path, effect, has):
    model = load_model_for_test(testdata.joinpath(*model_path))

    assert has_covariate_effect(model, *effect) is has


@pytest.mark.parametrize(
    ('model_path', 'effects', 'expected'),
    [
        (
            ('nonmem', 'pheno_real.mod'),
            [('CL', 'WGT')],
            '@@ -4 +4 @@\n' '-TVCL=THETA(1)*WGT\n' '+TVCL = THETA(1)\n',
        ),
        (
            ('nonmem', 'pheno_real.mod'),
            [('V', 'WGT')],
            '@@ -5 +5 @@\n' '-TVV=THETA(2)*WGT\n' '+TVV = THETA(2)\n',
        ),
        (
            ('nonmem', 'pheno_real.mod'),
            [('V', 'APGR')],
            '@@ -6 +5,0 @@\n' '-IF(APGR.LT.5) TVV=TVV*(1+THETA(3))\n',
        ),
        (
            ('nonmem', 'pheno_real.mod'),
            [('CL', 'WGT'), ('V', 'WGT')],
            '@@ -4,2 +4,2 @@\n'
            '-TVCL=THETA(1)*WGT\n'
            '-TVV=THETA(2)*WGT\n'
            '+TVCL = THETA(1)\n'
            '+TVV = THETA(2)\n',
        ),
        (
            ('nonmem', 'pheno_real.mod'),
            [('V', 'WGT'), ('CL', 'WGT')],
            '@@ -4,2 +4,2 @@\n'
            '-TVCL=THETA(1)*WGT\n'
            '-TVV=THETA(2)*WGT\n'
            '+TVCL = THETA(1)\n'
            '+TVV = THETA(2)\n',
        ),
        (
            ('nonmem', 'pheno_real.mod'),
            [('V', 'WGT'), ('CL', 'WGT'), ('V', 'APGR')],
            '@@ -4,3 +4,2 @@\n'
            '-TVCL=THETA(1)*WGT\n'
            '-TVV=THETA(2)*WGT\n'
            '-IF(APGR.LT.5) TVV=TVV*(1+THETA(3))\n'
            '+TVCL = THETA(1)\n'
            '+TVV = THETA(2)\n',
        ),
        (
            ('nonmem', 'models', 'fviii6.mod'),
            [('CL', 'WT')],
            '@@ -29,10 +29,6 @@\n'
            '-IF (PREP2.EQ.1) THEN\n'
            '-TVCL = THETA(1)*((WT/80)**THETA(7))*(1+THETA(13)*(AGE-40)) ;80 = Median of body weight\n'
            '-TVV1 = THETA(2)*((WT/80)**THETA(8)) \n'
            '-ENDIF\n'
            '-IF (PREP2.EQ.2) THEN\n'
            '-TVCL = THETA(1)*((WT/80)**THETA(7))*(1+THETA(12))*(1+THETA(13)*(AGE-40))\n'
            '-TVV1 = THETA(2)*((WT/80)**THETA(8))*(1+THETA(12))\n'
            '-ENDIF\n'
            '-TVQ  = THETA(5)*(WT/80)**THETA(9)\n'
            '-TVV2 = THETA(6)*(WT/80)**THETA(10)\n'
            '+IF (PREP2.EQ.1) TVCL = THETA(1)*(THETA(12)*(AGE - 40) + 1)\n'
            '+IF (PREP2.EQ.1) TVV1 = THETA(2)*(WT/80)**THETA(7)\n'
            '+IF (PREP2.EQ.2) TVCL = THETA(1)*(THETA(11) + 1)*(THETA(12)*(AGE - 40) + 1)\n'
            '+IF (PREP2.EQ.2) TVV1 = THETA(2)*(WT/80)**THETA(7)*(THETA(11) + 1)\n'
            '+TVQ = THETA(5)*(WT/80)**THETA(8)\n'
            '+TVV2 = THETA(6)*(WT/80)**THETA(9)\n'
            '@@ -48 +44 @@\n'
            '-TVBA = THETA(11)\n'
            '+TVBA = THETA(10)\n',
        ),
        (
            ('nonmem', 'models', 'fviii6.mod'),
            [('CL', 'WT'), ('CL', 'AGE')],
            '@@ -29,10 +29,6 @@\n'
            '-IF (PREP2.EQ.1) THEN\n'
            '-TVCL = THETA(1)*((WT/80)**THETA(7))*(1+THETA(13)*(AGE-40)) ;80 = Median of body weight\n'
            '-TVV1 = THETA(2)*((WT/80)**THETA(8)) \n'
            '-ENDIF\n'
            '-IF (PREP2.EQ.2) THEN\n'
            '-TVCL = THETA(1)*((WT/80)**THETA(7))*(1+THETA(12))*(1+THETA(13)*(AGE-40))\n'
            '-TVV1 = THETA(2)*((WT/80)**THETA(8))*(1+THETA(12))\n'
            '-ENDIF\n'
            '-TVQ  = THETA(5)*(WT/80)**THETA(9)\n'
            '-TVV2 = THETA(6)*(WT/80)**THETA(10)\n'
            '+IF (PREP2.EQ.1) TVCL = THETA(1)\n'
            '+IF (PREP2.EQ.1) TVV1 = THETA(2)*(WT/80)**THETA(7)\n'
            '+IF (PREP2.EQ.2) TVCL = THETA(1)*(THETA(11) + 1)\n'
            '+IF (PREP2.EQ.2) TVV1 = THETA(2)*(WT/80)**THETA(7)*(THETA(11) + 1)\n'
            '+TVQ = THETA(5)*(WT/80)**THETA(8)\n'
            '+TVV2 = THETA(6)*(WT/80)**THETA(9)\n'
            '@@ -48 +44 @@\n'
            '-TVBA = THETA(11)\n'
            '+TVBA = THETA(10)\n',
        ),
        (
            ('nonmem', 'models', 'fviii6.mod'),
            [('V1', 'WT')],
            '@@ -29,10 +29,6 @@\n'
            '-IF (PREP2.EQ.1) THEN\n'
            '-TVCL = THETA(1)*((WT/80)**THETA(7))*(1+THETA(13)*(AGE-40)) ;80 = Median of body weight\n'
            '-TVV1 = THETA(2)*((WT/80)**THETA(8)) \n'
            '-ENDIF\n'
            '-IF (PREP2.EQ.2) THEN\n'
            '-TVCL = THETA(1)*((WT/80)**THETA(7))*(1+THETA(12))*(1+THETA(13)*(AGE-40))\n'
            '-TVV1 = THETA(2)*((WT/80)**THETA(8))*(1+THETA(12))\n'
            '-ENDIF\n'
            '-TVQ  = THETA(5)*(WT/80)**THETA(9)\n'
            '-TVV2 = THETA(6)*(WT/80)**THETA(10)\n'
            '+IF (PREP2.EQ.1) TVCL = THETA(1)*(WT/80)**THETA(7)*(THETA(12)*(AGE - 40) + 1)\n'
            '+IF (PREP2.EQ.1) TVV1 = THETA(2)\n'
            '+IF (PREP2.EQ.2) TVCL = THETA(1)*(WT/80)**THETA(7)*(THETA(11) + 1)*(THETA(12)*(AGE - 40) + 1)\n'
            '+IF (PREP2.EQ.2) TVV1 = THETA(2)*(THETA(11) + 1)\n'
            '+TVQ = THETA(5)*(WT/80)**THETA(8)\n'
            '+TVV2 = THETA(6)*(WT/80)**THETA(9)\n'
            '@@ -48 +44 @@\n'
            '-TVBA = THETA(11)\n'
            '+TVBA = THETA(10)\n',
        ),
        (
            ('nonmem', 'models', 'fviii6.mod'),
            [('V2', 'WT')],
            '@@ -29,8 +29,4 @@\n'
            '-IF (PREP2.EQ.1) THEN\n'
            '-TVCL = THETA(1)*((WT/80)**THETA(7))*(1+THETA(13)*(AGE-40)) ;80 = Median of body weight\n'
            '-TVV1 = THETA(2)*((WT/80)**THETA(8)) \n'
            '-ENDIF\n'
            '-IF (PREP2.EQ.2) THEN\n'
            '-TVCL = THETA(1)*((WT/80)**THETA(7))*(1+THETA(12))*(1+THETA(13)*(AGE-40))\n'
            '-TVV1 = THETA(2)*((WT/80)**THETA(8))*(1+THETA(12))\n'
            '-ENDIF\n'
            '+IF (PREP2.EQ.1) TVCL = THETA(1)*(WT/80)**THETA(7)*(THETA(12)*(AGE - 40) + 1)\n'
            '+IF (PREP2.EQ.1) TVV1 = THETA(2)*(WT/80)**THETA(8)\n'
            '+IF (PREP2.EQ.2) TVCL = THETA(1)*(WT/80)**THETA(7)*(THETA(11) + 1)*(THETA(12)*(AGE - 40) + 1)\n'
            '+IF (PREP2.EQ.2) TVV1 = THETA(2)*(WT/80)**THETA(8)*(THETA(11) + 1)\n'
            '@@ -38 +34 @@\n'
            '-TVV2 = THETA(6)*(WT/80)**THETA(10)\n'
            '+TVV2 = THETA(6)\n'
            '@@ -48 +44 @@\n'
            '-TVBA = THETA(11)\n'
            '+TVBA = THETA(10)\n',
        ),
        (
            ('nonmem', 'models', 'fviii6.mod'),
            [('Q', 'WT')],
            '@@ -29,10 +29,6 @@\n'
            '-IF (PREP2.EQ.1) THEN\n'
            '-TVCL = THETA(1)*((WT/80)**THETA(7))*(1+THETA(13)*(AGE-40)) ;80 = Median of body weight\n'
            '-TVV1 = THETA(2)*((WT/80)**THETA(8)) \n'
            '-ENDIF\n'
            '-IF (PREP2.EQ.2) THEN\n'
            '-TVCL = THETA(1)*((WT/80)**THETA(7))*(1+THETA(12))*(1+THETA(13)*(AGE-40))\n'
            '-TVV1 = THETA(2)*((WT/80)**THETA(8))*(1+THETA(12))\n'
            '-ENDIF\n'
            '-TVQ  = THETA(5)*(WT/80)**THETA(9)\n'
            '-TVV2 = THETA(6)*(WT/80)**THETA(10)\n'
            '+IF (PREP2.EQ.1) TVCL = THETA(1)*(WT/80)**THETA(7)*(THETA(12)*(AGE - 40) + 1)\n'
            '+IF (PREP2.EQ.1) TVV1 = THETA(2)*(WT/80)**THETA(8)\n'
            '+IF (PREP2.EQ.2) TVCL = THETA(1)*(WT/80)**THETA(7)*(THETA(11) + 1)*(THETA(12)*(AGE - 40) + 1)\n'
            '+IF (PREP2.EQ.2) TVV1 = THETA(2)*(WT/80)**THETA(8)*(THETA(11) + 1)\n'
            '+TVQ = THETA(5)\n'
            '+TVV2 = THETA(6)*(WT/80)**THETA(9)\n'
            '@@ -48 +44 @@\n'
            '-TVBA = THETA(11)\n'
            '+TVBA = THETA(10)\n',
        ),
        (
            ('nonmem', 'models', 'fviii6.mod'),
            [('K21', 'WT')],
            '@@ -29,10 +29,6 @@\n'
            '-IF (PREP2.EQ.1) THEN\n'
            '-TVCL = THETA(1)*((WT/80)**THETA(7))*(1+THETA(13)*(AGE-40)) ;80 = Median of body weight\n'
            '-TVV1 = THETA(2)*((WT/80)**THETA(8)) \n'
            '-ENDIF\n'
            '-IF (PREP2.EQ.2) THEN\n'
            '-TVCL = THETA(1)*((WT/80)**THETA(7))*(1+THETA(12))*(1+THETA(13)*(AGE-40))\n'
            '-TVV1 = THETA(2)*((WT/80)**THETA(8))*(1+THETA(12))\n'
            '-ENDIF\n'
            '-TVQ  = THETA(5)*(WT/80)**THETA(9)\n'
            '-TVV2 = THETA(6)*(WT/80)**THETA(10)\n'
            '+IF (PREP2.EQ.1) TVCL = THETA(1)*(WT/80)**THETA(7)*(THETA(11)*(AGE - 40) + 1)\n'
            '+IF (PREP2.EQ.1) TVV1 = THETA(2)*(WT/80)**THETA(8)\n'
            '+IF (PREP2.EQ.2) TVCL = THETA(1)*(WT/80)**THETA(7)*(THETA(10) + 1)*(THETA(11)*(AGE - 40) + 1)\n'
            '+IF (PREP2.EQ.2) TVV1 = THETA(2)*(WT/80)**THETA(8)*(THETA(10) + 1)\n'
            '+TVQ = THETA(5)\n'
            '+TVV2 = THETA(6)\n'
            '@@ -48 +44 @@\n'
            '-TVBA = THETA(11)\n'
            '+TVBA = THETA(9)\n',
        ),
        (
            ('nonmem', 'models', 'fviii6.mod'),
            [('CL', 'WT'), ('V1', 'WT')],
            '@@ -29,10 +29,6 @@\n'
            '-IF (PREP2.EQ.1) THEN\n'
            '-TVCL = THETA(1)*((WT/80)**THETA(7))*(1+THETA(13)*(AGE-40)) ;80 = Median of body weight\n'
            '-TVV1 = THETA(2)*((WT/80)**THETA(8)) \n'
            '-ENDIF\n'
            '-IF (PREP2.EQ.2) THEN\n'
            '-TVCL = THETA(1)*((WT/80)**THETA(7))*(1+THETA(12))*(1+THETA(13)*(AGE-40))\n'
            '-TVV1 = THETA(2)*((WT/80)**THETA(8))*(1+THETA(12))\n'
            '-ENDIF\n'
            '-TVQ  = THETA(5)*(WT/80)**THETA(9)\n'
            '-TVV2 = THETA(6)*(WT/80)**THETA(10)\n'
            '+IF (PREP2.EQ.1) TVCL = THETA(1)*(THETA(11)*(AGE - 40) + 1)\n'
            '+IF (PREP2.EQ.1) TVV1 = THETA(2)\n'
            '+IF (PREP2.EQ.2) TVCL = THETA(1)*(THETA(10) + 1)*(THETA(11)*(AGE - 40) + 1)\n'
            '+IF (PREP2.EQ.2) TVV1 = THETA(2)*(THETA(10) + 1)\n'
            '+TVQ = THETA(5)*(WT/80)**THETA(7)\n'
            '+TVV2 = THETA(6)*(WT/80)**THETA(8)\n'
            '@@ -48 +44 @@\n'
            '-TVBA = THETA(11)\n'
            '+TVBA = THETA(9)\n',
        ),
        (
            ('nonmem', 'models', 'fviii6.mod'),
            [('V1', 'WT'), ('CL', 'AGE')],
            '@@ -29,10 +29,6 @@\n'
            '-IF (PREP2.EQ.1) THEN\n'
            '-TVCL = THETA(1)*((WT/80)**THETA(7))*(1+THETA(13)*(AGE-40)) ;80 = Median of body weight\n'
            '-TVV1 = THETA(2)*((WT/80)**THETA(8)) \n'
            '-ENDIF\n'
            '-IF (PREP2.EQ.2) THEN\n'
            '-TVCL = THETA(1)*((WT/80)**THETA(7))*(1+THETA(12))*(1+THETA(13)*(AGE-40))\n'
            '-TVV1 = THETA(2)*((WT/80)**THETA(8))*(1+THETA(12))\n'
            '-ENDIF\n'
            '-TVQ  = THETA(5)*(WT/80)**THETA(9)\n'
            '-TVV2 = THETA(6)*(WT/80)**THETA(10)\n'
            '+IF (PREP2.EQ.1) TVCL = THETA(1)*(WT/80)**THETA(7)\n'
            '+IF (PREP2.EQ.1) TVV1 = THETA(2)\n'
            '+IF (PREP2.EQ.2) TVCL = THETA(1)*(WT/80)**THETA(7)*(THETA(11) + 1)\n'
            '+IF (PREP2.EQ.2) TVV1 = THETA(2)*(THETA(11) + 1)\n'
            '+TVQ = THETA(5)*(WT/80)**THETA(8)\n'
            '+TVV2 = THETA(6)*(WT/80)**THETA(9)\n'
            '@@ -48 +44 @@\n'
            '-TVBA = THETA(11)\n'
            '+TVBA = THETA(10)\n',
        ),
        (
            ('nonmem', 'models', 'fviii6.mod'),
            [('CL', 'WT'), ('V1', 'WT'), ('V2', 'WT')],
            '@@ -29,10 +29,6 @@\n'
            '-IF (PREP2.EQ.1) THEN\n'
            '-TVCL = THETA(1)*((WT/80)**THETA(7))*(1+THETA(13)*(AGE-40)) ;80 = Median of body weight\n'
            '-TVV1 = THETA(2)*((WT/80)**THETA(8)) \n'
            '-ENDIF\n'
            '-IF (PREP2.EQ.2) THEN\n'
            '-TVCL = THETA(1)*((WT/80)**THETA(7))*(1+THETA(12))*(1+THETA(13)*(AGE-40))\n'
            '-TVV1 = THETA(2)*((WT/80)**THETA(8))*(1+THETA(12))\n'
            '-ENDIF\n'
            '-TVQ  = THETA(5)*(WT/80)**THETA(9)\n'
            '-TVV2 = THETA(6)*(WT/80)**THETA(10)\n'
            '+IF (PREP2.EQ.1) TVCL = THETA(1)*(THETA(10)*(AGE - 40) + 1)\n'
            '+IF (PREP2.EQ.1) TVV1 = THETA(2)\n'
            '+IF (PREP2.EQ.2) TVCL = THETA(1)*(THETA(9) + 1)*(THETA(10)*(AGE - 40) + 1)\n'
            '+IF (PREP2.EQ.2) TVV1 = THETA(2)*(THETA(9) + 1)\n'
            '+TVQ = THETA(5)*(WT/80)**THETA(7)\n'
            '+TVV2 = THETA(6)\n'
            '@@ -48 +44 @@\n'
            '-TVBA = THETA(11)\n'
            '+TVBA = THETA(8)\n',
        ),
        (
            ('nonmem', 'models', 'fviii6.mod'),
            [('CL', 'WT'), ('V1', 'WT'), ('Q', 'WT'), ('V2', 'WT')],
            '@@ -29,10 +29,6 @@\n'
            '-IF (PREP2.EQ.1) THEN\n'
            '-TVCL = THETA(1)*((WT/80)**THETA(7))*(1+THETA(13)*(AGE-40)) ;80 = Median of body weight\n'
            '-TVV1 = THETA(2)*((WT/80)**THETA(8)) \n'
            '-ENDIF\n'
            '-IF (PREP2.EQ.2) THEN\n'
            '-TVCL = THETA(1)*((WT/80)**THETA(7))*(1+THETA(12))*(1+THETA(13)*(AGE-40))\n'
            '-TVV1 = THETA(2)*((WT/80)**THETA(8))*(1+THETA(12))\n'
            '-ENDIF\n'
            '-TVQ  = THETA(5)*(WT/80)**THETA(9)\n'
            '-TVV2 = THETA(6)*(WT/80)**THETA(10)\n'
            '+IF (PREP2.EQ.1) TVCL = THETA(1)*(THETA(9)*(AGE - 40) + 1)\n'
            '+IF (PREP2.EQ.1) TVV1 = THETA(2)\n'
            '+IF (PREP2.EQ.2) TVCL = THETA(1)*(THETA(8) + 1)*(THETA(9)*(AGE - 40) + 1)\n'
            '+IF (PREP2.EQ.2) TVV1 = THETA(2)*(THETA(8) + 1)\n'
            '+TVQ = THETA(5)\n'
            '+TVV2 = THETA(6)\n'
            '@@ -48 +44 @@\n'
            '-TVBA = THETA(11)\n'
            '+TVBA = THETA(7)\n',
        ),
        (
            ('nonmem', 'models', 'fviii6.mod'),
            [('CL', 'WT'), ('V1', 'WT'), ('V2', 'WT'), ('CL', 'AGE')],
            '@@ -29,10 +29,6 @@\n'
            '-IF (PREP2.EQ.1) THEN\n'
            '-TVCL = THETA(1)*((WT/80)**THETA(7))*(1+THETA(13)*(AGE-40)) ;80 = Median of body weight\n'
            '-TVV1 = THETA(2)*((WT/80)**THETA(8)) \n'
            '-ENDIF\n'
            '-IF (PREP2.EQ.2) THEN\n'
            '-TVCL = THETA(1)*((WT/80)**THETA(7))*(1+THETA(12))*(1+THETA(13)*(AGE-40))\n'
            '-TVV1 = THETA(2)*((WT/80)**THETA(8))*(1+THETA(12))\n'
            '-ENDIF\n'
            '-TVQ  = THETA(5)*(WT/80)**THETA(9)\n'
            '-TVV2 = THETA(6)*(WT/80)**THETA(10)\n'
            '+IF (PREP2.EQ.1) TVCL = THETA(1)\n'
            '+IF (PREP2.EQ.1) TVV1 = THETA(2)\n'
            '+IF (PREP2.EQ.2) TVCL = THETA(1)*(THETA(9) + 1)\n'
            '+IF (PREP2.EQ.2) TVV1 = THETA(2)*(THETA(9) + 1)\n'
            '+TVQ = THETA(5)*(WT/80)**THETA(7)\n'
            '+TVV2 = THETA(6)\n'
            '@@ -48 +44 @@\n'
            '-TVBA = THETA(11)\n'
            '+TVBA = THETA(8)\n',
        ),
        (
            ('nonmem', 'models', 'fviii6.mod'),
            [('CL', 'WT'), ('V1', 'WT'), ('V2', 'WT'), ('CL', 'AGE'), ('Q', 'WT')],
            '@@ -29,10 +29,6 @@\n'
            '-IF (PREP2.EQ.1) THEN\n'
            '-TVCL = THETA(1)*((WT/80)**THETA(7))*(1+THETA(13)*(AGE-40)) ;80 = Median of body weight\n'
            '-TVV1 = THETA(2)*((WT/80)**THETA(8)) \n'
            '-ENDIF\n'
            '-IF (PREP2.EQ.2) THEN\n'
            '-TVCL = THETA(1)*((WT/80)**THETA(7))*(1+THETA(12))*(1+THETA(13)*(AGE-40))\n'
            '-TVV1 = THETA(2)*((WT/80)**THETA(8))*(1+THETA(12))\n'
            '-ENDIF\n'
            '-TVQ  = THETA(5)*(WT/80)**THETA(9)\n'
            '-TVV2 = THETA(6)*(WT/80)**THETA(10)\n'
            '+IF (PREP2.EQ.1) TVCL = THETA(1)\n'
            '+IF (PREP2.EQ.1) TVV1 = THETA(2)\n'
            '+IF (PREP2.EQ.2) TVCL = THETA(1)*(THETA(8) + 1)\n'
            '+IF (PREP2.EQ.2) TVV1 = THETA(2)*(THETA(8) + 1)\n'
            '+TVQ = THETA(5)\n'
            '+TVV2 = THETA(6)\n'
            '@@ -48 +44 @@\n'
            '-TVBA = THETA(11)\n'
            '+TVBA = THETA(7)\n',
        ),
    ],
    ids=repr,
)
def test_remove_covariate_effect(load_model_for_test, testdata, model_path, effects, expected):
    model = load_model_for_test(testdata.joinpath(*model_path))
    error_record_before = ''.join(map(str, model.internals.control_stream.get_records('ERROR')))

    for effect in effects:
        assert has_covariate_effect(model, effect[0], effect[1])

    for effect in effects:
        model = remove_covariate_effect(model, *effect)

    for effect in effects:
        assert not has_covariate_effect(model, effect[0], effect[1])

    error_record_after = ''.join(map(str, model.internals.control_stream.get_records('ERROR')))

    original_model = load_model_for_test(testdata.joinpath(*model_path))
    assert (
        diff(
            original_model.internals.control_stream.get_pred_pk_record(),
            model.internals.control_stream.get_pred_pk_record(),
        )
        == expected
    )
    assert error_record_after == error_record_before
