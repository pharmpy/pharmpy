import re

import numpy as np
import pytest

from pharmpy.modeling import add_covariate_effect, has_covariate_effect, remove_covariate_effect

from ..lib import diff


def test_nan_add_covariate_effect(load_model_for_test, pheno_path):
    model = load_model_for_test(pheno_path)
    data = model.dataset

    new_col = [np.nan] * 10 + ([1.0] * (len(data.index) - 10))

    data['new_col'] = new_col
    model.dataset = data

    add_covariate_effect(model, 'CL', 'new_col', 'cat')
    model.update_source(nofiles=True)

    assert not re.search('NaN', model.model_code)
    assert re.search(r'NEW_COL\.EQ\.-99', model.model_code)


def test_nested_add_covariate_effect(load_model_for_test, testdata):
    model_path = testdata / 'nonmem' / 'pheno.mod'
    model = load_model_for_test(model_path)

    add_covariate_effect(model, 'CL', 'WGT', 'exp')
    assert has_covariate_effect(model, 'CL', 'WGT')

    with pytest.warns(UserWarning, match='Covariate effect of WGT on CL already exists'):
        add_covariate_effect(model, 'CL', 'WGT', 'exp')

    assert has_covariate_effect(model, 'CL', 'WGT')

    model = load_model_for_test(model_path)

    add_covariate_effect(model, 'CL', 'WGT', 'exp')
    assert has_covariate_effect(model, 'CL', 'WGT')
    assert not has_covariate_effect(model, 'CL', 'APGR')
    add_covariate_effect(model, 'CL', 'APGR', 'exp')
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
        add_covariate_effect(model, *effect, allow_nested=allow_nested)

    for effect in effects:
        assert has_covariate_effect(model, effect[0], effect[1])

    model.update_source()
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
            remove_covariate_effect(model, effect[0], effect[1])

        for effect in effects:
            assert not has_covariate_effect(model, effect[0], effect[1])

        model.update_source()
        assert (
            diff(
                original_model.internals.control_stream.get_pred_pk_record(),
                model.internals.control_stream.get_pred_pk_record(),
            )
            == ''
        )

        for effect in effects:
            assert f'POP_{effect[0]}{effect[1]}' not in model.model_code
