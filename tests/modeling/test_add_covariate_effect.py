import re

import numpy as np
import pytest

from pharmpy.modeling import add_covariate_effect


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

    with pytest.warns(UserWarning, match='Covariate effect of WGT on CL already exists'):
        add_covariate_effect(model, 'CL', 'WGT', 'exp')

    model = load_model_for_test(model_path)

    add_covariate_effect(model, 'CL', 'WGT', 'exp')
    add_covariate_effect(model, 'CL', 'APGR', 'exp')

    assert 'CL = CL*CLAPGR*CLWGT' in model.model_code
    assert 'CL = CL*CLWGT' not in model.model_code


@pytest.mark.parametrize(
    ('model_path', 'effects', 'expected', 'allow_nested'),
    [
        (
            ('nonmem', 'pheno.mod'),
            [('CL', 'WGT', 'exp', '*')],
            '$PK\n'
            'WGT_MEDIAN = 1.30000\n'
            'CL=THETA(1)*EXP(ETA(1))\n'
            'CLWGT = EXP(THETA(3)*(WGT - WGT_MEDIAN))\n'
            'CL = CL*CLWGT\n'
            'V=THETA(2)*EXP(ETA(2))\n'
            'S1=V\n\n',
            False,
        ),
        (
            ('nonmem', 'pheno.mod'),
            [('CL', 'WGT', 'exp', '+')],
            '$PK\n'
            'WGT_MEDIAN = 1.30000\n'
            'CL=THETA(1)*EXP(ETA(1))\n'
            'CLWGT = EXP(THETA(3)*(WGT - WGT_MEDIAN))\n'
            'CL = CL + CLWGT\n'
            'V=THETA(2)*EXP(ETA(2))\n'
            'S1=V\n\n',
            False,
        ),
        (
            ('nonmem', 'pheno_real.mod'),
            [('CL', 'WGT', 'exp', '*')],
            '$PK\n'
            'WGT_MEDIAN = 1.30000\n'
            'IF(AMT.GT.0) BTIME=TIME\n'
            'TAD=TIME-BTIME\n'
            'TVCL=THETA(1)*WGT\n'
            'TVV=THETA(2)*WGT\n'
            'IF(APGR.LT.5) TVV=TVV*(1+THETA(3))\n'
            'CL=TVCL*EXP(ETA(1))\n'
            'CLWGT = EXP(THETA(4)*(WGT - WGT_MEDIAN))\n'
            'CL = CL*CLWGT\n'
            'V=TVV*EXP(ETA(2))\n'
            'S1=V\n\n',
            True,
        ),
        (
            ('nonmem', 'pheno_real.mod'),
            [('CL', 'WGT', 'exp', '+')],
            '$PK\n'
            'WGT_MEDIAN = 1.30000\n'
            'IF(AMT.GT.0) BTIME=TIME\n'
            'TAD=TIME-BTIME\n'
            'TVCL=THETA(1)*WGT\n'
            'TVV=THETA(2)*WGT\n'
            'IF(APGR.LT.5) TVV=TVV*(1+THETA(3))\n'
            'CL=TVCL*EXP(ETA(1))\n'
            'CLWGT = EXP(THETA(4)*(WGT - WGT_MEDIAN))\n'
            'CL = CL + CLWGT\n'
            'V=TVV*EXP(ETA(2))\n'
            'S1=V\n\n',
            True,
        ),
        (
            ('nonmem', 'pheno_real.mod'),
            [('CL', 'WGT', 'pow', '*')],
            '$PK\n'
            'WGT_MEDIAN = 1.30000\n'
            'IF(AMT.GT.0) BTIME=TIME\n'
            'TAD=TIME-BTIME\n'
            'TVCL=THETA(1)*WGT\n'
            'TVV=THETA(2)*WGT\n'
            'IF(APGR.LT.5) TVV=TVV*(1+THETA(3))\n'
            'CL=TVCL*EXP(ETA(1))\n'
            'CLWGT = (WGT/WGT_MEDIAN)**THETA(4)\n'
            'CL = CL*CLWGT\n'
            'V=TVV*EXP(ETA(2))\n'
            'S1=V\n\n',
            True,
        ),
        (
            ('nonmem', 'pheno_real.mod'),
            [('CL', 'WGT', 'lin', '*')],
            '$PK\n'
            'WGT_MEDIAN = 1.30000\n'
            'IF(AMT.GT.0) BTIME=TIME\n'
            'TAD=TIME-BTIME\n'
            'TVCL=THETA(1)*WGT\n'
            'TVV=THETA(2)*WGT\n'
            'IF(APGR.LT.5) TVV=TVV*(1+THETA(3))\n'
            'CL=TVCL*EXP(ETA(1))\n'
            'CLWGT = THETA(4)*(WGT - WGT_MEDIAN) + 1\n'
            'CL = CL*CLWGT\n'
            'V=TVV*EXP(ETA(2))\n'
            'S1=V\n\n',
            True,
        ),
        (
            ('nonmem', 'pheno_real.mod'),
            [('CL', 'FA1', 'cat', '*')],
            '$PK\n'
            'IF(AMT.GT.0) BTIME=TIME\n'
            'TAD=TIME-BTIME\n'
            'TVCL=THETA(1)*WGT\n'
            'TVV=THETA(2)*WGT\n'
            'IF(APGR.LT.5) TVV=TVV*(1+THETA(3))\n'
            'CL=TVCL*EXP(ETA(1))\n'
            'IF (FA1.EQ.0) THEN\n'
            '    CLFA1 = 1\n'
            'ELSE IF (FA1.EQ.1.0) THEN\n'
            '    CLFA1 = THETA(4) + 1\n'
            'END IF\n'
            'CL = CL*CLFA1\n'
            'V=TVV*EXP(ETA(2))\n'
            'S1=V\n\n',
            True,
        ),
        (
            ('nonmem', 'pheno_real.mod'),
            [('CL', 'WGT', 'piece_lin', '*')],
            '$PK\n'
            'WGT_MEDIAN = 1.30000\n'
            'IF(AMT.GT.0) BTIME=TIME\n'
            'TAD=TIME-BTIME\n'
            'TVCL=THETA(1)*WGT\n'
            'TVV=THETA(2)*WGT\n'
            'IF(APGR.LT.5) TVV=TVV*(1+THETA(3))\n'
            'CL=TVCL*EXP(ETA(1))\n'
            'IF (WGT.LE.WGT_MEDIAN) THEN\n'
            '    CLWGT = THETA(4)*(WGT - WGT_MEDIAN) + 1\n'
            'ELSE\n'
            '    CLWGT = THETA(5)*(WGT - WGT_MEDIAN) + 1\n'
            'END IF\n'
            'CL = CL*CLWGT\n'
            'V=TVV*EXP(ETA(2))\n'
            'S1=V\n\n',
            True,
        ),
        (
            ('nonmem', 'pheno_real.mod'),
            [('CL', 'WGT', 'theta - cov + median', '*')],
            '$PK\n'
            'WGT_MEDIAN = 1.30000\n'
            'IF(AMT.GT.0) BTIME=TIME\n'
            'TAD=TIME-BTIME\n'
            'TVCL=THETA(1)*WGT\n'
            'TVV=THETA(2)*WGT\n'
            'IF(APGR.LT.5) TVV=TVV*(1+THETA(3))\n'
            'CL=TVCL*EXP(ETA(1))\n'
            'CLWGT = THETA(4) - WGT + WGT_MEDIAN\n'
            'CL = CL*CLWGT\n'
            'V=TVV*EXP(ETA(2))\n'
            'S1=V\n\n',
            True,
        ),
        (
            ('nonmem', 'pheno_real.mod'),
            [('CL', 'WGT', 'theta - cov + std', '*')],
            '$PK\n'
            'WGT_STD = 0.704565\n'
            'IF(AMT.GT.0) BTIME=TIME\n'
            'TAD=TIME-BTIME\n'
            'TVCL=THETA(1)*WGT\n'
            'TVV=THETA(2)*WGT\n'
            'IF(APGR.LT.5) TVV=TVV*(1+THETA(3))\n'
            'CL=TVCL*EXP(ETA(1))\n'
            'CLWGT = THETA(4) - WGT + WGT_STD\n'
            'CL = CL*CLWGT\n'
            'V=TVV*EXP(ETA(2))\n'
            'S1=V\n\n',
            True,
        ),
        (
            ('nonmem', 'pheno_real.mod'),
            [('CL', 'WGT', 'theta1 * (cov/median)**theta2', '*')],
            '$PK\n'
            'WGT_MEDIAN = 1.30000\n'
            'IF(AMT.GT.0) BTIME=TIME\n'
            'TAD=TIME-BTIME\n'
            'TVCL=THETA(1)*WGT\n'
            'TVV=THETA(2)*WGT\n'
            'IF(APGR.LT.5) TVV=TVV*(1+THETA(3))\n'
            'CL=TVCL*EXP(ETA(1))\n'
            'CLWGT = THETA(4)*(WGT/WGT_MEDIAN)**THETA(5)\n'
            'CL = CL*CLWGT\n'
            'V=TVV*EXP(ETA(2))\n'
            'S1=V\n\n',
            True,
        ),
        (
            ('nonmem', 'pheno_real.mod'),
            [('CL', 'WGT', '((cov/std) - median) * theta', '*')],
            '$PK\n'
            'WGT_MEDIAN = 1.30000\n'
            'WGT_STD = 0.704565\n'
            'IF(AMT.GT.0) BTIME=TIME\n'
            'TAD=TIME-BTIME\n'
            'TVCL=THETA(1)*WGT\n'
            'TVV=THETA(2)*WGT\n'
            'IF(APGR.LT.5) TVV=TVV*(1+THETA(3))\n'
            'CL=TVCL*EXP(ETA(1))\n'
            'CLWGT = THETA(4)*(WGT/WGT_STD - WGT_MEDIAN)\n'
            'CL = CL*CLWGT\n'
            'V=TVV*EXP(ETA(2))\n'
            'S1=V\n\n',
            True,
        ),
        (
            ('nonmem', 'pheno_real.mod'),
            [
                ('CL', 'WGT', 'exp', '+'),
                ('V', 'WGT', 'exp', '+'),
            ],
            '$PK\n'
            'WGT_MEDIAN = 1.30000\n'
            'IF(AMT.GT.0) BTIME=TIME\n'
            'TAD=TIME-BTIME\n'
            'TVCL=THETA(1)*WGT\n'
            'TVV=THETA(2)*WGT\n'
            'IF(APGR.LT.5) TVV=TVV*(1+THETA(3))\n'
            'CL=TVCL*EXP(ETA(1))\n'
            'CLWGT = EXP(THETA(4)*(WGT - WGT_MEDIAN))\n'
            'CL = CL + CLWGT\n'
            'V=TVV*EXP(ETA(2))\n'
            'VWGT = EXP(THETA(5)*(WGT - WGT_MEDIAN))\n'
            'V = V + VWGT\n'
            'S1=V\n\n',
            True,
        ),
        (
            ('nonmem', 'pheno_real.mod'),
            [
                ('CL', 'WGT', 'exp', '*'),
                ('V', 'WGT', 'exp', '*'),
            ],
            '$PK\n'
            'WGT_MEDIAN = 1.30000\n'
            'IF(AMT.GT.0) BTIME=TIME\n'
            'TAD=TIME-BTIME\n'
            'TVCL=THETA(1)*WGT\n'
            'TVV=THETA(2)*WGT\n'
            'IF(APGR.LT.5) TVV=TVV*(1+THETA(3))\n'
            'CL=TVCL*EXP(ETA(1))\n'
            'CLWGT = EXP(THETA(4)*(WGT - WGT_MEDIAN))\n'
            'CL = CL*CLWGT\n'
            'V=TVV*EXP(ETA(2))\n'
            'VWGT = EXP(THETA(5)*(WGT - WGT_MEDIAN))\n'
            'V = V*VWGT\n'
            'S1=V\n\n',
            True,
        ),
        (
            ('nonmem', 'models', 'mox2.mod'),
            [
                ('V', 'WT', 'exp', '*'),
                ('V', 'AGE', 'exp', '*'),
            ],
            '$PK\n'
            'AGE_MEDIAN = 66.0000\n'
            'WT_MEDIAN = 78.0000\n'
            'CL = THETA(1) * EXP(ETA(1))\n'
            'VC = THETA(2) * EXP(ETA(2))\n'
            'MAT = THETA(3) * EXP(ETA(3))\n'
            'KA = 1/MAT\n'
            'V = VC\n'
            'VWT = EXP(THETA(4)*(WT - WT_MEDIAN))\n'
            'VAGE = EXP(THETA(5)*(AGE - AGE_MEDIAN))\n'
            'V = V*VAGE*VWT\n',
            True,
        ),
        (
            ('nonmem', 'models', 'mox2.mod'),
            [
                ('V', 'WT', 'exp', '*'),
                ('V', 'AGE', 'exp', '*'),
                ('V', 'CLCR', 'exp', '*'),
            ],
            '$PK\n'
            'CLCR_MEDIAN = 65.0000\n'
            'AGE_MEDIAN = 66.0000\n'
            'WT_MEDIAN = 78.0000\n'
            'CL = THETA(1) * EXP(ETA(1))\n'
            'VC = THETA(2) * EXP(ETA(2))\n'
            'MAT = THETA(3) * EXP(ETA(3))\n'
            'KA = 1/MAT\n'
            'V = VC\n'
            'VWT = EXP(THETA(4)*(WT - WT_MEDIAN))\n'
            'VAGE = EXP(THETA(5)*(AGE - AGE_MEDIAN))\n'
            'VCLCR = EXP(THETA(6)*(CLCR - CLCR_MEDIAN))\n'
            'V = V*VAGE*VCLCR*VWT\n',
            True,
        ),
        (
            ('nonmem', 'models', 'mox2.mod'),
            [
                ('CL', 'CLCR', 'exp', '*'),
                ('V', 'CLCR', 'exp', '*'),
            ],
            '$PK\n'
            'CLCR_MEDIAN = 65.0000\n'
            'CL = THETA(1) * EXP(ETA(1))\n'
            'CLCLCR = EXP(THETA(4)*(CLCR - CLCR_MEDIAN))\n'
            'CL = CL*CLCLCR\n'
            'VC = THETA(2) * EXP(ETA(2))\n'
            'MAT = THETA(3) * EXP(ETA(3))\n'
            'KA = 1/MAT\n'
            'V = VC\n'
            'VCLCR = EXP(THETA(5)*(CLCR - CLCR_MEDIAN))\n'
            'V = V*VCLCR\n',
            True,
        ),
        (
            ('nonmem', 'models', 'mox2.mod'),
            [
                ('V', 'CLCR', 'exp', '*'),
                ('CL', 'CLCR', 'exp', '*'),
            ],
            '$PK\n'
            'CLCR_MEDIAN = 65.0000\n'
            'CL = THETA(1) * EXP(ETA(1))\n'
            'CLCLCR = EXP(THETA(5)*(CLCR - CLCR_MEDIAN))\n'
            'CL = CL*CLCLCR\n'
            'VC = THETA(2) * EXP(ETA(2))\n'
            'MAT = THETA(3) * EXP(ETA(3))\n'
            'KA = 1/MAT\n'
            'V = VC\n'
            'VCLCR = EXP(THETA(4)*(CLCR - CLCR_MEDIAN))\n'
            'V = V*VCLCR\n',
            True,
        ),
    ],
    ids=repr,
)
def test_add_covariate_effect(
    load_model_for_test, testdata, model_path, effects, expected, allow_nested
):
    model = load_model_for_test(testdata.joinpath(*model_path))
    error_record_before = ''.join(map(str, model.internals.control_stream.get_records('ERROR')))

    for effect in effects:
        add_covariate_effect(model, *effect, allow_nested=allow_nested)

    model.update_source()
    error_record_after = ''.join(map(str, model.internals.control_stream.get_records('ERROR')))

    assert str(model.internals.control_stream.get_pred_pk_record()) == expected
    assert error_record_after == error_record_before

    for effect in effects:
        assert f'POP_{effect[0]}{effect[1]}' in model.model_code
