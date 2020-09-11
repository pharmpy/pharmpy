import re

import numpy as np
import pytest

from pharmpy import Model
from pharmpy.modeling import absorption, add_covariate_effect, boxcox, explicit_odes


@pytest.mark.parametrize('effect, covariate, operation, buf_new', [
    ('exp', 'WGT', '*', 'CLWGT = EXP((-CL_MEDIAN + WGT)*THETA(4))\n'
                        'CL_MEDIAN = 1.30000\n'
                        'CL = CLWGT*TVCL*EXP(ETA(1))\n'),
    ('exp', 'WGT', '+', 'CLWGT = EXP((-CL_MEDIAN + WGT)*THETA(4))\n'
                        'CL_MEDIAN = 1.30000\n'
                        'CL = CLWGT + TVCL*EXP(ETA(1))\n'),
    ('pow', 'WGT', '*', 'CLWGT = (WGT/CL_MEDIAN)**THETA(4)\n'
                        'CL_MEDIAN = 1.30000\n'
                        'CL = CLWGT*TVCL*EXP(ETA(1))\n'),
    ('lin_cont', 'WGT', '*', 'CLWGT = (-CL_MEDIAN + WGT)*THETA(4) + 1\n'
                             'CL_MEDIAN = 1.30000\n'
                             'CL = CLWGT*TVCL*EXP(ETA(1))\n'),
    ('lin_cat', 'FA1', '*', 'IF (FA1.EQ.1.0) THEN\n'
                            'CLFA1 = 1\n'
                            'ELSE IF (FA1.EQ.0.0) THEN\n'
                            'CLFA1 = THETA(4) + 1\n'
                            'END IF\n'
                            'CL_MEDIAN = 1.00000\n'
                            'CL = CLFA1*TVCL*EXP(ETA(1))\n'),
    ('piece_lin', 'WGT', '*', 'IF (CL_MEDIAN.GE.WGT) THEN\n'
                              'CLWGT = (-CL_MEDIAN + WGT)*THETA(4) + 1\n'
                              'ELSE\n'
                              'CLWGT = (-CL_MEDIAN + WGT)*THETA(5) + 1\n'
                              'END IF\n'
                              'CL_MEDIAN = 1.30000\n'
                              'CL = CLWGT*TVCL*EXP(ETA(1))\n'),
    ('theta - cov + median', 'WGT', '*',
     'CLWGT = CL_MEDIAN - WGT + THETA(4)\n'
     'CL_MEDIAN = 1.30000\n'
     'CL = CLWGT*TVCL*EXP(ETA(1))\n')
])
def test_add_covariate_effect(pheno_path, effect, covariate, operation, buf_new):
    model = Model(pheno_path)

    add_covariate_effect(model, 'CL', covariate, effect, operation)
    model.update_source()

    rec_ref = f'$PK\n\n\n' \
              f'IF(AMT.GT.0) BTIME=TIME\n' \
              f'TAD=TIME-BTIME\n' \
              f'      TVCL=THETA(1)*WGT\n' \
              f'      TVV=THETA(2)*WGT\n' \
              f'IF(APGR.LT.5) TVV=TVV*(1+THETA(3))\n' \
              f'      CL=TVCL*EXP(ETA(1))\n' \
              f'{buf_new}' \
              f'      V=TVV*EXP(ETA(2))\n' \
              f'      S1=V\n'

    assert str(model.get_pred_pk_record()) == rec_ref


def test_add_covariate_effect_nan(pheno_path):
    model = Model(pheno_path)
    data = model.dataset

    new_col = [np.nan] * 10 + ([1.0] * (len(data.index) - 10))

    data['new_col'] = new_col
    model.dataset = data

    add_covariate_effect(model, 'CL', 'new_col', 'lin_cat')
    model.update_source(nofiles=True)

    assert not re.search('NaN', str(model))
    assert re.search(r'new_col\.EQ\.-99', str(model))


def test_to_explicit_odes(pheno_path):
    model = Model(pheno_path)

    explicit_odes(model)
    model.update_source()
    lines = str(model).split('\n')
    assert lines[6] == '$MODEL TOL=3 COMPARTMENT=(CENTRAL DEFDOSE)'
    assert lines[18] == '$DES'
    assert lines[19] == 'DADT(1) = -A(1)*CL/V'


def test_absorption(testdata):
    model = Model(testdata / 'nonmem' / 'modeling' / 'pheno_advan1.mod')
    advan1_before = str(model)
    absorption(model, 0)
    assert advan1_before == str(model)

    model = Model(testdata / 'nonmem' / 'modeling' / 'pheno_advan2.mod')
    absorption(model, 0)
    model.update_source()
    assert str(model) == advan1_before

    model = Model(testdata / 'nonmem' / 'modeling' / 'pheno_advan3.mod')
    advan3_before = str(model)
    absorption(model, 0)
    model.update_source()
    assert str(model) == advan3_before

    model = Model(testdata / 'nonmem' / 'modeling' / 'pheno_advan4.mod')
    absorption(model, 0)
    model.update_source()
    assert str(model) == advan3_before

    model = Model(testdata / 'nonmem' / 'modeling' / 'pheno_advan11.mod')
    print(str(model))
    advan11_before = str(model)
    absorption(model, 0)
    model.update_source()
    assert str(model) == advan11_before

    model = Model(testdata / 'nonmem' / 'modeling' / 'pheno_advan12.mod')
    absorption(model, 0)
    model.update_source()
    print(str(model))
    assert str(model) == advan11_before


@pytest.mark.parametrize('etas, etab, buf_new', [
    (['ETA(1)'], 'ETAB1 = (EXP(ETA(1))**THETA(4) - 1)/THETA(4)',
     'CL = TVCL*EXP(ETAB1)'),
])
def test_boxcox(pheno_path, etas, etab, buf_new):
    model = Model(pheno_path)

    boxcox(model, etas)
    model.update_source()

    rec_ref = f'$PK\n\n\n' \
              f'{etab}\n' \
              f'IF(AMT.GT.0) BTIME=TIME\n' \
              f'TAD=TIME-BTIME\n' \
              f'      TVCL=THETA(1)*WGT\n' \
              f'      TVV=THETA(2)*WGT\n' \
              f'IF(APGR.LT.5) TVV=TVV*(1+THETA(3))\n' \
              f'{buf_new}\n' \
              f'      V=TVV*EXP(ETA(2))\n' \
              f'      S1=V\n'

    print(str(model.get_pred_pk_record()))
    print(rec_ref)
    assert str(model.get_pred_pk_record()) == rec_ref
