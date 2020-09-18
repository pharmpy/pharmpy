import re

import numpy as np
import pytest
from pyfakefs.fake_filesystem_unittest import Patcher

from pharmpy import Model
from pharmpy.modeling import (absorption_rate, add_covariate_effect, boxcox, explicit_odes,
                              john_draper, tdist)


@pytest.mark.parametrize('effect, covariate, operation, buf_new', [
    ('exp', 'WGT', '*', 'WGT_MEDIAN = 1.30000\n'
                        'CLWGT = EXP((WGT - WGT_MEDIAN)*THETA(4))\n'
                        'CL = CLWGT*TVCL*EXP(ETA(1))'),
    ('exp', 'WGT', '+', 'WGT_MEDIAN = 1.30000\n'
                        'CLWGT = EXP((WGT - WGT_MEDIAN)*THETA(4))\n'
                        'CL = CLWGT + TVCL*EXP(ETA(1))'),
    ('pow', 'WGT', '*', 'WGT_MEDIAN = 1.30000\n'
                        'CLWGT = (WGT/WGT_MEDIAN)**THETA(4)\n'
                        'CL = CLWGT*TVCL*EXP(ETA(1))'),
    ('lin', 'WGT', '*', 'WGT_MEDIAN = 1.30000\n'
                        'CLWGT = (WGT - WGT_MEDIAN)*THETA(4) + 1\n'
                        'CL = CLWGT*TVCL*EXP(ETA(1))'),
    ('cat', 'FA1', '*', 'IF (FA1.EQ.1.0) THEN\n'
                        'CLFA1 = 1\n'
                        'ELSE IF (FA1.EQ.0.0) THEN\n'
                        'CLFA1 = THETA(4) + 1\n'
                        'END IF\n'
                        'CL = CLFA1*TVCL*EXP(ETA(1))'),
    ('piece_lin', 'WGT', '*', 'WGT_MEDIAN = 1.30000\n'
                              'IF (WGT.LE.WGT_MEDIAN) THEN\n'
                              'CLWGT = (WGT - WGT_MEDIAN)*THETA(4) + 1\n'
                              'ELSE\n'
                              'CLWGT = (WGT - WGT_MEDIAN)*THETA(5) + 1\n'
                              'END IF\n'
                              'CL = CLWGT*TVCL*EXP(ETA(1))'),
    ('theta - cov + median', 'WGT', '*',
     'WGT_MEDIAN = 1.30000\n'
     'CLWGT = -WGT + WGT_MEDIAN + THETA(4)\n'
     'CL = CLWGT*TVCL*EXP(ETA(1))'),
    ('theta - cov + std', 'WGT', '*',
     'WGT_STD = 0.704565\n'
     'CLWGT = -WGT + WGT_STD + THETA(4)\n'
     'CL = CLWGT*TVCL*EXP(ETA(1))')
])
def test_add_covariate_effect(pheno_path, effect, covariate, operation, buf_new):
    model = Model(pheno_path)

    add_covariate_effect(model, 'CL', covariate, effect, operation)
    model.update_source()

    rec_ref = f'$PK\n' \
              f'IF(AMT.GT.0) BTIME=TIME\n' \
              f'TAD=TIME-BTIME\n' \
              f'TVCL=THETA(1)*WGT\n' \
              f'TVV=THETA(2)*WGT\n' \
              f'IF(APGR.LT.5) TVV=TVV*(1+THETA(3))\n' \
              f'CL=TVCL*EXP(ETA(1))\n' \
              f'{buf_new}\n' \
              f'V=TVV*EXP(ETA(2))\n' \
              f'S1=V\n\n'

    assert str(model.get_pred_pk_record()) == rec_ref


def test_add_covariate_effect_nan(pheno_path):
    model = Model(pheno_path)
    data = model.dataset

    new_col = [np.nan] * 10 + ([1.0] * (len(data.index) - 10))

    data['new_col'] = new_col
    model.dataset = data

    add_covariate_effect(model, 'CL', 'new_col', 'cat')
    model.update_source(nofiles=True)

    assert not re.search('NaN', str(model))
    assert re.search(r'new_col\.EQ\.-99', str(model))


def test_to_explicit_odes(pheno_path, testdata):
    model = Model(pheno_path)

    explicit_odes(model)
    model.update_source()
    lines = str(model).split('\n')
    print(lines)
    assert lines[5] == '$MODEL TOL=3 COMPARTMENT=(CENTRAL DEFDOSE)'
    assert lines[16] == '$DES'
    assert lines[17] == 'DADT(1) = -A(1)*CL/V'

    model = Model(testdata / 'nonmem' / 'modeling' / 'pheno_advan1_zero_order.mod')
    explicit_odes(model)
    model.update_source()
    lines = str(model).split('\n')
    assert lines[15] == 'D1 = THETA(4)'


def test_absorption_rate(testdata):
    model = Model(testdata / 'nonmem' / 'modeling' / 'pheno_advan1.mod')
    advan1_before = str(model)
    absorption_rate(model, 'instant')
    assert advan1_before == str(model)

    model = Model(testdata / 'nonmem' / 'modeling' / 'pheno_advan2.mod')
    absorption_rate(model, 'instant')
    model.update_source()
    assert str(model) == advan1_before

    model = Model(testdata / 'nonmem' / 'modeling' / 'pheno_advan3.mod')
    advan3_before = str(model)
    absorption_rate(model, 'instant')
    model.update_source()
    assert str(model) == advan3_before

    model = Model(testdata / 'nonmem' / 'modeling' / 'pheno_advan4.mod')
    absorption_rate(model, 'instant')
    model.update_source()
    assert str(model) == advan3_before

    model = Model(testdata / 'nonmem' / 'modeling' / 'pheno_advan11.mod')
    advan11_before = str(model)
    absorption_rate(model, 'instant')
    model.update_source()
    assert str(model) == advan11_before

    model = Model(testdata / 'nonmem' / 'modeling' / 'pheno_advan12.mod')
    absorption_rate(model, 'instant')
    model.update_source()
    assert str(model) == advan11_before

    model = Model(testdata / 'nonmem' / 'modeling' / 'pheno_advan5_nodepot.mod')
    advan5_nodepot_before = str(model)
    absorption_rate(model, 'instant')
    model.update_source()
    assert str(model) == advan5_nodepot_before

    model = Model(testdata / 'nonmem' / 'modeling' / 'pheno_advan5_depot.mod')
    absorption_rate(model, 'instant')
    model.update_source()
    assert str(model) == advan5_nodepot_before

    # 0-order to 0-order
    model = Model(testdata / 'nonmem' / 'modeling' / 'pheno_advan1_zero_order.mod')
    advan1_zero_order_before = str(model)
    absorption_rate(model, 'ZO')
    model.update_source()
    assert str(model) == advan1_zero_order_before

    # 0-order to Bolus
    model = Model(testdata / 'nonmem' / 'modeling' / 'pheno_advan1_zero_order.mod')
    absorption_rate(model, 'instant')
    model.update_source(nofiles=True)
    assert str(model).split('\n')[2:] == advan1_before.split('\n')[2:]

    # 1st order to 1st order
    model = Model(testdata / 'nonmem' / 'modeling' / 'pheno_advan2.mod')
    advan2_before = str(model)
    absorption_rate(model, 'FO')
    model.update_source(nofiles=True)
    assert str(model) == advan2_before

    # 0-order to 1st order
    model = Model(testdata / 'nonmem' / 'modeling' / 'pheno_advan1_zero_order.mod')
    absorption_rate(model, 'FO')
    model.update_source(nofiles=True)
    correct = '''$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno_zero_order.csv IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN2 TRANS2

$PK
MAT = THETA(4)
IF(AMT.GT.0) BTIME=TIME
TAD=TIME-BTIME
TVCL=THETA(1)*WGT
TVV=THETA(2)*WGT
IF(APGR.LT.5) TVV=TVV*(1+THETA(3))
CL=TVCL*EXP(ETA(1))
V=TVV*EXP(ETA(2))
S2 = V
KA = 1/MAT

$ERROR
W=F
Y=F+W*EPS(1)
IPRED=F
IRES=DV-IPRED
IWRES=IRES/W

$THETA (0,0.00469307) ; CL
$THETA (0,1.00916) ; V
$THETA (-.99,.1)
$THETA  (0,0.1) ; TVMAT
$OMEGA DIAGONAL(2)
 0.0309626  ;       IVCL
 0.031128  ;        IVV

$SIGMA 1e-7
$ESTIMATION METHOD=1 INTERACTION
$COVARIANCE UNCONDITIONAL
$TABLE ID TIME DV AMT WGT APGR IPRED PRED RES TAD CWRES NPDE NOAPPEND
       NOPRINT ONEHEADER FILE=sdtab1
'''
    assert str(model) == correct

    # Bolus to 1st order
    model = Model(testdata / 'nonmem' / 'modeling' / 'pheno_advan1.mod')
    absorption_rate(model, 'FO')
    model.update_source(nofiles=True)
    assert str(model).split('\n')[2:] == correct.split('\n')[2:]

    # Bolus to 0-order
    with Patcher(additional_skip_names=['pkgutil']) as patcher:
        fs = patcher.fs
        datadir = testdata / 'nonmem' / 'modeling'
        fs.add_real_file(datadir / 'pheno_advan1.mod', target_path='dir/pheno_advan1.mod')
        fs.add_real_file(datadir / 'pheno_advan2.mod', target_path='dir/pheno_advan2.mod')
        fs.add_real_file(datadir.parent / 'pheno.dta', target_path='pheno.dta')
        model = Model('dir/pheno_advan1.mod')
        absorption_rate(model, 'ZO')
        model.update_source()
        correct = '''$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.csv IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2 RATE
$SUBROUTINE ADVAN1 TRANS2

$PK
MAT = THETA(4)
IF(AMT.GT.0) BTIME=TIME
TAD=TIME-BTIME
TVCL=THETA(1)*WGT
TVV=THETA(2)*WGT
IF(APGR.LT.5) TVV=TVV*(1+THETA(3))
CL=TVCL*EXP(ETA(1))
V=TVV*EXP(ETA(2))
S1=V
D1 = 2*MAT

$ERROR
W=F
Y=F+W*EPS(1)
IPRED=F
IRES=DV-IPRED
IWRES=IRES/W

$THETA (0,0.00469307) ; CL
$THETA (0,1.00916) ; V
$THETA (-.99,.1)
$THETA  (0,0.1) ; TVMAT
$OMEGA DIAGONAL(2)
 0.0309626  ;       IVCL
 0.031128  ;        IVV

$SIGMA 1e-7
$ESTIMATION METHOD=1 INTERACTION
$COVARIANCE UNCONDITIONAL
$TABLE ID TIME DV AMT WGT APGR IPRED PRED RES TAD CWRES NPDE NOAPPEND
       NOPRINT ONEHEADER FILE=sdtab1
'''
        assert str(model) == correct

        # 1st to 0-order
        model = Model('dir/pheno_advan2.mod')
        absorption_rate(model, 'ZO')
        model.update_source(force=True)
        assert str(model) == correct


@pytest.mark.parametrize('etas, etab, buf_new', [
    (['ETA(1)'], 'ETAB1 = (EXP(ETA(1))**THETA(4) - 1)/THETA(4)',
     'CL = TVCL*EXP(ETAB1)\nV=TVV*EXP(ETA(2))'),
    (['ETA(1)', 'ETA(2)'], 'ETAB1 = (EXP(ETA(1))**THETA(4) - 1)/THETA(4)\n'
                           'ETAB2 = (EXP(ETA(2))**THETA(5) - 1)/THETA(5)',
     'CL = TVCL*EXP(ETAB1)\nV = TVV*EXP(ETAB2)'),
    (None, 'ETAB1 = (EXP(ETA(1))**THETA(4) - 1)/THETA(4)\n'
           'ETAB2 = (EXP(ETA(2))**THETA(5) - 1)/THETA(5)',
     'CL = TVCL*EXP(ETAB1)\nV = TVV*EXP(ETAB2)'),
    (['eta(1)'], 'ETAB1 = (EXP(ETA(1))**THETA(4) - 1)/THETA(4)',
     'CL = TVCL*EXP(ETAB1)\nV=TVV*EXP(ETA(2))')
])
def test_boxcox(pheno_path, etas, etab, buf_new):
    model = Model(pheno_path)

    boxcox(model, etas)
    model.update_source()

    rec_ref = f'$PK\n' \
              f'{etab}\n' \
              f'IF(AMT.GT.0) BTIME=TIME\n' \
              f'TAD=TIME-BTIME\n' \
              f'TVCL=THETA(1)*WGT\n' \
              f'TVV=THETA(2)*WGT\n' \
              f'IF(APGR.LT.5) TVV=TVV*(1+THETA(3))\n' \
              f'{buf_new}\n' \
              f'S1=V\n\n'

    assert str(model.get_pred_pk_record()) == rec_ref
    assert model.parameters['lambda1'].init == 0.01


def test_tdist(pheno_path):
    model = Model(pheno_path)

    tdist(model, ['ETA(1)'])
    model.update_source()

    symbol = 'ETAT1'

    eta = 'ETA(1)'
    theta = 'THETA(4)'

    num_1 = f'{eta}**2 + 1'
    denom_1 = f'4*{theta}'

    num_2 = f'5*{eta}**4 + 16*{eta}**2 + 3'
    denom_2 = f'96*{theta}**2'

    num_3 = f'3*{eta}**6 + 19*{eta}**4 + 17*{eta}**2 - 15'
    denom_3 = f'384*{theta}**3'

    expression = f'(({num_1})/({denom_1}) + ({num_2})/({denom_2}) + ' \
                 f'({num_3})/({denom_3}) + 1)*ETA(1)'

    rec_ref = f'$PK\n' \
              f'{symbol} = {expression}\n' \
              f'IF(AMT.GT.0) BTIME=TIME\n' \
              f'TAD=TIME-BTIME\n' \
              f'TVCL=THETA(1)*WGT\n' \
              f'TVV=THETA(2)*WGT\n' \
              f'IF(APGR.LT.5) TVV=TVV*(1+THETA(3))\n' \
              f'CL = TVCL*EXP(ETAT1)\n' \
              f'V=TVV*EXP(ETA(2))\n' \
              f'S1=V\n\n'

    assert str(model.get_pred_pk_record()) == rec_ref
    assert model.parameters['df1'].init == 80


@pytest.mark.parametrize('etas, etad, buf_new', [
    (['ETA(1)'], 'ETAD1 = ((ABS(ETA(1)) + 1)**THETA(4) - 1)*ABS(ETA(1))/(ETA(1)*THETA(4))',
     'CL = TVCL*EXP(ETAD1)\nV=TVV*EXP(ETA(2))'),
])
def test_john_draper(pheno_path, etas, etad, buf_new):
    model = Model(pheno_path)

    john_draper(model, etas)
    model.update_source()

    rec_ref = f'$PK\n' \
              f'{etad}\n' \
              f'IF(AMT.GT.0) BTIME=TIME\n' \
              f'TAD=TIME-BTIME\n' \
              f'TVCL=THETA(1)*WGT\n' \
              f'TVV=THETA(2)*WGT\n' \
              f'IF(APGR.LT.5) TVV=TVV*(1+THETA(3))\n' \
              f'{buf_new}\n' \
              f'S1=V\n\n'

    assert str(model.get_pred_pk_record()) == rec_ref
