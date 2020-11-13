import re

import numpy as np
import pytest
from pyfakefs.fake_filesystem_unittest import Patcher

from pharmpy import Model
from pharmpy.modeling import (
    absorption_rate,
    add_covariate_effect,
    add_etas,
    add_lag_time,
    boxcox,
    create_rv_block,
    explicit_odes,
    iiv_on_ruv,
    john_draper,
    michaelis_menten_elimination,
    remove_lag_time,
    set_transit_compartments,
    tdist,
)
from pharmpy.plugins.nonmem.nmtran_parser import NMTranParser


def test_michaelis_menten_elimination(testdata):
    model = Model(testdata / 'nonmem' / 'pheno.mod')
    michaelis_menten_elimination(model)
    model.update_source()
    correct = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV
$SUBROUTINE ADVAN6 TOL=3

$MODEL COMPARTMENT=(CENTRAL DEFDOSE)
$PK
CLMM = THETA(4)
KM = THETA(3)
CL=THETA(1)*EXP(ETA(1))
V=THETA(2)*EXP(ETA(2))
S1=V

$DES
DADT(1) = -A(1)*CLMM/(A(1) + KM)
$ERROR
Y=F+F*EPS(1)

$THETA (0,0.00469307) ; TVCL
$THETA (0,1.00916) ; TVV
$THETA  (0,0.1) ; POP_KM
$THETA  (0,0.1) ; POP_CLMM
$OMEGA 0.0309626  ; IVCL
$OMEGA 0.031128  ; IVV
$SIGMA 0.013241

$ESTIMATION METHOD=1 INTERACTION
"""
    assert str(model) == correct


def test_transit_compartments(testdata):
    model = Model(testdata / 'nonmem' / 'modeling' / 'pheno_advan1.mod')
    set_transit_compartments(model, 0)
    transits = model.statements.ode_system.find_transit_compartments(model.statements)
    assert len(transits) == 0
    model = Model(testdata / 'nonmem' / 'modeling' / 'pheno_2transits.mod')
    set_transit_compartments(model, 1)
    transits = model.statements.ode_system.find_transit_compartments(model.statements)
    assert len(transits) == 1
    model.update_source()
    correct = '''$PROBLEM PHENOBARB SIMPLE MODEL
$DATA ../pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN5 TRANS1
$MODEL COMP=(TRANS2 DEFDOSE) COMP=(DEPOT) COMP=(CENTRAL) COMP=(PERIPHERAL)

$PK
IF(AMT.GT.0) BTIME=TIME
TAD=TIME-BTIME
TVCL=THETA(1)*WGT
TVV=THETA(2)*WGT
IF(APGR.LT.5) TVV=TVV*(1+THETA(3))
CL=TVCL*EXP(ETA(1))
V=TVV*EXP(ETA(2))
K23 = THETA(6)
K30 = CL/V
K34 = THETA(4)
K43 = THETA(5)
K12 = THETA(7)
S3 = V

$ERROR
W=F
Y=F+W*EPS(1)
IPRED=F
IRES=DV-IPRED
IWRES=IRES/W

$THETA (0,0.00469307) ; CL
$THETA (0,1.00916) ; V
$THETA (-.99,.1)
$THETA (0,10)
$THETA (0,10)
$THETA (1,10)
$THETA (1,10)
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
    model = Model(testdata / 'nonmem' / 'modeling' / 'pheno_2transits.mod')
    set_transit_compartments(model, 4)
    transits = model.statements.ode_system.find_transit_compartments(model.statements)
    assert len(transits) == 4
    model.update_source()
    correct = (
        '''$PROBLEM PHENOBARB SIMPLE MODEL
$DATA ../pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN5 TRANS1
$MODEL COMPARTMENT=(TRANSIT3 DEFDOSE) COMPARTMENT=(TRANSIT4) COMP=(TRANS1) COMP=(TRANS2) '''
        + '''COMP=(DEPOT) COMP=(CENTRAL) COMP=(PERIPHERAL)

$PK
IF(AMT.GT.0) BTIME=TIME
TAD=TIME-BTIME
TVCL=THETA(1)*WGT
TVV=THETA(2)*WGT
IF(APGR.LT.5) TVV=TVV*(1+THETA(3))
CL=TVCL*EXP(ETA(1))
V=TVV*EXP(ETA(2))
K56 = THETA(6)
K60 = CL/V
K67 = THETA(4)
K76 = THETA(5)
K34 = THETA(7)
K45 = THETA(7)
S6 = V
K23 = K34
K12 = K34

$ERROR
W=F
Y=F+W*EPS(1)
IPRED=F
IRES=DV-IPRED
IWRES=IRES/W

$THETA (0,0.00469307) ; CL
$THETA (0,1.00916) ; V
$THETA (-.99,.1)
$THETA (0,10)
$THETA (0,10)
$THETA (1,10)
$THETA (1,10)
$OMEGA DIAGONAL(2)
 0.0309626  ;       IVCL
 0.031128  ;        IVV

$SIGMA 1e-7
$ESTIMATION METHOD=1 INTERACTION
$COVARIANCE UNCONDITIONAL
$TABLE ID TIME DV AMT WGT APGR IPRED PRED RES TAD CWRES NPDE NOAPPEND
       NOPRINT ONEHEADER FILE=sdtab1
'''
    )
    assert str(model) == correct


def test_transit_compartments_added_mdt(testdata):
    model = Model(testdata / 'nonmem' / 'modeling' / 'pheno_advan5_nodepot.mod')
    set_transit_compartments(model, 2)
    transits = model.statements.ode_system.find_transit_compartments(model.statements)
    assert len(transits) == 2
    model.update_source()
    correct = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA ../pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN5 TRANS1
$MODEL COMPARTMENT=(TRANSIT1 DEFDOSE) COMPARTMENT=(TRANSIT2) COMP=(CENTRAL) COMP=(PERIPHERAL)

$PK
MDT = THETA(6)
IF(AMT.GT.0) BTIME=TIME
TAD=TIME-BTIME
TVCL=THETA(1)*WGT
TVV=THETA(2)*WGT
IF(APGR.LT.5) TVV=TVV*(1+THETA(3))
CL=TVCL*EXP(ETA(1))
V=TVV*EXP(ETA(2))
K30 = CL/V
K34 = THETA(4)
K43 = THETA(5)
S3 = V
K23 = 2/MDT
K12 = 2/MDT

$ERROR
W=F
Y=F+W*EPS(1)
IPRED=F
IRES=DV-IPRED
IWRES=IRES/W

$THETA (0,0.00469307) ; CL
$THETA (0,1.00916) ; V
$THETA (-.99,.1)
$THETA (0,10)
$THETA (0,10)
$THETA  (0,0.1) ; POP_MDT
$OMEGA DIAGONAL(2)
 0.0309626  ;       IVCL
 0.031128  ;        IVV

$SIGMA 1e-7
$ESTIMATION METHOD=1 INTERACTION
$COVARIANCE UNCONDITIONAL
$TABLE ID TIME DV AMT WGT APGR IPRED PRED RES TAD CWRES NPDE NOAPPEND
       NOPRINT ONEHEADER FILE=sdtab1
"""
    assert str(model) == correct


def test_transit_compartments_change_advan(testdata):
    model = Model(testdata / 'nonmem' / 'modeling' / 'pheno_advan1.mod')
    set_transit_compartments(model, 3)
    transits = model.statements.ode_system.find_transit_compartments(model.statements)
    assert len(transits) == 3
    model.update_source()
    correct = (
        """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA ../pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN5 TRANS1

$MODEL COMPARTMENT=(TRANSIT1 DEFDOSE) COMPARTMENT=(TRANSIT2) COMPARTMENT=(TRANSIT3) """
        + """COMPARTMENT=(CENTRAL) COMPARTMENT=(OUTPUT)
$PK
MDT = THETA(4)
IF(AMT.GT.0) BTIME=TIME
TAD=TIME-BTIME
TVCL=THETA(1)*WGT
TVV=THETA(2)*WGT
IF(APGR.LT.5) TVV=TVV*(1+THETA(3))
CL=TVCL*EXP(ETA(1))
V=TVV*EXP(ETA(2))
S4 = V
K40 = CL/V
K34 = 3/MDT
K23 = 3/MDT
K12 = 3/MDT

$ERROR
W=F
Y=F+W*EPS(1)
IPRED=F
IRES=DV-IPRED
IWRES=IRES/W

$THETA (0,0.00469307) ; CL
$THETA (0,1.00916) ; V
$THETA (-.99,.1)
$THETA  (0,0.1) ; POP_MDT
$OMEGA DIAGONAL(2)
 0.0309626  ;       IVCL
 0.031128  ;        IVV

$SIGMA 1e-7
$ESTIMATION METHOD=1 INTERACTION
$COVARIANCE UNCONDITIONAL
$TABLE ID TIME DV AMT WGT APGR IPRED PRED RES TAD CWRES NPDE NOAPPEND
       NOPRINT ONEHEADER FILE=sdtab1
"""
    )
    assert str(model) == correct


def test_lag_time(testdata):
    model = Model(testdata / 'nonmem' / 'modeling' / 'pheno_advan1.mod')
    before = str(model)
    add_lag_time(model)
    model.update_source()
    correct = '''$PROBLEM PHENOBARB SIMPLE MODEL
$DATA ../pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN1 TRANS2

$PK
MDT = THETA(4)
IF(AMT.GT.0) BTIME=TIME
TAD=TIME-BTIME
TVCL=THETA(1)*WGT
TVV=THETA(2)*WGT
IF(APGR.LT.5) TVV=TVV*(1+THETA(3))
CL=TVCL*EXP(ETA(1))
V=TVV*EXP(ETA(2))
S1=V
ALAG1 = MDT

$ERROR
W=F
Y=F+W*EPS(1)
IPRED=F
IRES=DV-IPRED
IWRES=IRES/W

$THETA (0,0.00469307) ; CL
$THETA (0,1.00916) ; V
$THETA (-.99,.1)
$THETA  (0,0.1) ; TVMDT
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

    remove_lag_time(model)
    model.update_source()
    assert str(model) == before


@pytest.mark.parametrize(
    'effect, covariate, operation, buf_new',
    [
        (
            'exp',
            'WGT',
            '*',
            'WGT_MEDIAN = 1.30000\n' 'CLWGT = EXP(THETA(4)*(WGT - WGT_MEDIAN))\n' 'CL = CL*CLWGT',
        ),
        (
            'exp',
            'WGT',
            '+',
            'WGT_MEDIAN = 1.30000\n' 'CLWGT = EXP(THETA(4)*(WGT - WGT_MEDIAN))\n' 'CL = CL + CLWGT',
        ),
        (
            'pow',
            'WGT',
            '*',
            'WGT_MEDIAN = 1.30000\n' 'CLWGT = (WGT/WGT_MEDIAN)**THETA(4)\n' 'CL = CL*CLWGT',
        ),
        (
            'lin',
            'WGT',
            '*',
            'WGT_MEDIAN = 1.30000\n' 'CLWGT = THETA(4)*(WGT - WGT_MEDIAN) + 1\n' 'CL = CL*CLWGT',
        ),
        (
            'cat',
            'FA1',
            '*',
            'IF (FA1.EQ.1.0) THEN\n'
            'CLFA1 = 1\n'
            'ELSE IF (FA1.EQ.0.0) THEN\n'
            'CLFA1 = THETA(4) + 1\n'
            'END IF\n'
            'CL = CL*CLFA1',
        ),
        (
            'piece_lin',
            'WGT',
            '*',
            'WGT_MEDIAN = 1.30000\n'
            'IF (WGT.LE.WGT_MEDIAN) THEN\n'
            'CLWGT = THETA(4)*(WGT - WGT_MEDIAN) + 1\n'
            'ELSE\n'
            'CLWGT = THETA(5)*(WGT - WGT_MEDIAN) + 1\n'
            'END IF\n'
            'CL = CL*CLWGT',
        ),
        (
            'theta - cov + median',
            'WGT',
            '*',
            'WGT_MEDIAN = 1.30000\n' 'CLWGT = THETA(4) - WGT + WGT_MEDIAN\n' 'CL = CL*CLWGT',
        ),
        (
            'theta - cov + std',
            'WGT',
            '*',
            'WGT_STD = 0.704565\n' 'CLWGT = THETA(4) - WGT + WGT_STD\n' 'CL = CL*CLWGT',
        ),
        (
            'theta1 * (cov/median)**theta2',
            'WGT',
            '*',
            'WGT_MEDIAN = 1.30000\n'
            'CLWGT = THETA(4)*(WGT/WGT_MEDIAN)**THETA(5)\n'
            'CL = CL*CLWGT',
        ),
        (
            '((cov/std) - median) * theta',
            'WGT',
            '*',
            'WGT_MEDIAN = 1.30000\n'
            'WGT_STD = 0.704565\n'
            'CLWGT = THETA(4)*(WGT/WGT_STD - WGT_MEDIAN)\n'
            'CL = CL*CLWGT',
        ),
    ],
)
def test_add_covariate_effect(pheno_path, effect, covariate, operation, buf_new):
    model = Model(pheno_path)

    add_covariate_effect(model, 'CL', covariate, effect, operation)
    model.update_source()

    rec_ref = (
        f'$PK\n'
        f'IF(AMT.GT.0) BTIME=TIME\n'
        f'TAD=TIME-BTIME\n'
        f'TVCL=THETA(1)*WGT\n'
        f'TVV=THETA(2)*WGT\n'
        f'IF(APGR.LT.5) TVV=TVV*(1+THETA(3))\n'
        f'CL=TVCL*EXP(ETA(1))\n'
        f'{buf_new}\n'
        f'V=TVV*EXP(ETA(2))\n'
        f'S1=V\n\n'
    )

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


def test_add_covariate_effect_duplicates(pheno_path):
    model = Model(pheno_path)

    add_covariate_effect(model, 'CL', 'WGT', 'exp')

    with pytest.warns(UserWarning):
        add_covariate_effect(model, 'CL', 'WGT', 'exp')


def test_to_explicit_odes(pheno_path, testdata):
    model = Model(pheno_path)

    explicit_odes(model)
    model.update_source()
    lines = str(model).split('\n')
    assert lines[5] == '$MODEL COMPARTMENT=(CENTRAL DEFDOSE)'
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
    absorption_rate(model, 'bolus')
    assert advan1_before == str(model)

    model = Model(testdata / 'nonmem' / 'modeling' / 'pheno_advan2.mod')
    absorption_rate(model, 'bolus')
    model.update_source()
    assert str(model) == advan1_before

    model = Model(testdata / 'nonmem' / 'modeling' / 'pheno_advan3.mod')
    advan3_before = str(model)
    absorption_rate(model, 'bolus')
    model.update_source()
    assert str(model) == advan3_before

    model = Model(testdata / 'nonmem' / 'modeling' / 'pheno_advan4.mod')
    absorption_rate(model, 'bolus')
    model.update_source()
    assert str(model) == advan3_before

    model = Model(testdata / 'nonmem' / 'modeling' / 'pheno_advan11.mod')
    advan11_before = str(model)
    absorption_rate(model, 'bolus')
    model.update_source()
    assert str(model) == advan11_before

    model = Model(testdata / 'nonmem' / 'modeling' / 'pheno_advan12.mod')
    absorption_rate(model, 'bolus')
    model.update_source()
    assert str(model) == advan11_before

    model = Model(testdata / 'nonmem' / 'modeling' / 'pheno_advan5_nodepot.mod')
    advan5_nodepot_before = str(model)
    absorption_rate(model, 'bolus')
    model.update_source()
    assert str(model) == advan5_nodepot_before

    model = Model(testdata / 'nonmem' / 'modeling' / 'pheno_advan5_depot.mod')
    absorption_rate(model, 'bolus')
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
    absorption_rate(model, 'bolus')
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


def test_seq_to_FO(testdata):
    model = Model(testdata / 'nonmem' / 'modeling' / 'pheno_advan2_seq.mod')
    absorption_rate(model, 'FO')
    model.update_source(nofiles=True)
    correct = '''$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno_zero_order.csv IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN2 TRANS2

$PK
IF(AMT.GT.0) BTIME=TIME
TAD=TIME-BTIME
TVCL=THETA(1)*WGT
TVV=THETA(2)*WGT
IF(APGR.LT.5) TVV=TVV*(1+THETA(3))
CL=TVCL*EXP(ETA(1))
V=TVV*EXP(ETA(2))
S2=V
KA = THETA(4)

$ERROR
W=F
Y=F+W*EPS(1)
IPRED=F
IRES=DV-IPRED
IWRES=IRES/W

$THETA (0,0.00469307) ; CL
$THETA (0,1.00916) ; V
$THETA (-.99,.1)
$THETA (0,0.1)
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


def test_seq_to_ZO(testdata):
    model = Model(testdata / 'nonmem' / 'modeling' / 'pheno_advan2_seq.mod')
    absorption_rate(model, 'ZO')
    model.update_source(nofiles=True)
    correct = '''$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno_zero_order.csv IGNORE=@
$INPUT ID TIME AMT RATE WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN1 TRANS2

$PK
IF(AMT.GT.0) BTIME=TIME
TAD=TIME-BTIME
TVCL=THETA(1)*WGT
TVV=THETA(2)*WGT
IF(APGR.LT.5) TVV=TVV*(1+THETA(3))
CL=TVCL*EXP(ETA(1))
V=TVV*EXP(ETA(2))
S1 = V
D1 = THETA(4)

$ERROR
W=F
Y=F+W*EPS(1)
IPRED=F
IRES=DV-IPRED
IWRES=IRES/W

$THETA (0,0.00469307) ; CL
$THETA (0,1.00916) ; V
$THETA (-.99,.1)
$THETA (0,0.1)
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


def test_bolus_to_seq(testdata):
    model = Model(testdata / 'nonmem' / 'modeling' / 'pheno_advan1.mod')
    absorption_rate(model, 'seq-ZO-FO')
    model.update_source(nofiles=True)
    correct = '''$PROBLEM PHENOBARB SIMPLE MODEL
$DATA ../pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2 RATE
$SUBROUTINE ADVAN2 TRANS2

$PK
MDT = THETA(5)
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
D1 = 2*MDT

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
$THETA  (0,0.1) ; TVMDT
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


def test_ZO_to_seq(testdata):
    model = Model(testdata / 'nonmem' / 'modeling' / 'pheno_advan1_zero_order.mod')
    absorption_rate(model, 'seq-ZO-FO')
    model.update_source(nofiles=True)
    correct = '''$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno_zero_order.csv IGNORE=@
$INPUT ID TIME AMT RATE WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN2 TRANS2

$PK
MAT = THETA(5)
IF(AMT.GT.0) BTIME=TIME
TAD=TIME-BTIME
TVCL=THETA(1)*WGT
TVV=THETA(2)*WGT
IF(APGR.LT.5) TVV=TVV*(1+THETA(3))
CL=TVCL*EXP(ETA(1))
V=TVV*EXP(ETA(2))
S2 = V
D1 = THETA(4)
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
$THETA (0,0.1)
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


def test_FO_to_seq(testdata):
    model = Model(testdata / 'nonmem' / 'modeling' / 'pheno_advan2.mod')
    absorption_rate(model, 'seq-ZO-FO')
    model.update_source(nofiles=True)
    correct = '''$PROBLEM PHENOBARB SIMPLE MODEL
$DATA ../pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2 RATE
$SUBROUTINE ADVAN2 TRANS2

$PK
MDT = THETA(5)
IF(AMT.GT.0) BTIME=TIME
TAD=TIME-BTIME
TVCL=THETA(1)*WGT
TVV=THETA(2)*WGT
IF(APGR.LT.5) TVV=TVV*(1+THETA(3))
CL=TVCL*EXP(ETA(1))
V=TVV*EXP(ETA(2))
S1=V
KA=THETA(4)
D1 = 2*MDT

$ERROR
W=F
Y=F+W*EPS(1)
IPRED=F
IRES=DV-IPRED
IWRES=IRES/W

$THETA (0,0.00469307) ; CL
$THETA (0,1.00916) ; V
$THETA (-.99,.1)
$THETA (0,0.1) ; KA
$THETA  (0,0.1) ; TVMDT
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


@pytest.mark.parametrize(
    'etas, etab, buf_new',
    [
        (
            ['ETA(1)'],
            'ETAB1 = (EXP(ETA(1))**THETA(4) - 1)/THETA(4)',
            'CL = TVCL*EXP(ETAB1)\nV=TVV*EXP(ETA(2))',
        ),
        (
            ['ETA(1)', 'ETA(2)'],
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
            ['eta(1)'],
            'ETAB1 = (EXP(ETA(1))**THETA(4) - 1)/THETA(4)',
            'CL = TVCL*EXP(ETAB1)\nV=TVV*EXP(ETA(2))',
        ),
    ],
)
def test_boxcox(pheno_path, etas, etab, buf_new):
    model = Model(pheno_path)

    boxcox(model, etas)
    model.update_source()

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

    expression = (
        f'(1 + ({num_1})/({denom_1}) + ({num_2})/({denom_2}) + ' f'({num_3})/({denom_3}))*ETA(1)'
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

    assert str(model.get_pred_pk_record()) == rec_ref
    assert model.parameters['df1'].init == 80


@pytest.mark.parametrize(
    'etas, etad, buf_new',
    [
        (
            ['ETA(1)'],
            'ETAD1 = ((ABS(ETA(1)) + 1)**THETA(4) - 1)*ABS(ETA(1))/(THETA(4)*ETA(1))',
            'CL = TVCL*EXP(ETAD1)\nV=TVV*EXP(ETA(2))',
        ),
    ],
)
def test_john_draper(pheno_path, etas, etad, buf_new):
    model = Model(pheno_path)

    john_draper(model, etas)
    model.update_source()

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

    assert str(model.get_pred_pk_record()) == rec_ref


@pytest.mark.parametrize(
    'parameter, expression, operation, buf_new',
    [
        ('S1', 'exp', '+', 'V=TVV*EXP(ETA(2))\nS1 = V + EXP(ETA(3))'),
        ('S1', 'exp', '*', 'V=TVV*EXP(ETA(2))\nS1 = V*EXP(ETA(3))'),
        ('V', 'exp', '+', 'V = TVV*EXP(ETA(2)) + EXP(ETA(3))\nS1=V'),
        ('S1', 'add', None, 'V=TVV*EXP(ETA(2))\nS1 = ETA(3) + V'),
        ('S1', 'prop', None, 'V=TVV*EXP(ETA(2))\nS1 = ETA(3)*V'),
        ('S1', 'log', None, 'V=TVV*EXP(ETA(2))\nS1 = V*EXP(ETA(3))/(EXP(ETA(3)) + 1)'),
        ('S1', 'eta_new', '+', 'V=TVV*EXP(ETA(2))\nS1 = ETA(3) + V'),
        ('S1', 'eta_new**2', '+', 'V=TVV*EXP(ETA(2))\nS1 = ETA(3)**2 + V'),
    ],
)
def test_add_etas(pheno_path, parameter, expression, operation, buf_new):
    model = Model(pheno_path)

    add_etas(model, parameter, expression, operation)
    model.update_source()

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

    assert str(model.get_pred_pk_record()) == rec_ref

    last_rec = model.control_stream.get_records('OMEGA')[-1]

    assert str(last_rec) == f'$OMEGA  0.1 ; IIV_{parameter}\n'


@pytest.mark.parametrize(
    'etas, pk_ref, omega_ref',
    [
        (
            ['ETA(1)', 'ETA(2)'],
            '$PK\n'
            'CL = THETA(1)*EXP(ETA(4))\n'
            'V = THETA(2)*EXP(ETA(5))\n'
            'S1 = ETA(1) + ETA(2) + ETA(3) + V\n\n',
            '$OMEGA 0.1\n$OMEGA BLOCK(2)\n0.0309626\n0.0005 0.031128\n'
            '$OMEGA BLOCK(2)\n0.0309626\n0.0031045\t0.031128\n',
        ),
        (
            ['ETA(1)', 'ETA(3)'],
            '$PK\nCL = THETA(1)*EXP(ETA(4))\nV = THETA(2)*EXP(ETA(1))\n'
            'S1 = ETA(2) + ETA(3) + ETA(5) + V\n\n',
            '$OMEGA 0.031128  ; IVV\n$OMEGA BLOCK(2)\n0.0309626\n0.0005 0.031128\n'
            '$OMEGA BLOCK(2)\n0.0309626\n0.0055644\t0.1\n',
        ),
        (
            ['ETA(2)', 'ETA(3)'],
            '$PK\nCL=THETA(1)*EXP(ETA(1))\n'
            'V = THETA(2)*EXP(ETA(4))\n'
            'S1 = ETA(2) + ETA(3) + ETA(5) + V\n\n',
            '$OMEGA 0.0309626  ; IVCL\n$OMEGA BLOCK(2)\n0.0309626\n0.0005 0.031128\n'
            '$OMEGA BLOCK(2)\n0.031128\n0.0055792\t0.1\n',
        ),
        (
            ['ETA(1)', 'ETA(2)', 'ETA(4)'],
            '$PK\n'
            'CL = THETA(1)*EXP(ETA(3))\n'
            'V = THETA(2)*EXP(ETA(4))\n'
            'S1 = ETA(1) + ETA(2) + ETA(5) + V\n\n',
            '$OMEGA 0.1\n$OMEGA  0.031128\n$OMEGA BLOCK(3)\n0.0309626\n0.0031045\t0.031128\n'
            '0.0030963\t0.0031045\t0.0309626\n',
        ),
        (
            None,
            '$PK\nCL=THETA(1)*EXP(ETA(1))\nV=THETA(2)*EXP(ETA(2))\nS1=V+ETA(3)+ETA(4)+ETA(5)\n\n',
            '$OMEGA BLOCK(5)\n'
            '0.0309626\n'
            '0.0031045\t0.031128\n'
            '0.0055644\t0.0055792\t0.1\n'
            '0.0030963\t0.0031045\t0.0055644\t0.0309626\n'
            '0.0031045\t0.0031128\t0.0055792\t0.0005\t0.031128\n',
        ),
    ],
)
def test_block_rvs(testdata, etas, pk_ref, omega_ref):
    model = Model(testdata / 'nonmem/pheno_block.mod')
    create_rv_block(model, etas)
    model.update_source()

    assert str(model.get_pred_pk_record()) == pk_ref

    rec_omega = ''.join(str(rec) for rec in model.control_stream.get_records('OMEGA'))

    assert rec_omega == omega_ref


@pytest.mark.parametrize(
    'epsilons, err_ref, omega_ref',
    [
        (
            ['EPS(1)'],
            'Y = EPS(1)*W*EXP(ETA(3)) + F\n' 'IPRED=F+EPS(2)\n' 'IRES=DV-IPRED+EPS(3)\n',
            '$OMEGA  0.01 ; IIV_RUV1',
        ),
        (
            ['EPS(1)', 'EPS(2)'],
            'Y = EPS(1)*W*EXP(ETA(3)) + F\n'
            'IPRED = EPS(2)*EXP(ETA(4)) + F\n'
            'IRES=DV-IPRED+EPS(3)\n',
            '$OMEGA  0.01 ; IIV_RUV1\n' '$OMEGA  0.01 ; IIV_RUV2',
        ),
        (
            ['EPS(1)', 'EPS(3)'],
            'Y = EPS(1)*W*EXP(ETA(3)) + F\n'
            'IPRED = EPS(2) + F\n'
            'IRES = DV + EPS(3)*EXP(ETA(4)) - IPRED\n',
            '$OMEGA  0.01 ; IIV_RUV1\n' '$OMEGA  0.01 ; IIV_RUV2',
        ),
        (
            None,
            'Y = EPS(1)*W*EXP(ETA(3)) + F\n'
            'IPRED = EPS(2)*EXP(ETA(4)) + F\n'
            'IRES = DV + EPS(3)*EXP(ETA(5)) - IPRED\n',
            '$OMEGA  0.01 ; IIV_RUV1\n' '$OMEGA  0.01 ; IIV_RUV2\n' '$OMEGA  0.01 ; IIV_RUV3',
        ),
    ],
)
def test_iiv_on_ruv(pheno_path, epsilons, err_ref, omega_ref):
    model = Model(pheno_path)

    model_str = str(model)
    model_more_eps = re.sub(
        'IPRED=F\nIRES=DV-IPRED', 'IPRED=F+EPS(2)\nIRES=DV-IPRED+EPS(3)', model_str
    )
    model_sigma = re.sub(
        r'\$SIGMA 0.013241', '$SIGMA 0.013241\n$SIGMA 0.1\n$SIGMA 0.1', model_more_eps
    )
    model.control_stream = NMTranParser().parse(model_sigma)

    iiv_on_ruv(model, epsilons)
    model.update_source()

    err_rec = model.control_stream.get_records('ERROR')[0]

    assert str(err_rec) == f'$ERROR\n' f'W=F\n' f'{err_ref}' f'IWRES=IRES/W\n\n'

    omega_rec = ''.join(str(rec) for rec in model.control_stream.get_records('OMEGA'))

    assert omega_rec == (
        f'$OMEGA DIAGONAL(2)\n'
        f' 0.0309626  ;       IVCL\n'
        f' 0.031128  ;        IVV\n\n'
        f'{omega_ref}\n'
    )
