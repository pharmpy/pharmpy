from io import StringIO

from pharmpy import Model
from pharmpy.modeling import additive_error, combined_error, proportional_error, remove_error
from pharmpy.modeling.error import _get_prop_init


def test_remove_error_model(testdata):
    model = Model(testdata / 'nonmem' / 'pheno.mod')
    remove_error(model)
    model.update_source()
    assert str(model).split('\n')[11] == 'Y = F'


def test_additive_error_model(testdata):
    model = Model(testdata / 'nonmem' / 'pheno.mod')
    additive_error(model)
    model.update_source()
    assert str(model).split('\n')[11] == 'Y = F + EPS(1)'
    assert str(model).split('\n')[17] == '$SIGMA  11.2225 ; sigma'


def test_proportional_error_model(testdata):
    model = Model(testdata / 'nonmem' / 'pheno.mod')
    proportional_error(model)
    model.update_source()
    assert str(model).split('\n')[11] == 'Y=F+F*EPS(1)'
    assert str(model).split('\n')[17] == '$SIGMA  0.09 ; sigma'


def test_combined_error_model(testdata):
    model = Model(testdata / 'nonmem' / 'pheno.mod')
    combined_error(model)
    model.update_source()
    assert str(model).split('\n')[11] == 'Y = F + EPS(1)*F + EPS(2)'
    assert str(model).split('\n')[17] == '$SIGMA  0.09 ; sigma_prop'
    assert str(model).split('\n')[18] == '$SIGMA  11.2225 ; sigma_add'


def test_remove_error_without_f(testdata):
    code = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV
$SUBROUTINE ADVAN1 TRANS2
$PK
CL=THETA(1)*EXP(ETA(1))
V=THETA(2)*EXP(ETA(2))
$ERROR
CONC=A(1)/V
Y=CONC+CONC*EPS(1)
$THETA (0,0.00469307) ; TVCL
$THETA (0,1.00916) ; TVV
$OMEGA 0.0309626  ; IVCL
$OMEGA 0.031128  ; IVV
$SIGMA 0.013241
$ESTIMATION METHOD=1 INTERACTION
"""
    model = Model(StringIO(code))
    model.source.path = testdata / 'nonmem' / 'pheno.mod'  # To be able to find dataset
    remove_error(model)
    model.update_source()
    correct = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV
$SUBROUTINE ADVAN1 TRANS2
$PK
CL=THETA(1)*EXP(ETA(1))
V=THETA(2)*EXP(ETA(2))
$ERROR
CONC=A(1)/V
Y = CONC
$THETA (0,0.00469307) ; TVCL
$THETA (0,1.00916) ; TVV
$OMEGA 0.0309626  ; IVCL
$OMEGA 0.031128  ; IVV
$ESTIMATION METHOD=1 INTERACTION
"""
    assert str(model) == correct


def test_additive_error_without_f(testdata):
    code = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV
$SUBROUTINE ADVAN1 TRANS2
$PK
CL=THETA(1)*EXP(ETA(1))
V=THETA(2)*EXP(ETA(2))
$ERROR
PRED=A(1)/V
CONC=PRED
Y=CONC+CONC*EPS(1)
$THETA (0,0.00469307) ; TVCL
$THETA (0,1.00916) ; TVV
$OMEGA 0.0309626  ; IVCL
$OMEGA 0.031128  ; IVV
$SIGMA 0.013241
$ESTIMATION METHOD=1 INTERACTION
"""
    model = Model(StringIO(code))
    model.source.path = testdata / 'nonmem' / 'pheno.mod'  # To be able to find dataset
    additive_error(model)
    model.update_source()
    correct = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV
$SUBROUTINE ADVAN1 TRANS2
$PK
CL=THETA(1)*EXP(ETA(1))
V=THETA(2)*EXP(ETA(2))
$ERROR
PRED=A(1)/V
CONC=PRED
Y = CONC + EPS(1)
$THETA (0,0.00469307) ; TVCL
$THETA (0,1.00916) ; TVV
$OMEGA 0.0309626  ; IVCL
$OMEGA 0.031128  ; IVV
$SIGMA  11.2225 ; sigma
$ESTIMATION METHOD=1 INTERACTION
"""
    assert str(model) == correct


def test_get_prop_init(testdata):
    model = Model(testdata / 'nonmem' / 'pheno.mod')
    model.source.path = testdata / 'nonmem' / 'pheno.mod'  # To be able to find dataset

    init = _get_prop_init(model.dataset)
    assert init == 11.2225

    model.dataset['DV'].values[:] = 0.0
    init = _get_prop_init(model.dataset)
    assert init == 0.01
