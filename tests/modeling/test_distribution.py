from io import StringIO

from pharmpy import Model
from pharmpy.modeling import add_peripheral_compartment


def create_model(s, testdata):
    model = Model(StringIO(s))
    model.source.path = testdata / 'nonmem' / 'pheno.mod'
    return model


def test_advan1(testdata):
    code = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV
$SUBROUTINE ADVAN1 TRANS2
$PK
CL=THETA(1)*EXP(ETA(1))
VC=THETA(2)*EXP(ETA(2))
V=VC
$ERROR
Y=F+F*EPS(1)
$THETA (0,0.00469307) ; POP_CL
$THETA (0,0.22) ; POP_VC
$OMEGA 0.0309626  ; IVCL
$OMEGA 0.0309626  ; IVV
$SIGMA 0.013241
$ESTIMATION METHOD=1 INTERACTION
"""
    model = create_model(code, testdata)
    add_peripheral_compartment(model)
    model.update_source()
    correct = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV
$SUBROUTINE ADVAN3 TRANS4
$PK
VP1 = THETA(4)
Q1 = THETA(3)
CL=THETA(1)*EXP(ETA(1))
VC=THETA(2)*EXP(ETA(2))
V1 = VC
Q = Q1
V2 = VP1
$ERROR
Y=F+F*EPS(1)
$THETA (0,0.00469307) ; POP_CL
$THETA (0,0.22) ; POP_VC
$THETA  (0,0.1) ; POP_Q1
$THETA  (0,0.1) ; POP_VP1
$OMEGA 0.0309626  ; IVCL
$OMEGA 0.0309626  ; IVV
$SIGMA 0.013241
$ESTIMATION METHOD=1 INTERACTION
"""
    assert str(model) == correct


def test_advan2(testdata):
    code = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV
$SUBROUTINE ADVAN2 TRANS2
$PK
CL=THETA(1)*EXP(ETA(1))
VC=THETA(2)*EXP(ETA(2))
MAT=THETA(3)*EXP(ETA(3))
V=VC
KA=1/MAT
$ERROR
Y=F+F*EPS(1)
$THETA (0,0.00469307) ; POP_CL
$THETA (0,0.22) ; POP_VC
$OMEGA 0.0309626  ; IVCL
$OMEGA 0.0309626  ; IVV
$OMEGA 0.0309626  ; IVMAT
$SIGMA 0.013241
$ESTIMATION METHOD=1 INTERACTION
"""
    model = create_model(code, testdata)
    add_peripheral_compartment(model)
    model.update_source()
    correct = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV
$SUBROUTINE ADVAN4 TRANS4
$PK
VP1 = THETA(4)
Q1 = THETA(3)
CL=THETA(1)*EXP(ETA(1))
VC=THETA(2)*EXP(ETA(2))
MAT=THETA(3)*EXP(ETA(3))
V2 = VC
KA=1/MAT
Q = Q1
V3 = VP1
$ERROR
Y=F+F*EPS(1)
$THETA (0,0.00469307) ; POP_CL
$THETA (0,0.22) ; POP_VC
$THETA  (0,0.1) ; POP_Q1
$THETA  (0,0.1) ; POP_VP1
$OMEGA 0.0309626  ; IVCL
$OMEGA 0.0309626  ; IVV
$OMEGA 0.0309626  ; IVMAT
$SIGMA 0.013241
$ESTIMATION METHOD=1 INTERACTION
"""
    assert str(model) == correct
