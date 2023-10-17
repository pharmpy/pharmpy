from pharmpy.model import Assignment
from pharmpy.modeling import replace_non_random_rvs


def test_replace_non_random_rvs(create_model_for_test):
    code = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN1 TRANS2
$PK
CL=THETA(1)*EXP(ETA(1))
VC=THETA(2)*EXP(ETA(2))+OMEGA(2,2)
V=VC
$ERROR
Y=F+F*EPS(1)
$THETA (0,0.00469307) ; POP_CL
$THETA (0,0.22) ; POP_VC
$OMEGA 0.0309626  ; IVCL
$OMEGA 0 FIX  ; IVV
$SIGMA 0.013241 ; SIGMA
$ESTIMATION METHOD=1 INTERACTION
"""
    model = create_model_for_test(code, dataset='pheno')
    new = replace_non_random_rvs(model)
    assert new.parameters == model.parameters[['POP_CL', 'POP_VC', 'IVCL', 'SIGMA']]
    assert new.random_variables == model.random_variables[['ETA_1', 'EPS_1']]
    assert new.statements[0] == Assignment.create("CL", "POP_CL*exp(ETA_1)")
    assert new.statements[1] == Assignment.create("VC", "POP_VC")
