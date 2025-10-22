from io import StringIO

import pytest

from pharmpy.deps import pandas as pd
from pharmpy.model import DataInfo
from pharmpy.tools.external.results import parse_modelfit_results
from pharmpy.tools.linearize.delinearize import delinearize_model
from pharmpy.tools.linearize.results import calculate_results, psn_linearize_results
from pharmpy.tools.linearize.tool import (
    _create_linearized_model,
    _create_linearized_model_statements,
    create_derivative_model,
)
from pharmpy.tools.psn_helpers import create_results
from pharmpy.workflows import ModelEntry
from pharmpy.workflows.contexts import NullContext


def test_ofv(load_model_for_test, testdata):
    base = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    baseres = parse_modelfit_results(base, testdata / 'nonmem' / 'pheno.mod')
    lin = load_model_for_test(testdata / 'nonmem' / 'qa' / 'pheno_linbase.mod')
    linres = parse_modelfit_results(lin, testdata / 'nonmem' / 'qa' / 'pheno_linbase.mod')
    res = calculate_results(base, baseres, lin, linres)
    correct = """,OFV
base,730.894727
lin_evaluated,730.894727
lin_estimated,730.847272
"""
    correct = pd.read_csv(StringIO(correct), index_col=[0])['OFV']
    pd.testing.assert_series_equal(res.ofv, correct, atol=1e-6)


def test_iofv(load_model_for_test, testdata):
    base = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    baseres = parse_modelfit_results(base, testdata / 'nonmem' / 'pheno.mod')
    lin = load_model_for_test(testdata / 'nonmem' / 'qa' / 'pheno_linbase.mod')
    linres = parse_modelfit_results(lin, testdata / 'nonmem' / 'qa' / 'pheno_linbase.mod')
    res = calculate_results(base, baseres, lin, linres)
    correct = """,base,linear,delta
1,7.742852,7.722670,-0.020182
2,12.049275,12.072932,0.023657
3,12.042000,12.025054,-0.016946
4,12.812738,12.767298,-0.045440
5,10.092658,10.052713,-0.039945
6,14.345532,14.466305,0.120773
7,11.092995,11.062689,-0.030306
8,13.515743,13.483011,-0.032733
9,15.320529,15.253098,-0.067431
10,10.998785,10.959470,-0.039316
11,5.216710,5.214551,-0.002159
12,12.099920,12.125264,0.025345
13,10.321688,10.306268,-0.015421
14,18.261250,18.333781,0.072531
15,7.671241,7.651472,-0.019769
16,12.330722,12.297535,-0.033187
17,12.936166,12.906462,-0.029704
18,19.714083,19.871216,0.157133
19,12.019829,12.011825,-0.008004
20,12.056143,12.013481,-0.042662
21,12.248740,12.213907,-0.034834
22,7.605212,7.571470,-0.033742
23,19.815930,19.898940,0.083010
24,27.454119,27.483709,0.029590
25,27.964652,28.119465,0.154813
26,13.186694,13.170101,-0.016592
27,9.077661,9.064011,-0.013650
28,7.940614,7.941905,0.001291
29,5.074891,5.073444,-0.001446
30,9.256362,9.245473,-0.010890
31,5.103892,5.101951,-0.001941
32,19.907691,19.900512,-0.007179
33,7.743716,7.709957,-0.033759
34,8.047320,8.020993,-0.026327
35,9.430297,9.400876,-0.029421
36,13.781610,13.798023,0.016413
37,8.378942,8.371434,-0.007508
38,16.194730,16.237192,0.042462
39,15.599207,15.525574,-0.073633
40,6.709165,6.667489,-0.041676
41,11.219045,11.180049,-0.038996
42,18.122720,18.296645,0.173925
43,6.229697,6.228533,-0.001165
44,10.756400,10.734272,-0.022128
45,10.979745,10.927799,-0.051946
46,4.813994,4.812182,-0.001813
47,6.234951,6.233800,-0.001151
48,35.390033,35.431557,0.041524
49,12.057166,12.047648,-0.009517
50,19.429921,19.365236,-0.064686
51,15.011225,15.105334,0.094108
52,16.302748,16.342949,0.040200
53,9.292516,9.307615,0.015098
54,15.067187,14.977040,-0.090147
55,4.359968,4.357374,-0.002594
56,7.340782,7.341149,0.000366
57,9.515368,9.511703,-0.003665
58,11.970487,11.940629,-0.029858
59,13.638466,13.592238,-0.046228
"""
    correct = pd.read_csv(StringIO(correct), index_col=[0])
    correct.index.name = 'ID'
    pd.testing.assert_frame_equal(res.iofv, correct, atol=1e-6)


def test_psn_linearize_results(testdata):
    path = testdata / 'nonmem' / 'linearize' / 'linearize_dir1'
    res = psn_linearize_results(path)
    assert len(res.iofv) == 59
    assert res.ofv['base'] == pytest.approx(730.894727)


def test_create_results(testdata):
    path = testdata / 'nonmem' / 'linearize' / 'linearize_dir1'
    res = create_results(path)
    assert len(res.iofv) == 59
    assert res.ofv['base'] == pytest.approx(730.894727)


def test_derivative_model(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    ctx = NullContext()
    modelentry = create_derivative_model(ctx, ModelEntry.create(model=model))
    assert len(modelentry.model.execution_steps[0].derivatives) == 5
    assert (
        modelentry.model.code
        == '''$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV
$SUBROUTINE ADVAN1 TRANS2

$PK
CL=THETA(1)*EXP(ETA(1))
V=THETA(2)*EXP(ETA(2))
S1=V

$ERROR
Y=F+F*EPS(1)
D_EPSETA_1_1 = 0
D_EPSETA_1_2 = 0

"LAST
"  D_EPSETA_1_1=HH(1, 2)
"  D_EPSETA_1_2=HH(1, 3)
$THETA (0,0.00469307) ; TVCL
$THETA (0,1.00916) ; TVV
$OMEGA 0.0309626  ; IVCL
$OMEGA 0.031128  ; IVV
$SIGMA 0.013241

$TABLE ID TIME DV CIPREDI G011 G021 H011 D_EPSETA_1_1 D_EPSETA_1_2'''
        + ''' FILE=mytab ONEHEADER NOAPPEND NOPRINT RFORMAT="(1PE16.9,300(1PE24.16))"
$ESTIMATION METHOD=COND INTER MAXEVAL=0\n'''
    )


def test_create_linearized_model(load_model_for_test, testdata):
    model = load_model_for_test(testdata / "nonmem" / "pheno.mod")
    derivative_model_path = (
        testdata / "nonmem" / "linearize" / "linearize_dir1" / "scm_dir1" / "derivatives.mod"
    )
    derivative_model = load_model_for_test(derivative_model_path)
    modelres = parse_modelfit_results(derivative_model, derivative_model_path)

    derivative_modelentry = ModelEntry.create(model=derivative_model, modelfit_results=modelres)

    ctx = NullContext()
    linearized_model = _create_linearized_model(
        ctx, "linbase", "Linearized model", model, derivative_modelentry
    )
    assert len(linearized_model.model.statements) == 9


def test_delinearize_model(load_model_for_test, testdata):
    path = testdata / 'nonmem' / 'pheno_real.mod'
    model = load_model_for_test(path)
    datainfo = DataInfo.create(
        [
            'D_ETA_1',
            'D_ETA_2',
            'D_EPS_1',
            'D_EPSETA_1_1',
            'D_EPSETA_1_2',
            'OETA_1',
            'OETA_2',
            'OPRED',
        ]
    )
    linbase = model.replace(datainfo=datainfo)
    statements = _create_linearized_model_statements(linbase, model)
    linbase = linbase.replace(statements=statements)

    delinearized_model = delinearize_model(linbase, model)
    assert model == delinearized_model

    pm_delinearized_model = delinearize_model(
        linbase, model, param_mapping={"ETA_1": "V", "ETA_2": "CL"}
    )
    assert model != pm_delinearized_model
    assert (
        delinearized_model.parameters["IVCL"].init == pm_delinearized_model.parameters["IIV_V"].init
    )
    assert (
        delinearized_model.parameters["IVV"].init == pm_delinearized_model.parameters["IIV_CL"].init
    )
