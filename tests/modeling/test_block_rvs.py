from io import StringIO

import pytest

from pharmpy import Model
from pharmpy.modeling import add_iiv, create_rv_block
from pharmpy.modeling.block_rvs import _choose_param_init
from pharmpy.random_variables import RandomVariable, RandomVariables
from pharmpy.results import ModelfitResults
from pharmpy.symbols import symbol as S


@pytest.mark.parametrize(
    'rvs, exception_msg',
    [
        (['ETA(3)', 'NON_EXISTENT_RV'], r'.*NON_EXISTENT_RV.*'),
        (['ETA(3)', 'ETA(6)'], r'.*ETA\(6\).*'),
        (['ETA(1)'], 'At least two random variables are needed'),
    ],
)
def test_incorrect_params(testdata, rvs, exception_msg):
    model = Model(
        testdata / 'nonmem' / 'modelfit_results' / 'onePROB' / 'multEST' / 'noSIM' / 'withBayes.mod'
    )

    with pytest.raises(Exception, match=exception_msg):
        create_rv_block(model, rvs)


def test_choose_param_init(pheno_path, testdata):
    model = Model(pheno_path)
    params = (model.parameters['OMEGA(1,1)'], model.parameters['OMEGA(2,2)'])
    rvs = RandomVariables(model.random_variables.etas)
    init = _choose_param_init(model, rvs, params)

    assert init == 0.0118179

    model = Model(pheno_path)
    model.source.path = testdata  # Path where there is no .ext-file
    init = _choose_param_init(model, rvs, params)

    assert init == 0.0031045

    model = Model(pheno_path)

    omega1 = S('OMEGA(3,3)')
    x = RandomVariable.normal('ETA(3)', 'IIV', 0, omega1)
    rvs.append(x)

    ie = model.modelfit_results.individual_estimates
    ie['ETA(3)'] = ie['ETA(1)']
    model.modelfit_results = ModelfitResults(individual_estimates=ie)

    init = _choose_param_init(model, rvs, params)

    assert init == 0.0118179

    # If one eta doesn't have individual estimates
    model = Model(pheno_path)
    add_iiv(model, 'S1', 'add')
    params = (model.parameters['OMEGA(1,1)'], model.parameters['IIV_S1'])
    rvs = RandomVariables([model.random_variables['ETA(1)'], model.random_variables['ETA_S1']])
    init = _choose_param_init(model, rvs, params)

    assert init == 0.0052789


def test_names(testdata):
    model = Model(
        StringIO(
            '''$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV
$SUBROUTINE ADVAN1 TRANS2

$PK
CL=THETA(1)*EXP(ETA(1))
V=THETA(2)*EXP(ETA(1))
S1=V*EXP(ETA(2))

$ERROR
Y=F+F*EPS(1)

$THETA (0,0.00469307) ; TVCL
$THETA (0,1.00916) ; TVV
$OMEGA 0.0309626  ; IVCL
$OMEGA 0.031128  ; IVV
$SIGMA 0.013241

$ESTIMATION METHOD=1 INTERACTION
'''
        )
    )
    model.source.path = testdata / 'nonmem' / 'pheno.mod'
    create_rv_block(model, model.random_variables.names)
    assert 'IIV_CL_V_IIV_S1' in model.parameters.names
