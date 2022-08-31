from io import StringIO

import pytest
from sympy import Symbol as S

from pharmpy import Model
from pharmpy.modeling import add_iiv, create_joint_distribution
from pharmpy.modeling.block_rvs import _choose_param_init
from pharmpy.random_variables import RandomVariable, RandomVariables
from pharmpy.results import ModelfitResults


@pytest.mark.parametrize(
    'rvs, exception_msg',
    [
        (['ETA(3)', 'NON_EXISTENT_RV'], r'.*NON_EXISTENT_RV.*'),
        (['ETA(3)', 'ETA(6)'], r'.*ETA\(6\).*'),
        (['ETA(1)'], 'At least two random variables are needed'),
    ],
)
def test_incorrect_params(testdata, rvs, exception_msg):
    model = Model.create_model(
        testdata / 'nonmem' / 'modelfit_results' / 'onePROB' / 'multEST' / 'noSIM' / 'withBayes.mod'
    )

    with pytest.raises(Exception, match=exception_msg):
        create_joint_distribution(model, rvs)


def test_choose_param_init(pheno_path, testdata):
    model = Model.create_model(pheno_path)
    params = (model.parameters['OMEGA(1,1)'], model.parameters['OMEGA(2,2)'])
    rvs = RandomVariables(model.random_variables.etas)
    init = _choose_param_init(model, rvs, params)

    assert init == 0.0108944

    model = Model.create_model(pheno_path)
    model.modelfit_results = None
    model.name = 'run23'  # So that no results could be found
    init = _choose_param_init(model, rvs, params)

    assert init == 0.0031045

    model = Model.create_model(pheno_path)

    omega1 = S('OMEGA(3,3)')
    x = RandomVariable.normal('ETA(3)', 'IIV', 0, omega1)
    rvs.append(x)
    res = model.modelfit_results
    ie = res.individual_estimates
    ie['ETA(3)'] = ie['ETA(1)']
    model.modelfit_results = ModelfitResults(
        parameter_estimates=res.parameter_estimates, individual_estimates=ie
    )

    init = _choose_param_init(model, rvs, params)

    assert init == 0.0108944

    # If one eta doesn't have individual estimates
    model = Model.create_model(pheno_path)
    add_iiv(model, 'S1', 'add')
    params = (model.parameters['OMEGA(1,1)'], model.parameters['IIV_S1'])
    rvs = RandomVariables([model.random_variables['ETA(1)'], model.random_variables['ETA_S1']])
    init = _choose_param_init(model, rvs, params)

    assert init == 0.0051396

    # If the standard deviation in individual estimates of one eta is 0
    model = Model.create_model(pheno_path)
    res = model.modelfit_results
    ie = res.individual_estimates
    ie['ETA(1)'] = 0
    model.modelfit_results = ModelfitResults(
        parameter_estimates=res.parameter_estimates, individual_estimates=ie
    )
    params = (model.parameters['OMEGA(1,1)'], model.parameters['OMEGA(2,2)'])
    rvs = RandomVariables([model.random_variables['ETA(1)'], model.random_variables['ETA(2)']])
    with pytest.warns(UserWarning, match='Correlation of individual estimates'):
        init = _choose_param_init(model, rvs, params)
        assert init == 0.0028619


def test_choose_param_init_fo():
    model = Model.create_model(
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
$OMEGA 0.1  ; IVCL
$OMEGA 0.1  ; IVV
$SIGMA 0.013241

$ESTIMATION METHOD=0
'''
        )
    )
    params = (model.parameters['OMEGA(1,1)'], model.parameters['OMEGA(2,2)'])
    rvs = RandomVariables(model.random_variables.etas)
    init = _choose_param_init(model, rvs, params)

    assert init == 0.01


def test_names(testdata):
    model = Model.create_model(
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
    create_joint_distribution(model, model.random_variables.names)
    assert 'IIV_CL_V_IIV_S1' in model.parameters.names
