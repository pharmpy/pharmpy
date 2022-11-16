import pytest
from sympy import Symbol as S

from pharmpy.model import NormalDistribution
from pharmpy.modeling import add_iiv, create_joint_distribution
from pharmpy.modeling.block_rvs import _choose_param_init
from pharmpy.results import ModelfitResults


@pytest.mark.parametrize(
    'rvs, exception_msg',
    [
        (['ETA(3)', 'NON_EXISTENT_RV'], r'.*non-existing.*'),
        (['ETA(3)', 'ETA(6)'], r'.*ETA\(6\).*'),
        (['ETA(1)'], 'At least two random variables are needed'),
    ],
)
def test_incorrect_params(load_model_for_test, testdata, rvs, exception_msg):
    model = load_model_for_test(
        testdata / 'nonmem' / 'modelfit_results' / 'onePROB' / 'multEST' / 'noSIM' / 'withBayes.mod'
    )

    with pytest.raises(Exception, match=exception_msg):
        create_joint_distribution(
            model, rvs, individual_estimates=model.modelfit_results.individual_estimates
        )


def test_choose_param_init(load_model_for_test, pheno_path):
    model = load_model_for_test(pheno_path)
    params = (model.parameters['OMEGA(1,1)'], model.parameters['OMEGA(2,2)'])
    rvs = model.random_variables.etas
    init = _choose_param_init(model, model.modelfit_results.individual_estimates, rvs, *params)
    assert init == 0.0118179

    model = load_model_for_test(pheno_path)
    model.modelfit_results = None
    init = _choose_param_init(model, None, rvs, *params)
    assert init == 0.0031045

    model = load_model_for_test(pheno_path)
    rv_new = NormalDistribution.create('ETA(3)', 'IIV', 0, S('OMEGA(3,3)'))
    rvs += rv_new
    res = model.modelfit_results
    ie = res.individual_estimates.copy()
    ie['ETA(3)'] = ie['ETA(1)']
    model.modelfit_results = ModelfitResults(
        parameter_estimates=res.parameter_estimates, individual_estimates=ie
    )
    init = _choose_param_init(model, res.individual_estimates, rvs, *params)
    assert init == 0.0118179

    # If one eta doesn't have individual estimates
    model = load_model_for_test(pheno_path)
    add_iiv(model, 'S1', 'add')
    params = (model.parameters['OMEGA(1,1)'], model.parameters['IIV_S1'])
    rvs = model.random_variables[('ETA(1)', 'ETA_S1')]
    init = _choose_param_init(model, model.modelfit_results.individual_estimates, rvs, *params)
    assert init == 0.0052789

    # If the standard deviation in individual estimates of one eta is 0
    model = load_model_for_test(pheno_path)
    res = model.modelfit_results
    ie = res.individual_estimates.copy()
    ie['ETA(1)'] = 0
    model.modelfit_results = ModelfitResults(
        parameter_estimates=res.parameter_estimates, individual_estimates=ie
    )
    params = (model.parameters['OMEGA(1,1)'], model.parameters['OMEGA(2,2)'])
    rvs = model.random_variables[('ETA(1)', 'ETA(2)')]
    with pytest.warns(UserWarning, match='Correlation of individual estimates'):
        init = _choose_param_init(model, model.modelfit_results.individual_estimates, rvs, *params)
        assert init == 0.0031045


def test_choose_param_init_fo(create_model_for_test):
    model = create_model_for_test(
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
    params = (model.parameters['OMEGA(1,1)'], model.parameters['OMEGA(2,2)'])
    rvs = model.random_variables.etas
    init = _choose_param_init(model, None, rvs, *params)

    assert init == 0.01


def test_names(create_model_for_test):
    model = create_model_for_test(
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
    create_joint_distribution(
        model,
        model.random_variables.names,
        individual_estimates=model.modelfit_results.individual_estimates
        if model.modelfit_results is not None
        else None,
    )
    assert 'IIV_CL_V_IIV_S1' in model.parameters.names
