import shutil

import pytest
import sympy
from sympy import Symbol as symbol

from pharmpy.deps import pandas as pd
from pharmpy.internals.fs.cwd import chdir
from pharmpy.model import (
    Assignment,
    EstimationStep,
    EstimationSteps,
    Model,
    ModelSyntaxError,
    NormalDistribution,
    ODESystem,
    Parameter,
    Parameters,
    Statements,
)
from pharmpy.model.external.nonmem import convert_model
from pharmpy.model.external.nonmem.nmtran_parser import NMTranParser
from pharmpy.model.external.nonmem.records.factory import create_record
from pharmpy.modeling import (
    add_iiv,
    add_population_parameter,
    create_joint_distribution,
    remove_iiv,
    set_estimation_step,
    set_initial_condition,
    set_initial_estimates,
    set_transit_compartments,
    set_zero_order_absorption,
    set_zero_order_elimination,
    set_zero_order_input,
)
from pharmpy.tools import read_modelfit_results
from pharmpy.tools.amd.funcs import create_start_model


def _ensure_trailing_newline(buf):
    # FIXME This should not be necessary
    return buf if buf[-1] == '\n' else buf + '\n'


def S(x):
    return symbol(x)


def test_source(pheno):
    assert pheno.model_code.startswith('$PROBLEM PHENOBARB')


def test_update_inits(load_model_for_test, pheno_path):
    from pharmpy.modeling import update_inits

    model = load_model_for_test(pheno_path)
    res = read_modelfit_results(pheno_path)
    model = update_inits(model, res.parameter_estimates)


def test_empty_ext_file(testdata):
    # assert existing but empty ext-file does not give modelfit_results
    res = read_modelfit_results(
        testdata / 'nonmem' / 'modelfit_results' / 'onePROB' / 'noESTwithSIM' / 'onlysim.mod'
    )
    assert res is None


def test_detection():
    Model.parse_model_from_string("$PROBLEM this\n$PRED\n")
    Model.parse_model_from_string("   \t$PROBLEM skld fjl\n$PRED\n")
    Model.parse_model_from_string(" $PRO l907\n$PRED\n")


def test_validate():
    with pytest.raises(ModelSyntaxError):
        Model.parse_model_from_string("$PROBLEM this\n$SIZES LIM1=3000\n$PRED\n")


def test_parameters(pheno):
    params = pheno.parameters
    assert len(params) == 6
    assert pheno.parameters['PTVCL'] == Parameter('PTVCL', 0.00469307, lower=0, upper=1000000)
    assert pheno.parameters['PTVV'] == Parameter('PTVV', 1.00916, lower=0, upper=1000000)
    assert pheno.parameters['THETA_3'] == Parameter('THETA_3', 0.1, lower=-0.99, upper=1000000)
    assert pheno.parameters['IVCL'] == Parameter('IVCL', 0.0309626, lower=0, upper=sympy.oo)
    assert pheno.parameters['IVV'] == Parameter('IVV', 0.031128, lower=0, upper=sympy.oo)
    assert pheno.parameters['SIGMA_1_1'] == Parameter(
        'SIGMA_1_1', 0.013241, lower=0, upper=sympy.oo
    )


def test_set_parameters(pheno_path, load_model_for_test):
    pheno = load_model_for_test(pheno_path)
    res = read_modelfit_results(pheno_path)
    params = {
        'PTVCL': 0.75,
        'PTVV': 0.5,
        'THETA_3': 0.25,
        'IVCL': 0.1,
        'IVV': 0.2,
        'SIGMA_1_1': 0.3,
    }
    model = set_initial_estimates(pheno, params)
    assert model.parameters['PTVCL'] == Parameter('PTVCL', 0.75, lower=0, upper=1000000)
    assert model.parameters['PTVV'] == Parameter('PTVV', 0.5, lower=0, upper=1000000)
    assert model.parameters['THETA_3'] == Parameter('THETA_3', 0.25, lower=-0.99, upper=1000000)
    assert model.parameters['IVCL'] == Parameter('IVCL', 0.1, lower=0, upper=sympy.oo)
    assert model.parameters['IVV'] == Parameter('IVV', 0.2, lower=0, upper=sympy.oo)
    assert model.parameters['SIGMA_1_1'] == Parameter('SIGMA_1_1', 0.3, lower=0, upper=sympy.oo)
    model = model.update_source()
    thetas = model.internals.control_stream.get_records('THETA')
    assert str(thetas[0]) == '$THETA (0,0.75) ; PTVCL\n'
    assert str(thetas[1]) == '$THETA (0,0.5) ; PTVV\n'
    assert str(thetas[2]) == '$THETA (-.99,0.25)\n'
    omegas = model.internals.control_stream.get_records('OMEGA')
    assert str(omegas[0]) == '$OMEGA DIAGONAL(2)\n 0.1  ;       IVCL\n 0.2  ;        IVV\n\n'
    sigmas = model.internals.control_stream.get_records('SIGMA')
    assert str(sigmas[0]) == '$SIGMA 0.3\n'

    model = set_initial_estimates(pheno, {'PTVCL': 18})
    assert model.parameters['PTVCL'] == Parameter('PTVCL', 18, lower=0, upper=1000000)
    assert model.parameters['PTVV'] == Parameter('PTVV', 1.00916, lower=0, upper=1000000)

    model = create_joint_distribution(pheno, individual_estimates=res.individual_estimates)
    with pytest.raises(UserWarning, match='Adjusting initial'):
        set_initial_estimates(model, {'IVV': 0.000001})


def test_adjust_iovs(load_model_for_test, testdata):
    model = load_model_for_test(
        testdata / 'nonmem' / 'modelfit_results' / 'onePROB' / 'multEST' / 'noSIM' / 'withBayes.mod'
    )
    model.parameters
    rvs = model.random_variables

    assert rvs[0].level == 'IIV'
    assert rvs[3].level == 'IOV'
    assert rvs[4].level == 'IOV'
    assert rvs[6].level == 'IOV'

    model = load_model_for_test(testdata / 'nonmem' / 'qa' / 'iov.mod')
    dists = model.random_variables
    assert dists[0].level == 'IIV'
    assert dists[1].level == 'IOV'
    assert dists[2].level == 'IOV'


@pytest.mark.parametrize(
    'param_new,init_expected,buf_new',
    [
        (Parameter.create('COVEFF', 0.2), 0.2, '$THETA  0.2 ; COVEFF'),
        (Parameter.create('THETA', 0.1), 0.1, '$THETA  0.1 ; THETA'),
        (Parameter.create('THETA', 0.1, 0, fix=True), 0.1, '$THETA  (0,0.1) FIX ; THETA'),
        (Parameter.create('RUV_prop', 0.1), 0.1, '$THETA  0.1 ; RUV_prop'),
    ],
)
def test_add_parameters(pheno, param_new, init_expected, buf_new):
    pset = [p for p in pheno.parameters]

    assert len(pset) == 6

    pset.append(param_new)
    model = pheno.replace(parameters=Parameters.create(pset))

    assert len(pset) == 7
    assert model.parameters[param_new.name].init == init_expected

    rec_ref = (
        f'$THETA (0,0.00469307) ; PTVCL\n'
        f'$THETA (0,1.00916) ; PTVV\n'
        f'$THETA (-.99,.1)\n'
        f'{buf_new}\n'
    )

    model = model.update_source()
    rec_mod = ''
    for rec in model.internals.control_stream.get_records('THETA'):
        rec_mod += str(rec)

    assert rec_ref == rec_mod


def test_add_two_parameters(pheno):
    assert len(pheno.parameters) == 6

    model = add_population_parameter(pheno, 'COVEFF1', 0.2)
    model = add_population_parameter(model, 'COVEFF2', 0.1)

    assert len(model.parameters) == 8
    assert model.parameters['COVEFF1'].init == 0.2
    assert model.parameters['COVEFF2'].init == 0.1


@pytest.mark.parametrize(
    'statement_new,param_new,buf_new',
    [
        (Assignment(S('CL'), sympy.Integer(2)), None, 'CL = 2'),
        (
            Assignment(S('Y'), S('THETA(4)') + S('THETA(5)')),
            [Parameter.create('THETA(4)', 0.1), Parameter.create('THETA(5)', 0.1)],
            'Y = THETA(4) + THETA(5)',
        ),
    ],
)
def test_add_statements(pheno, statement_new, buf_new, param_new):
    sset = pheno.statements
    assert len(sset) == 15

    # Insert new statement before ODE system.
    new_sset = []
    for s in sset:
        if isinstance(s, ODESystem):
            new_sset.append(statement_new)
        new_sset.append(s)

    model = pheno.replace(
        statements=Statements(new_sset), parameters=pheno.parameters + Parameters.create(param_new)
    )
    model = model.update_source()

    assert len(model.statements) == 16

    parser = NMTranParser()
    stream = parser.parse(model.model_code)

    assert str(model.internals.control_stream) == str(stream)

    rec_ref = (
        f'$PK\n'
        f'IF(AMT.GT.0) BTIME=TIME\n'
        f'TAD=TIME-BTIME\n'
        f'TVCL=THETA(1)*WGT\n'
        f'TVV=THETA(2)*WGT\n'
        f'IF(APGR.LT.5) TVV=TVV*(1+THETA(3))\n'
        f'CL=TVCL*EXP(ETA(1))\n'
        f'V=TVV*EXP(ETA(2))\n'
        f'S1=V\n'
        f'{buf_new}\n\n'
    )

    rec_mod = str(model.internals.control_stream.get_records('PK')[0])

    assert rec_ref == rec_mod


@pytest.mark.parametrize(
    'param_new, statement_new, buf_new',
    [
        (Parameter.create('X', 0.1), Assignment(S('Y'), S('X') + S('S1')), 'Y = S1 + THETA(4)'),
    ],
)
def test_add_parameters_and_statements(pheno, param_new, statement_new, buf_new):
    model = add_population_parameter(pheno, param_new.name, param_new.init)

    sset = model.statements

    # Insert new statement before ODE system.
    new_sset = []
    for s in sset:
        if isinstance(s, ODESystem):
            new_sset.append(statement_new)
        new_sset.append(s)

    model = model.replace(statements=Statements(new_sset))
    model = model.update_source()

    rec = (
        f'$PK\n'
        f'IF(AMT.GT.0) BTIME=TIME\n'
        f'TAD=TIME-BTIME\n'
        f'TVCL=THETA(1)*WGT\n'
        f'TVV=THETA(2)*WGT\n'
        f'IF(APGR.LT.5) TVV=TVV*(1+THETA(3))\n'
        f'CL=TVCL*EXP(ETA(1))\n'
        f'V=TVV*EXP(ETA(2))\n'
        f'S1=V\n'
        f'{buf_new}\n\n'
    )

    assert str(model.internals.control_stream.get_pred_pk_record()) == rec


@pytest.mark.parametrize('rv_new, buf_new', [(Parameter('omega', 0.1), '$OMEGA  0.1')])
def test_add_random_variables(pheno, rv_new, buf_new):
    rvs = pheno.random_variables

    eta = NormalDistribution.create('eta_new', 'iiv', 0, S(rv_new.name))

    model = add_population_parameter(pheno, rv_new.name, rv_new.init)
    model = model.replace(random_variables=rvs + eta)

    model = model.update_source()

    rec_ref = (
        f'$OMEGA DIAGONAL(2)\n'
        f' 0.0309626  ;       IVCL\n'
        f' 0.031128  ;        IVV\n\n'
        f'{buf_new} ; omega\n'
    )

    rec_mod = ''
    for rec in model.internals.control_stream.get_records('OMEGA'):
        rec_mod += str(rec)

    assert rec_mod == rec_ref

    rv = model.random_variables['eta_new']

    assert rv.mean == 0
    assert rv.variance.name == 'omega'


def test_add_random_variables_and_statements(pheno):
    rvs = pheno.random_variables

    eta = NormalDistribution.create('ETA_NEW', 'iiv', 0, S('omega'))
    rvs = rvs + eta
    model = add_population_parameter(pheno, 'omega', 0.1)

    eps = NormalDistribution.create('EPS_NEW', 'ruv', 0, S('sigma'))
    rvs = rvs + eps
    model = add_population_parameter(model, 'sigma', 0.1)

    sset = model.statements
    statement_new = Assignment(S('X'), 1 + S(eps.names[0]) + S(eta.names[0]))
    model = model.replace(
        statements=sset.before_odes + statement_new + sset.ode_system + sset.after_odes,
        random_variables=rvs,
    )

    model = model.update_source()
    print(model.internals.control_stream.get_pred_pk_record())
    assert str(model.internals.control_stream.get_pred_pk_record()).endswith(
        'X = ETA_NEW + EPS(2) + 1\n\n'
    )
    assert '$ABBR REPLACE ETA_NEW=ETA(1)'


def test_minimal(load_model_for_test, datadir):
    path = datadir / 'minimal.mod'
    model = load_model_for_test(path)
    assert len(model.statements) == 1
    assert model.statements[0].expression == symbol('THETA_1') + symbol('ETA_1') + symbol('EPS_1')


def test_initial_individual_estimates(load_model_for_test, datadir):
    path = datadir / 'minimal.mod'
    model = load_model_for_test(path)
    assert model.initial_individual_estimates is None

    path = datadir / 'pheno_etas.mod'
    model = load_model_for_test(path)
    inits = model.initial_individual_estimates
    assert len(inits) == 59
    assert len(inits.columns) == 2
    assert inits['ETA_1'][2] == -0.166321


def test_deterministic_theta_comments(pheno):
    no_option = 0
    for theta_record in pheno.internals.control_stream.get_records('THETA'):
        no_option += len(list(theta_record.root.subtrees('option')))

    assert no_option == 0


def test_remove_eta(pheno):
    model = remove_iiv(pheno, 'ETA_1')
    assert model.model_code.split('\n')[13] == 'V = TVV*EXP(ETA_2)'
    assert '$ABBR REPLACE ETA_2=ETA(1)'


def test_symbol_names_in_comment(load_model_for_test, pheno_path):
    model = load_model_for_test(pheno_path)
    assert model.statements[2].expression == S('PTVCL') * S('WGT')

    code = """$PROBLEM base model
$INPUT ID DV TIME
$DATA file.csv IGNORE=@

$PRED
TV = THETA(1)
Y = TV + ETA(1) + ERR(1)

$THETA 0.1  ; TV
$OMEGA 0.01
$SIGMA 1
$ESTIMATION METHOD=1 INTER MAXEVALS=9990 PRINT=2 POSTHOC
"""
    with pytest.warns(UserWarning):
        model = Model.parse_model_from_string(code)
        assert model.parameters.names == ['THETA_1', 'OMEGA_1_1', 'SIGMA_1_1']


def test_symbol_names_in_abbr(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno_abbr.mod')
    pset, rvs = model.parameters, model.random_variables

    assert 'THETA_CL' in pset.names
    assert 'ETA_CL' in rvs.etas.names


def test_clashing_parameter_names(load_model_for_test, datadir):
    with pytest.warns(UserWarning):
        model = load_model_for_test(datadir / 'pheno_clashing_symbols.mod')
    assert model.parameters.names == ['THETA_1', 'TVV', 'IVCL', 'OMEGA_2_2', 'SIGMA_1_1']

    code = """$PROBLEM base model
$INPUT ID DV TIME
$DATA file.csv IGNORE=@

$PRED
Y = THETA(1) + ETA(1) + ERR(1)

$THETA 0.1  ; TV
$OMEGA 0.01 ; TV
$SIGMA 1 ; TV
$ESTIMATION METHOD=1 INTER MAXEVALS=9990 PRINT=2 POSTHOC
"""
    with pytest.warns(UserWarning):
        model = Model.parse_model_from_string(code)
        assert model.parameters.names == ['TV', 'OMEGA_1_1', 'SIGMA_1_1']

    code = """$PROBLEM base model
$INPUT ID DV TIME
$DATA file.csv IGNORE=@

$PRED
Y = THETA(1) + THETA(2)

$THETA 0.1  ; TV
$THETA 0.1  ; TV
$ESTIMATION METHOD=1 INTER MAXEVALS=9990 PRINT=2 POSTHOC
"""
    with pytest.warns(UserWarning):
        model = Model.parse_model_from_string(code)
        assert model.parameters.names == ['TV', 'THETA_2']


def test_abbr_write(load_model_for_test, pheno_path):
    model = load_model_for_test(pheno_path)
    model = add_iiv(model, 'S1', 'add')

    assert 'ETA_S1' in model.model_code
    assert 'ETA_S1' in model.random_variables.names
    assert S('ETA_S1') in model.statements.free_symbols


def test_abbr_read_write(load_model_for_test, pheno_path):
    model_write = load_model_for_test(pheno_path)
    model_write = add_iiv(model_write, 'S1', 'add')
    model_read = Model.parse_model_from_string(model_write.model_code)
    assert model_read.model_code == model_write.model_code
    assert model_read.statements == model_write.statements
    assert not (
        set(model_read.random_variables.names) - set(model_write.random_variables.names)
    )  # Different order due to renaming in read


def test_dv_symbol(pheno):
    assert list(pheno.dependent_variables.keys())[0].name == 'Y'


def test_insert_unknown_record(pheno):
    rec = create_record('$TRIREME one')
    newcs = pheno.internals.control_stream.insert_record(rec)
    model = pheno.replace(internals=pheno.internals.replace(control_stream=newcs))
    assert model.model_code.split('\n')[-1] == '$TRIREME one'

    rec = create_record('\n$OA two')
    newcs = model.internals.control_stream.insert_record(rec)
    model = model.replace(internals=model.internals.replace(control_stream=newcs))
    assert model.model_code.split('\n')[-1] == '$OA two'


def test_parse_illegal_record_name():
    parser = NMTranParser()
    with pytest.raises(ModelSyntaxError):
        parser.parse('$PROBLEM MYPROB\n$1REC\n')


def test_frem_model():
    code = """$PROBLEM    MOXONIDINE PK,FINAL ESTIMATES,simulated data
$INPUT      ID VISI DROP DGRP DOSE DROP DROP DROP DROP NEUY SCR AGE
            SEX DROP WT DROP ACE DIG DIU DROP TAD TIME DROP CRCL AMT
            SS II DROP DROP DROP DV DROP DROP MDV FREMTYPE
$DATA      ../frem_dataset.dta IGNORE=@
$SUBROUTINE ADVAN2 TRANS1
$OMEGA  BLOCK(1)
 1.47E-06  ;     IOV_CL
$OMEGA  BLOCK(1) SAME
$OMEGA  BLOCK(1)
 5.06E-05  ;     IOV_KA
$OMEGA  BLOCK(1) SAME
$OMEGA  BLOCK(5)
 0.416046
 0.389578 0.578819  ;   IIV_CL_V
 0.00327757 0.00386591 0.258203  ;     IIV_KA
 -0.0809376 -0.130036 0.0777389 1  ;    BSV_AGE
 0.0880473 0.0821262 -0.0578989 -0.11524 1  ;    BSV_SEX
$PK
   VIS3               = 0
   IF(VISI.EQ.3) VIS3 = 1
   VIS8               = 0
   IF(VISI.EQ.8) VIS8 = 1
   KPCL  = VIS3*ETA(1)+VIS8*ETA(2)
   KPKA  = VIS3*ETA(3)+VIS8*ETA(4)
   TVCL  = THETA(1)
   TVV   = THETA(2)
   CL    = TVCL*EXP(ETA(5)+KPCL)
   V     = TVV*EXP(ETA(6))
   KA    = THETA(3)*EXP(ETA(7)+KPKA)
   ALAG1 = THETA(4)
   K     = CL/V
   S2    = V
    SDC8 = 7.82226906804
    SDC9 = 0.404756978659

$ERROR
     IPRED = LOG(.025)
     WA     = 1
     W      = WA
     IF(F.GT.0) IPRED = LOG(F)
     IRES  = IPRED-DV
     IWRES = IRES/W
     Y     = IPRED+ERR(1)*W

;;;FREM CODE BEGIN COMPACT
;;;DO NOT MODIFY
    IF (FREMTYPE.EQ.100) THEN
;      AGE  7.82226906804
       Y = THETA(5) + ETA(8)*SDC8 + EPS(2)
       IPRED = THETA(5) + ETA(8)*SDC8
    END IF
    IF (FREMTYPE.EQ.200) THEN
;      SEX  0.404756978659
       Y = THETA(6) + ETA(9)*SDC9 + EPS(2)
       IPRED = THETA(6) + ETA(9)*SDC9
    END IF
;;;FREM CODE END COMPACT
$THETA  (0,32.901) ; POP_TVCL
$THETA  (0,115.36) ; POP_TVV
$THETA  (0,1.45697) ; POP_TVKA
$THETA  (0,0.0818029) ; POP_LAG
$THETA  65.1756756757 FIX ; TV_AGE
 1.2027027027 FIX ; TV_SEX
$SIGMA  0.112373
$SIGMA  0.0000001  FIX  ;     EPSCOV
$ESTIMATION METHOD=1 MAXEVAL=9999 NONINFETA=1 MCETA=1
"""
    model = Model.parse_model_from_string(code)
    rvs = model.random_variables
    assert len(rvs.names) == 11


@pytest.mark.parametrize(
    'model_path, transformation',
    [
        ('nonmem/pheno.mod', set_zero_order_elimination),
    ],
)
def test_des(load_model_for_test, testdata, model_path, transformation):
    model_ref = load_model_for_test(testdata / model_path)
    model_ref = transformation(model_ref)

    model_des = Model.parse_model_from_string(model_ref.model_code)

    assert model_ref.statements.ode_system == model_des.statements.ode_system


def test_cmt_update(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox1.mod')
    old_cmt = model.dataset["CMT"]
    old_cmt = pd.to_numeric(old_cmt)

    model = set_transit_compartments(model, 2)
    updated_cmt = model.dataset["CMT"]

    assert old_cmt[0] == 1 and updated_cmt[0] == 1
    assert old_cmt[1] == 2 and updated_cmt[1] == 4


def test_zero_order_cmt(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox1.mod')
    old_cmt = model.dataset["CMT"]
    old_cmt = pd.to_numeric(old_cmt)

    model = set_zero_order_absorption(model)
    updated_cmt = model.dataset["CMT"]

    assert old_cmt[0] == 1 and updated_cmt[0] == 1
    assert old_cmt[1] == 2 and updated_cmt[1] == 1


@pytest.mark.parametrize(
    'estcode,est_steps',
    [
        ('$ESTIMATION METH=COND INTERACTION', [EstimationStep.create('foce', interaction=True)]),
        ('$ESTIMATION INTER METH=COND', [EstimationStep.create('foce', interaction=True)]),
        ('$ESTM METH=1 INTERACTION', [EstimationStep.create('foce', interaction=True)]),
        ('$ESTIM METH=1', [EstimationStep.create('foce')]),
        ('$ESTIMA METH=0', [EstimationStep.create('fo')]),
        ('$ESTIMA METH=ZERO', [EstimationStep.create('fo')]),
        ('$ESTIMA INTER', [EstimationStep.create('fo', interaction=True)]),
        ('$ESTIMA INTER\n$COV', [EstimationStep.create('fo', interaction=True, cov='sandwich')]),
        (
            '$ESTIMA METH=COND INTER\n$EST METH=COND',
            [
                EstimationStep.create('foce', interaction=True),
                EstimationStep.create('foce', interaction=False),
            ],
        ),
        ('$ESTIMATION METH=SAEM', [EstimationStep.create('saem')]),
        ('$ESTIMATION METH=1 LAPLACE', [EstimationStep.create('foce', laplace=True)]),
        (
            '$ESTIMATION METH=0 MAXEVAL=0',
            [EstimationStep.create('fo', evaluation=True)],
        ),
        (
            '$ESTIMATION METH=IMP EONLY=1',
            [EstimationStep.create('imp', evaluation=True)],
        ),
        (
            '$ESTIMATION METH=COND MAXEVAL=9999',
            [EstimationStep.create('foce', maximum_evaluations=9999)],
        ),
        (
            '$ESTIMATION METH=COND ISAMPLE=10 NITER=5 AUTO=1 PRINT=2',
            [EstimationStep.create('foce', isample=10, niter=5, auto=True, keep_every_nth_iter=2)],
        ),
    ],
)
def test_estimation_steps_getter(estcode, est_steps):
    code = '''$PROBLEM base model
$INPUT ID DV TIME
$DATA file.csv IGNORE=@
$PRED
Y = THETA(1) + ETA(1) + ERR(1)
$THETA 0.1
$OMEGA 0.01
$SIGMA 1
'''
    code += estcode
    model = Model.parse_model_from_string(code)
    correct = EstimationSteps.create(steps=est_steps)
    assert model.estimation_steps == correct


def test_estimation_steps_getter_options():
    code = '''$PROBLEM base model
$INPUT ID DV TIME
$DATA file.csv IGNORE=@
$PRED
Y = THETA(1) + ETA(1) + ERR(1)
$THETA 0.1
$OMEGA 0.01
$SIGMA 1
$ESTIMATION METHOD=1 SADDLE_RESET=1
'''
    model = Model.parse_model_from_string(code)
    assert model.estimation_steps[0].method == 'FOCE'
    assert model.estimation_steps[0].tool_options['SADDLE_RESET'] == '1'


@pytest.mark.parametrize(
    'estcode,kwargs,rec_ref',
    [
        ('$EST METH=COND INTER', {'method': 'fo'}, '$ESTIMATION METHOD=ZERO INTER'),
        ('$EST METH=COND INTER', {'interaction': False}, '$ESTIMATION METHOD=COND'),
        ('$EST METH=COND INTER', {'cov': 'sandwich'}, '$COVARIANCE'),
        ('$EST METH=COND INTER', {'cov': 'cpg'}, '$COVARIANCE MATRIX=S'),
        ('$EST METH=COND INTER', {'cov': 'ofim'}, '$COVARIANCE MATRIX=R'),
        (
            '$EST METH=COND INTER MAXEVAL=99999',
            {'method': 'fo'},
            '$ESTIMATION METHOD=ZERO INTER MAXEVAL=99999',
        ),
        (
            '$EST METH=COND INTER POSTHOC',
            {'method': 'fo'},
            '$ESTIMATION METHOD=ZERO INTER POSTHOC',
        ),
        ('$EST METH=COND INTER', {'laplace': True}, '$ESTIMATION METHOD=COND LAPLACE INTER'),
        (
            '$EST METH=COND INTER',
            {'isample': 10, 'niter': 5},
            '$ESTIMATION METHOD=COND INTER ISAMPLE=10 NITER=5',
        ),
        (
            '$EST METH=COND INTER',
            {'auto': True, 'keep_every_nth_iter': 2},
            '$ESTIMATION METHOD=COND INTER AUTO=1 PRINT=2',
        ),
        ('$EST METH=COND INTER', {'auto': False}, '$ESTIMATION METHOD=COND INTER AUTO=0'),
    ],
)
def test_estimation_steps_setter(estcode, kwargs, rec_ref):
    code = '''$PROBLEM base model
$INPUT ID DV TIME
$DATA tests/testdata/nonmem/file.csv IGNORE=@
$PRED
Y = THETA(1) + ETA(1) + ERR(1)
$THETA 0.1
$OMEGA 0.01
$SIGMA 1
'''
    code += estcode
    model = Model.parse_model_from_string(code)
    steps = model.estimation_steps
    newstep = steps[0].replace(**kwargs)
    model = model.replace(estimation_steps=newstep + steps[1:])
    model = model.update_source()
    assert model.model_code.split('\n')[-2] == rec_ref


@pytest.mark.parametrize(
    'estcode,kwargs,error_msg',
    [
        (
            '$EST METH=COND MAXEVAL=0',
            {'tool_options': {'MAXEVAL': 999}},
            'MAXEVAL already set as attribute in estimation method object',
        ),
        (
            '$EST METH=COND INTER',
            {'tool_options': {'INTERACTION': None}},
            'INTERACTION already set as attribute in estimation method object',
        ),
    ],
)
def test_set_estimation_steps_option_clash(estcode, kwargs, error_msg):
    code = '''$PROBLEM base model
$INPUT ID DV TIME
$DATA tests/testdata/nonmem/file.csv IGNORE=@
$PRED
Y = THETA(1) + ETA(1) + ERR(1)
$THETA 0.1
$OMEGA 0.01
$SIGMA 1
'''
    code += estcode
    model = Model.parse_model_from_string(code)

    steps = model.estimation_steps
    newstep = steps[0].replace(**kwargs)
    newsteps = newstep + steps[1:]
    model = model.replace(estimation_steps=newsteps)

    with pytest.raises(ValueError) as excinfo:
        model.update_source()
    assert error_msg == str(excinfo.value)


def test_add_estimation_step():
    code = '''$PROBLEM base model
$INPUT ID DV TIME
$DATA tests/testdata/nonmem/file.csv IGNORE=@
$PRED
Y = THETA(1) + ETA(1) + ERR(1)
$THETA 0.1
$OMEGA 0.01
$SIGMA 1
$EST METH=COND INTER
'''
    model = Model.parse_model_from_string(code)
    est_new = EstimationStep.create('IMP', interaction=True, tool_options={'saddle_reset': 1})
    model = model.replace(estimation_steps=model.estimation_steps + est_new)
    model = model.update_source()
    assert model.model_code.split('\n')[-2] == '$ESTIMATION METHOD=IMP INTER SADDLE_RESET=1'
    est_new = EstimationStep.create('SAEM', interaction=True)
    model = model.replace(estimation_steps=est_new + model.estimation_steps)
    model = model.update_source()
    assert model.model_code.split('\n')[-4] == '$ESTIMATION METHOD=SAEM INTER'
    est_new = EstimationStep.create('FO', evaluation=True)
    model = model.replace(estimation_steps=model.estimation_steps + est_new)
    model = model.update_source()
    assert model.model_code.split('\n')[-2] == '$ESTIMATION METHOD=ZERO MAXEVAL=0'
    est_new = EstimationStep.create('IMP', evaluation=True)
    model = model.replace(estimation_steps=model.estimation_steps + est_new)
    model = model.update_source()
    assert model.model_code.split('\n')[-2] == '$ESTIMATION METHOD=IMP EONLY=1'


def test_remove_estimation_step():
    code = '''$PROBLEM base model
$INPUT ID DV TIME
$DATA tests/testdata/nonmem/file.csv IGNORE=@
$PRED
Y = THETA(1) + ETA(1) + ERR(1)
$THETA 0.1
$OMEGA 0.01
$SIGMA 1
$EST METH=COND INTER
'''
    model = Model.parse_model_from_string(code)
    model = model.replace(estimation_steps=model.estimation_steps[1:])
    assert not model.estimation_steps
    model = model.update_source()
    assert model.model_code.split('\n')[-2] == '$SIGMA 1'


def test_update_source_comments():
    code = """
$PROBLEM    run3.mod PHENOBARB SIMPLE MODEL
$DATA      ../../pheno.dta IGNORE=@
$INPUT      ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN1 TRANS2
$PK


IF(AMT.GT.0) BTIME=TIME
TAD=TIME-BTIME
      TVCL=THETA(1)*WGT
      TVV=THETA(2)*WGT
IF(APGR.LT.5) TVV=TVV*(1+THETA(3))
      CL=TVCL*EXP(ETA(1))
      V=TVV*EXP(ETA(2))
      S1=V
$ERROR

      W=F
      Y=F+W*EPS(1)

      IPRED=F         ;  individual-specific prediction
      IRES=DV-IPRED   ;  individual-specific residual
      IWRES=IRES/W    ;  individual-specific weighted residual

$THETA  (0,0.00469555) ; CL
$THETA  (0,0.984258) ; V
$THETA  (-.99,0.15892)
$OMEGA  DIAGONAL(2)
 0.0293508  ;       IVCL
 0.027906  ;        IVV
$SIGMA  0.013241
$ESTIMATION METHOD=1 INTERACTION MAXEVALS=9999
$COVARIANCE UNCONDITIONAL
$TABLE      ID TIME AMT WGT APGR IPRED PRED TAD CWRES NPDE NOAPPEND
            NOPRINT ONEHEADER FILE=mytab3
"""
    with pytest.warns(UserWarning):
        model = Model.parse_model_from_string(code)
        model.update_source()


def test_convert_model(testdata):
    code = """$PROBLEM base model
$INPUT ID DV TIME
$DATA file.csv IGNORE=@

$PRED
Y = THETA(1) + ETA(1) + EPS(1)

$THETA 1  ; TH1
$OMEGA 2 ; OM1
$SIGMA 3 ; SI1
$ESTIMATION METHOD=1 INTER
"""
    base = Model.parse_model_from_string(code)
    base.dataset_path = testdata / 'nonmem' / 'file.csv'
    model = convert_model(base)
    model.dataset_path = "file.csv"  # Otherwise we get full path
    correct = """$PROBLEM base model
$INPUT ID DV TIME
$DATA file.csv IGNORE=@

$PRED
Y = THETA(1) + ETA(1) + EPS(1)

$THETA 1  ; TH1
$OMEGA 2 ; OM1
$SIGMA 3 ; SI1
$ESTIMATION METHOD=1 INTER
"""
    assert model.model_code == correct


def test_table_long_ids(testdata):
    code = f"""$PROBLEM base model
    $INPUT ID TIME AMT WGT APGR DV FA1 FA2
    $DATA {testdata / "nonmem" / "pheno.dta"} IGNORE=@

    $PRED
    Y = THETA(1) + ETA(1) + EPS(1)

    $THETA 1  ; TH1
    $OMEGA 2 ; OM1
    $SIGMA 3 ; SI1
    $ESTIMATION METHOD=1 INTER
    """
    model = Model.parse_model_from_string(code)
    dataset_new = model.dataset.copy()
    dataset_new['ID'] = dataset_new['ID'] * 10000
    model = model.replace(dataset=dataset_new)
    model = set_estimation_step(model, 'FO', residuals=['CWRES'])
    assert 'FORMAT=' in model.model_code


def test_convert_model_iv(testdata, tmp_path):
    # FIXME move to unit test for amd?
    with chdir(tmp_path):
        shutil.copy2(testdata / 'nonmem' / 'pheno_rate.dta', '.')
        start_model = create_start_model('pheno_rate.dta', modeltype='pk_iv')
        convert_model(start_model)


def test_parse_derivatives(load_model_for_test, testdata):
    model = load_model_for_test(
        testdata / "nonmem" / "linearize" / "linearize_dir1" / "scm_dir1" / "derivatives.mod"
    )
    assert model.estimation_steps[0].eta_derivatives == ('ETA_1', 'ETA_2')
    assert model.estimation_steps[0].epsilon_derivatives == ('EPS_1',)


def test_no_etas_in_model(pheno):
    pheno = remove_iiv(pheno)
    assert 'DUMMYETA' in pheno.model_code
    assert 'ETA(1)' in pheno.model_code


def test_0_fix_diag_omega():
    code = """$PROBLEM base model
$INPUT ID DV TIME
$DATA file.csv IGNORE=@

$PRED
Y = THETA(1) + ETA(1) + EPS(1)

$THETA 1  ; TH1
$OMEGA 0 FIX ; OM1
$SIGMA 3 ; SI1
$ESTIMATION METHOD=1 INTER
"""
    model = Model.parse_model_from_string(code)
    assert len(model.random_variables.etas) == 1


def test_solver():
    code = """
$PROBLEM
$DATA ../../pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN6 TOL=5
$MODEL COMP=(CENTRAL)
$PK
CL=THETA(1)*EXP(ETA(1))
V=THETA(2)*EXP(ETA(2))
S1=V
$DES
DADT(1) = -CL/V * A(1)
$ERROR
Y=F+F*EPS(1)
$THETA (0,0.00469555)
$THETA (0,0.984258)
$THETA (0.15892)
$OMEGA 0.0293508 0.027906
$SIGMA 0.013241
$ESTIMATION METHOD=1 INTERACTION MAXEVALS=9999
"""
    model = Model.parse_model_from_string(code)
    assert len(model.estimation_steps) == 1
    step = model.estimation_steps[0]
    assert step.solver == 'DVERK'
    assert step.solver_rtol == 5
    assert step.solver_atol == 1e-12

    code = """
$PROBLEM
$DATA ../../pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN13 TOL=5 ATOL=1.5
$MODEL COMP=(CENTRAL)
$PK
CL=THETA(1)*EXP(ETA(1))
V=THETA(2)*EXP(ETA(2))
S1=V
$DES
DADT(1) = -CL/V * A(1)
$ERROR
Y=F+F*EPS(1)
$THETA (0,0.00469555)
$THETA (0,0.984258)
$THETA (0.15892)
$OMEGA 0.0293508 0.027906
$SIGMA 0.013241
$ESTIMATION METHOD=1 INTERACTION MAXEVALS=9999
"""
    model = Model.parse_model_from_string(code)
    assert len(model.estimation_steps) == 1
    step = model.estimation_steps[0]
    assert step.solver == 'LSODA'
    assert step.solver_rtol == 5
    assert step.solver_atol == 1.5


def test_if_in_des():
    # Also test parsing of A_0(N) =
    code = """
$PROBLEM
$DATA ../../pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN6 TOL=5
$MODEL COMP=(CENTRAL)
$PK
CL=THETA(1)*EXP(ETA(1))
V=THETA(2)*EXP(ETA(2))
S1=V
A_0(1) = 2
$DES
IF (CL>0) THEN
DADT(1) = -CL/V * A(1)
ELSE
DADT(1) = -1
END IF
$ERROR
Y=F+F*EPS(1)
$THETA (0,0.00469555)
$THETA (0,0.984258)
$THETA (0.15892)
$OMEGA 0.0293508 0.027906
$SIGMA 0.013241
$ESTIMATION METHOD=1 INTERACTION MAXEVALS=9999
"""
    model = Model.parse_model_from_string(code)
    assert type(model.statements.ode_system.eqs[0].rhs) == sympy.Piecewise
    assert model.statements[3].symbol == sympy.Function("A_CENTRAL")(0)
    assert model.statements[3].expression == sympy.Integer(2)


@pytest.mark.parametrize(
    'estline,likelihood',
    [
        ('$ESTIMATION METHOD=1 LIKELIHOOD INTER', 'LIKELIHOOD'),
        ('$ESTIMATION METHOD=1 -2LL INTER', '-2LL'),
    ],
)
def test_likelihood(estline, likelihood):
    code = """$PROBLEM base model
$INPUT ID DV TIME
$DATA file.csv IGNORE=@

$PRED
Y = THETA(1) + ETA(1) + EPS(1)

$THETA 1  ; TH1
$OMEGA 0 FIX ; OM1
$SIGMA 3 ; SI1
"""
    model = Model.parse_model_from_string(code + estline)
    assert model.value_type == likelihood


def test_f_flag():
    code = """$PROBLEM base model
$INPUT ID DV TIME
$DATA file.csv IGNORE=@

$PRED
F_FLAG = 1
Y = THETA(1) + ETA(1) + EPS(1)

$THETA 1  ; TH1
$OMEGA 0 FIX ; OM1
$SIGMA 3 ; SI1
$ESTIMATION METHOD=1 INTER
"""
    model = Model.parse_model_from_string(code)
    assert model.value_type == sympy.Symbol('F_FLAG')


def test_datainfo_model_drop_clash(testdata):
    datapath = testdata / 'nonmem' / 'pheno.dta'
    code = f"""$PROBLEM
$DATA {datapath} IGNORE=@
$INPUT ID TIME AMT WGT=DROP APGR DV
$SUBROUTINE ADVAN1 TRANS2

$PK
CL=THETA(1)

$ERROR
Y=F+F*EPS(1)

$THETA (0,0.00469307) ; TVCL
$SIGMA 0.013241

$ESTIMATION METHOD=1 INTERACTION
"""
    with pytest.warns(
        UserWarning, match='NONMEM .mod and dataset .datainfo disagree on DROP for columns WGT'
    ):
        model = Model.parse_model_from_string(code)

    assert model.datainfo['WGT'].drop


def test_abbr_etas():
    code = '''$PROBLEM
$INPUT ID DV TIME
$DATA file.csv IGNORE=@
$ABBR REPLACE ETA_MY=ETA(1)
$PRED
VAR = ETA_MY
Y = THETA(1) + VAR + ERR(1)
$THETA 0.1
$OMEGA 0.01
$SIGMA 1
'''
    model = Model.parse_model_from_string(code)
    assert model.random_variables.etas.names == ['ETA_MY']

    model = add_iiv(model, ['Y'], 'exp', '+', eta_names=['ETA_DUMMY'])
    model = remove_iiv(model, ['ETA_MY'])
    model = add_iiv(model, ['VAR'], 'exp', '+', eta_names=['ETA_MY'])
    assert model.model_code.split('\n')[3] == '$ABBR REPLACE ETA_DUMMY=ETA(1)'
    assert model.model_code.split('\n')[4] == '$ABBR REPLACE ETA_MY=ETA(2)'
    assert not model.model_code.split('\n')[5].startswith('$ABBR')
    assert 'VAR = EXP(ETA_MY)' in model.model_code
    assert 'Y = THETA(1) + VAR + ERR(1) + EXP(ETA_DUMMY)'
    model = remove_iiv(model, ['ETA_DUMMY'])
    assert model.model_code.split('\n')[3] == '$ABBR REPLACE ETA_MY=ETA(1)'
    assert not model.model_code.split('\n')[4].startswith('$ABBR')


def test_abbr_not_replace():
    code = '''$PROBLEM
$INPUT ID DV TIME
$DATA file.csv IGNORE=@
$ABBR PROTECT DERIV2=NO
$PRED
VAR = ETA(1)
Y = THETA(1) + VAR + ERR(1)
$THETA 0.1
$OMEGA 0.01
$SIGMA 1
'''
    model = Model.parse_model_from_string(code)
    model = add_iiv(model, ['Y'], 'exp', '+', eta_names=['ETA_DUMMY'])
    assert model.model_code.split('\n')[3] == '$ABBR PROTECT DERIV2=NO'


def test_parse_dvid(testdata, load_model_for_test):
    model = load_model_for_test(testdata / 'nonmem' / 'models' / 'pheno_dvid.mod')
    assert model.statements[-1] == Assignment.create("Y_2", "EPS_1 * F + EPS_2 + F")
    model = model.update_source()
    assert model.statements[-1] == Assignment.create("Y_2", "EPS_1 * F + EPS_2 + F")


def test_parse_observation_transformation(testdata, load_model_for_test):
    model = load_model_for_test(testdata / 'nonmem' / 'models' / 'pheno_dvid.mod')
    assert model.observation_transformation == {
        sympy.Symbol('Y_1'): sympy.Symbol('Y_1'),
        sympy.Symbol('Y_2'): sympy.Symbol('Y_2'),
    }


def test_validate_eta_names():
    code = '''$PROBLEM
    $INPUT ID DV TIME
    $DATA file.csv IGNORE=@
    $PRED
    Y = THETA(1) + ETA(1) + ERR(1)
    $THETA 0.1
    $OMEGA 0.01
    $SIGMA 1
    '''
    model = Model.parse_model_from_string(code)

    with pytest.raises(ValueError, match='NONMEM does not allow etas named `eta`'):
        add_iiv(model, ['Y'], 'exp', '+', eta_names=['eta'])


def test_ics(load_model_for_test, pheno_path):
    model = load_model_for_test(pheno_path)
    model = set_initial_condition(model, 'CENTRAL', 23)
    pk = model.internals.control_stream.get_pred_pk_record()
    a = str(pk).split('\n')
    assert a[9] == 'A_0(1) = 23'


def test_zo(load_model_for_test, pheno_path):
    model = load_model_for_test(pheno_path)
    model = set_zero_order_input(model, "CENTRAL", 10)
    des = model.internals.control_stream.get_records("DES")[0]
    print(des)
    assert str(des) == "$DES\nDADT(1) = -A(1)*CL/V + 10\n"


def test_des_assignments(load_model_for_test, testdata):
    model = load_model_for_test(testdata / "nonmem" / "models" / "pheno_des_assignments.mod")

    stats = model.statements

    assert stats[3] == Assignment.create("KE", "CL/VC")
    assert stats[4] == Assignment.create("EXTRA", "2 * A_CENTRAL(t)")
