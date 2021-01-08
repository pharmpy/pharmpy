import re
from io import StringIO

import pytest
import sympy
from pyfakefs.fake_filesystem_unittest import Patcher

from pharmpy import Model
from pharmpy.config import ConfigurationContext
from pharmpy.model import ModelSyntaxError
from pharmpy.parameter import Parameter
from pharmpy.plugins.nonmem import conf
from pharmpy.plugins.nonmem.nmtran_parser import NMTranParser
from pharmpy.random_variables import VariabilityLevel
from pharmpy.statements import Assignment, ModelStatements, ODESystem
from pharmpy.symbols import symbol


def S(x):
    return symbol(x)


def test_source(pheno_path):
    model = Model(pheno_path)
    assert model.source.code.startswith('$PROBLEM PHENOBARB')


def test_update_inits(pheno_path):
    model = Model(pheno_path)
    model.update_inits()

    with ConfigurationContext(conf, parameter_names='comment'):
        model = Model(pheno_path)
        model.update_inits()
        model.update_source()


def test_empty_ext_file(testdata):
    # assert existing but empty ext-file does not give modelfit_results
    model = Model(
        testdata / 'nonmem' / 'modelfit_results' / 'onePROB' / 'noESTwithSIM' / 'onlysim.mod'
    )
    assert model.source.path.with_suffix('.ext').exists() is True
    assert model.modelfit_results is None


def test_detection():
    Model(StringIO("$PROBLEM this"))
    Model(StringIO("   \t$PROBLEM skld fjl"))
    Model(StringIO(" $PRO l907"))


def test_validate(pheno_path):
    model = Model(pheno_path)
    model.validate()

    model = Model(StringIO("$PROBLEM this\n$SIZES LIM1=3000"))
    with pytest.raises(ModelSyntaxError):
        model.validate()


def test_parameters(pheno_path):
    model = Model(pheno_path)
    params = model.parameters
    assert len(params) == 6
    assert model.parameters['THETA(1)'] == Parameter('THETA(1)', 0.00469307, lower=0, upper=1000000)
    assert model.parameters['THETA(2)'] == Parameter('THETA(2)', 1.00916, lower=0, upper=1000000)
    assert model.parameters['THETA(3)'] == Parameter('THETA(3)', 0.1, lower=-0.99, upper=1000000)
    assert model.parameters['OMEGA(1,1)'] == Parameter(
        'OMEGA(1,1)', 0.0309626, lower=0, upper=sympy.oo
    )
    assert model.parameters['OMEGA(2,2)'] == Parameter(
        'OMEGA(2,2)', 0.031128, lower=0, upper=sympy.oo
    )
    assert model.parameters['SIGMA(1,1)'] == Parameter(
        'SIGMA(1,1)', 0.013241, lower=0, upper=sympy.oo
    )


def test_set_parameters(pheno_path):
    model = Model(pheno_path)
    params = {
        'THETA(1)': 0.75,
        'THETA(2)': 0.5,
        'THETA(3)': 0.25,
        'OMEGA(1,1)': 0.1,
        'OMEGA(2,2)': 0.2,
        'SIGMA(1,1)': 0.3,
    }
    model.parameters = params
    assert model.parameters['THETA(1)'] == Parameter('THETA(1)', 0.75, lower=0, upper=1000000)
    assert model.parameters['THETA(2)'] == Parameter('THETA(2)', 0.5, lower=0, upper=1000000)
    assert model.parameters['THETA(3)'] == Parameter('THETA(3)', 0.25, lower=-0.99, upper=1000000)
    assert model.parameters['OMEGA(1,1)'] == Parameter('OMEGA(1,1)', 0.1, lower=0, upper=sympy.oo)
    assert model.parameters['OMEGA(2,2)'] == Parameter('OMEGA(2,2)', 0.2, lower=0, upper=sympy.oo)
    assert model.parameters['SIGMA(1,1)'] == Parameter('SIGMA(1,1)', 0.3, lower=0, upper=sympy.oo)
    model.update_source()
    thetas = model.control_stream.get_records('THETA')
    assert str(thetas[0]) == '$THETA (0,0.75) ; PTVCL\n'
    assert str(thetas[1]) == '$THETA (0,0.5) ; PTVV\n'
    assert str(thetas[2]) == '$THETA (-.99,0.25)\n'
    omegas = model.control_stream.get_records('OMEGA')
    assert str(omegas[0]) == '$OMEGA DIAGONAL(2)\n 0.1  ;       IVCL\n 0.2  ;        IVV\n\n'
    sigmas = model.control_stream.get_records('SIGMA')
    assert str(sigmas[0]) == '$SIGMA 0.3\n'

    model = Model(pheno_path)
    params = model.parameters
    params['THETA(1)'].init = 18
    model.parameters = params
    assert model.parameters['THETA(1)'] == Parameter('THETA(1)', 18, lower=0, upper=1000000)
    assert model.parameters['THETA(2)'] == Parameter('THETA(2)', 1.00916, lower=0, upper=1000000)


def test_adjust_iovs(testdata):
    model = Model(
        testdata / 'nonmem' / 'modelfit_results' / 'onePROB' / 'multEST' / 'noSIM' / 'withBayes.mod'
    )
    model.parameters
    rvs = model.random_variables

    assert rvs[0].variability_level == VariabilityLevel.IIV
    assert rvs[3].variability_level == VariabilityLevel.IOV
    assert rvs[4].variability_level == VariabilityLevel.IOV
    assert rvs[6].variability_level == VariabilityLevel.IOV

    model = Model(testdata / 'nonmem' / 'qa' / 'iov.mod')
    rvs = model.random_variables
    assert rvs[0].variability_level == VariabilityLevel.IIV
    assert rvs[1].variability_level == VariabilityLevel.IIV
    assert rvs[2].variability_level == VariabilityLevel.IOV
    assert rvs[3].variability_level == VariabilityLevel.IOV
    assert rvs[4].variability_level == VariabilityLevel.IOV
    assert rvs[5].variability_level == VariabilityLevel.IOV


@pytest.mark.parametrize(
    'param_new,init_expected,buf_new',
    [
        (Parameter('COVEFF', 0.2), 0.2, '$THETA  0.2 ; COVEFF'),
        (Parameter('THETA', 0.1), 0.1, '$THETA  0.1 ; THETA'),
        (Parameter('THETA', 0.1, 0, fix=True), 0.1, '$THETA  (0,0.1) FIX ; THETA'),
        (Parameter('RUV_prop', 0.1), 0.1, '$THETA  0.1 ; RUV_prop'),
    ],
)
def test_add_parameters(pheno_path, param_new, init_expected, buf_new):
    model = Model(pheno_path)
    pset = model.parameters

    assert len(pset) == 6

    pset.add(param_new)
    model.parameters = pset
    model.update_source()

    assert len(pset) == 7
    assert model.parameters[param_new.name].init == init_expected

    parser = NMTranParser()
    stream = parser.parse(str(model))

    assert str(model.control_stream) == str(stream)

    rec_ref = (
        f'$THETA (0,0.00469307) ; PTVCL\n'
        f'$THETA (0,1.00916) ; PTVV\n'
        f'$THETA (-.99,.1)\n'
        f'{buf_new}\n'
    )

    rec_mod = ''
    for rec in model.control_stream.get_records('THETA'):
        rec_mod += str(rec)

    assert rec_ref == rec_mod


def test_add_two_parameters(pheno_path):
    model = Model(pheno_path)
    pset = model.parameters

    assert len(pset) == 6

    param_1 = Parameter('COVEFF1', 0.2)
    param_2 = Parameter('COVEFF2', 0.1)
    pset.add(param_1)
    pset.add(param_2)
    model.parameters = pset
    model.update_source()

    assert len(pset) == 8
    assert model.parameters[param_1.name].init == 0.2
    assert model.parameters[param_2.name].init == 0.1


@pytest.mark.parametrize(
    'statement_new,buf_new',
    [
        (Assignment(S('CL'), sympy.Integer(2)), 'CL = 2'),
        (Assignment(S('Y'), S('THETA(4)') + S('THETA(5)')), 'Y = THETA(4) + THETA(5)'),
    ],
)
def test_add_statements(pheno_path, statement_new, buf_new):
    model = Model(pheno_path)
    sset = model.statements
    assert len(sset) == 15

    # Insert new statement before ODE system.
    new_sset = ModelStatements()
    for s in sset:
        if isinstance(s, ODESystem):
            new_sset.append(statement_new)
        new_sset.append(s)

    model.statements = new_sset
    model.update_source()

    assert len(model.statements) == 16

    parser = NMTranParser()
    stream = parser.parse(str(model))

    assert str(model.control_stream) == str(stream)

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

    rec_mod = str(model.control_stream.get_records('PK')[0])

    assert rec_ref == rec_mod


@pytest.mark.parametrize(
    'param_new, statement_new, buf_new',
    [
        (Parameter('X', 0.1), Assignment(S('Y'), S('X') + S('S1')), 'Y = S1 + THETA(4)'),
    ],
)
def test_add_parameters_and_statements(pheno_path, param_new, statement_new, buf_new):
    model = Model(pheno_path)

    pset = model.parameters
    pset.add(param_new)
    model.parameters = pset

    sset = model.statements

    # Insert new statement before ODE system.
    new_sset = ModelStatements()
    for s in sset:
        if isinstance(s, ODESystem):
            new_sset.append(statement_new)
        new_sset.append(s)

    model.statements = new_sset
    model.update_source()

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

    assert str(model.get_pred_pk_record()) == rec


def test_error_model(pheno_path):
    model_prop = Model(pheno_path)
    assert model_prop.error_model == 'PROP'

    model_add_str = re.sub(
        r'Y=F\+W\*EPS\(1\)',
        'Y=F+EPS(1)',
        str(model_prop),
    )
    model_add = Model(StringIO(model_add_str))
    assert model_add.error_model == 'ADD'

    model_add_prop_str = re.sub(
        r'Y=F\+W\*EPS\(1\)',
        'Y=EPS(1)*F+EPS(2)+F',
        str(model_prop),
    )
    model_add_prop = Model(StringIO(model_add_prop_str))
    assert model_add_prop.error_model == 'ADD_PROP'

    model_none_str = re.sub(
        r'Y=F\+W\*EPS\(1\)',
        'Y=F',
        str(model_prop),
    )
    model_none = Model(StringIO(model_none_str))
    assert model_none.error_model == 'NONE'


@pytest.mark.parametrize('rv_new,buf_new', [(Parameter('omega', 0.1), '$OMEGA  0.1')])
def test_add_random_variables(pheno_path, rv_new, buf_new):
    model = Model(pheno_path)
    rvs = model.random_variables
    pset = model.parameters

    eta = sympy.stats.Normal('eta_new', 0, sympy.sqrt(S(rv_new.name)))
    eta.variability_level = VariabilityLevel.IIV

    rvs.add(eta)
    pset.add(rv_new)

    model.random_variables = rvs
    model.parameters = pset

    model.update_source()

    rec_ref = (
        f'$OMEGA DIAGONAL(2)\n'
        f' 0.0309626  ;       IVCL\n'
        f' 0.031128  ;        IVV\n\n'
        f'{buf_new} ; omega\n'
    )

    rec_mod = ''
    for rec in model.control_stream.get_records('OMEGA'):
        rec_mod += str(rec)

    assert rec_mod == rec_ref

    rv = model.random_variables['eta_new']

    assert rv.pspace.distribution.mean == 0
    assert rv.pspace.distribution.std ** 2 == rv_new.symbol


def test_add_random_variables_and_statements(pheno_path):
    model = Model(pheno_path)

    rvs = model.random_variables
    pset = model.parameters

    eta = sympy.stats.Normal('ETA_NEW', 0, sympy.sqrt(S('omega')))
    eta.variability_level = VariabilityLevel.IIV
    rvs.add(eta)
    pset.add(Parameter('omega', 0.1))

    eps = sympy.stats.Normal('EPS_NEW', 0, sympy.sqrt(S('sigma')))
    eps.variability_level = VariabilityLevel.RUV
    rvs.add(eps)
    pset.add(Parameter('sigma', 0.1))

    model.random_variables = rvs
    model.parameters = pset

    sset = model.get_pred_pk_record().statements

    statement_new = Assignment(S('X'), 1 + S(eps.name) + S(eta.name))
    sset.append(statement_new)
    model.get_pred_pk_record().statements = sset

    model.update_source()

    assert str(model.get_pred_pk_record()).endswith('X = 1 + ETA(3) + EPS(2)\n\n')


def test_results(pheno_path):
    model = Model(pheno_path)
    assert len(model.modelfit_results) == 0
    assert bool(model.modelfit_results) is True  # results loaded on access
    assert len(model.modelfit_results) == 1  # A chain of one estimation


def test_minimal(datadir):
    path = datadir / 'minimal.mod'
    model = Model(path)
    assert len(model.statements) == 1
    assert model.statements[0].expression == symbol('THETA(1)') + symbol('ETA(1)') + symbol(
        'EPS(1)'
    )


def test_copy(datadir):
    path = datadir / 'minimal.mod'
    model = Model(path)
    copy = model.copy()
    assert id(model) != id(copy)
    assert model.statements[0].expression == symbol('THETA(1)') + symbol('ETA(1)') + symbol(
        'EPS(1)'
    )


def test_initial_individual_estimates(datadir):
    path = datadir / 'minimal.mod'
    model = Model(path)
    assert model.initial_individual_estimates is None

    path = datadir / 'pheno_etas.mod'
    model = Model(path)
    inits = model.initial_individual_estimates
    assert len(inits) == 59
    assert len(inits.columns) == 2
    assert inits['ETA(1)'][2] == -0.166321


def test_update_individual_estimates(datadir):
    with Patcher(additional_skip_names=['pkgutil']) as patcher:
        fs = patcher.fs
        fs.add_real_file(datadir / 'pheno_real.mod', target_path='run1.mod')
        fs.add_real_file(datadir / 'pheno_real.phi', target_path='run1.phi')
        fs.add_real_file(datadir / 'pheno_real.lst', target_path='run1.lst')
        fs.add_real_file(datadir / 'pheno_real.ext', target_path='run1.ext')
        fs.add_real_file(datadir / 'pheno.dta', target_path='pheno.dta')
        model = Model('run1.mod')
        model.name = 'run2'
        model.update_individual_estimates(model)
        model.update_source()
        with open('run2_input.phi', 'r') as fp, open('run1.phi') as op:
            assert fp.read() == op.read()
        assert str(model).endswith(
            """$ESTIMATION METHOD=1 INTERACTION MCETA=1
$COVARIANCE UNCONDITIONAL
$TABLE ID TIME DV AMT WGT APGR IPRED PRED RES TAD CWRES NPDE NOAPPEND
       NOPRINT ONEHEADER FILE=sdtab2
$ETAS FILE=run2_input.phi"""
        )


@pytest.mark.parametrize(
    'buf_new, len_expected',
    [
        (
            'IF(AMT.GT.0) BTIME=TIME\nTAD=TIME-BTIME\n'
            'TVCL=THETA(1)*WGT\nTVV=THETA(2)*WGT\n'
            'IF(APGR.LT.5) TVV=TVV*(1+THETA(3))\nCL=TVCL*EXP(ETA(1))'
            '\nV=TVV*EXP(ETA(2))\nS1=V\nY=A+B',
            9,
        ),
        (
            'IF(AMT.GT.0) BTIME=TIME\nTAD=TIME-BTIME\n'
            'TVCL=THETA(1)*WGT\nTVV=THETA(2)*WGT\n'
            'IF(APGR.LT.5) TVV=TVV*(1+THETA(3))\nCL=TVCL*EXP(ETA(1))'
            '\nV=TVV*EXP(ETA(2))\nS1=2*V',
            8,
        ),
    ],
)
def test_statements_setter(pheno_path, buf_new, len_expected):
    model = Model(pheno_path)

    parser = NMTranParser()
    statements_new = parser.parse(f'$PRED\n{buf_new}').records[0].statements

    assert len(model.statements) == 15
    assert len(statements_new) == len_expected

    model.statements = statements_new

    assert len(model.statements) == len_expected
    assert model.statements == statements_new


def test_deterministic_theta_comments(pheno_path):
    model = Model(pheno_path)

    no_option = 0
    for theta_record in model.control_stream.get_records('THETA'):
        no_option += len(theta_record.root.all('option'))

    assert no_option == 0


def test_remove_eta(pheno_path):
    model = Model(pheno_path)
    rvs = model.random_variables
    eta1 = rvs['ETA(1)']
    rvs.discard(eta1)
    model.update_source()
    assert str(model).split('\n')[12] == 'V = TVV*EXP(ETA(1))'


def test_symbol_names_in_comment(pheno_path):
    with ConfigurationContext(conf, parameter_names='comment'):
        model = Model(pheno_path)
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
        model = Model(StringIO(code))
        with pytest.warns(UserWarning):
            assert model.parameters.names == ['THETA(1)', 'OMEGA(1,1)', 'SIGMA(1,1)']


def test_clashing_parameter_names(datadir):
    with ConfigurationContext(conf, parameter_names='comment'):
        model = Model(datadir / 'pheno_clashing_symbols.mod')
        with pytest.warns(UserWarning):
            model.statements
        assert model.parameters.names == ['THETA(1)', 'TVV', 'IVCL', 'OMEGA(2,2)', 'SIGMA(1,1)']

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
        model = Model(StringIO(code))
        with pytest.warns(UserWarning):
            assert model.parameters.names == ['TV', 'OMEGA(1,1)', 'SIGMA(1,1)']

        code = """$PROBLEM base model
$INPUT ID DV TIME
$DATA file.csv IGNORE=@

$PRED
Y = THETA(1) + THETA(2)

$THETA 0.1  ; TV
$THETA 0.1  ; TV
$ESTIMATION METHOD=1 INTER MAXEVALS=9990 PRINT=2 POSTHOC
"""
        model = Model(StringIO(code))
        with pytest.warns(UserWarning):
            assert model.parameters.names == ['TV', 'THETA(2)']


def test_dv_symbol(pheno_path):
    model = Model(pheno_path)
    assert model.dependent_variable_symbol.name == 'Y'


def test_insert_unknown_record(pheno_path):
    model = Model(pheno_path)
    model.control_stream.insert_record('$TRIREME one')
    assert str(model).split('\n')[-1] == '$TRIREME one'

    model.control_stream.insert_record('\n$OA two')
    assert str(model).split('\n')[-1] == '$OA two'


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
    model = Model(StringIO(code))
    rvs = model.random_variables
    assert len(rvs) == 11
