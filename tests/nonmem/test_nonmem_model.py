from io import StringIO

import pytest
import sympy
from pyfakefs.fake_filesystem_unittest import Patcher

from pharmpy import Model
from pharmpy.config import ConfigurationContext
from pharmpy.estimation import EstimationMethod
from pharmpy.model import ModelSyntaxError
from pharmpy.modeling import add_iiv, explicit_odes, zero_order_elimination
from pharmpy.parameter import Parameter
from pharmpy.plugins.nonmem import conf
from pharmpy.plugins.nonmem.nmtran_parser import NMTranParser
from pharmpy.random_variables import RandomVariable
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

    with ConfigurationContext(conf, parameter_names=['comment', 'basic']):
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

    assert rvs[0].level == 'IIV'
    assert rvs[3].level == 'IOV'
    assert rvs[4].level == 'IOV'
    assert rvs[6].level == 'IOV'

    model = Model(testdata / 'nonmem' / 'qa' / 'iov.mod')
    rvs = model.random_variables
    assert rvs[0].level == 'IIV'
    assert rvs[1].level == 'IIV'
    assert rvs[2].level == 'IOV'
    assert rvs[3].level == 'IOV'
    assert rvs[4].level == 'IOV'
    assert rvs[5].level == 'IOV'


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

    pset.append(param_new)
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
    pset.append(param_1)
    pset.append(param_2)
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
    pset.append(param_new)
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


@pytest.mark.parametrize('rv_new, buf_new', [(Parameter('omega', 0.1), '$OMEGA  0.1')])
def test_add_random_variables(pheno_path, rv_new, buf_new):
    model = Model(pheno_path)
    rvs = model.random_variables
    pset = model.parameters

    eta = RandomVariable.normal('eta_new', 'iiv', 0, S(rv_new.name))

    rvs.append(eta)
    pset.append(rv_new)

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

    assert rv.sympy_rv.pspace.distribution.mean == 0
    assert (rv.sympy_rv.pspace.distribution.std ** 2).name == 'omega'


def test_add_random_variables_and_statements(pheno_path):
    model = Model(pheno_path)

    rvs = model.random_variables
    pset = model.parameters

    eta = RandomVariable.normal('ETA_NEW', 'iiv', 0, S('omega'))
    rvs.append(eta)
    pset.append(Parameter('omega', 0.1))

    eps = RandomVariable.normal('EPS_NEW', 'ruv', 0, S('sigma'))
    rvs.append(eps)
    pset.append(Parameter('sigma', 0.1))

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
    del rvs[eta1]
    model.update_source()
    assert str(model).split('\n')[12] == 'V = TVV*EXP(ETA(1))'


def test_symbol_names_in_comment(pheno_path):
    with ConfigurationContext(conf, parameter_names=['comment', 'basic']):
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


def test_symbol_names_in_abbr(testdata):
    with ConfigurationContext(conf, parameter_names=['abbr', 'basic']):
        model = Model(testdata / 'nonmem' / 'pheno_abbr.mod')
        sset, pset, rvs = model.statements, model.parameters, model.random_variables

        assert S('THETA_CL') in sset.find_assignment('THETA_CL', is_symbol=False).free_symbols
        assert 'THETA_CL' in pset.names
        assert 'ETA_CL' in [eta.name for eta in rvs.etas]


@pytest.mark.parametrize(
    'parameter_names, assignments, params, etas',
    [
        (
            ['abbr', 'comment', 'basic'],
            [
                Assignment(S('CL'), S('THETA_CL') * sympy.exp(S('ETA_CL'))),
                Assignment(S('V'), S('TVV') * sympy.exp(S('ETA(2)'))),
            ],
            ['THETA_CL', 'TVV', 'IVCL', 'OMEGA(2,2)'],
            ['ETA_CL', 'ETA(2)'],
        ),
        (
            ['comment', 'abbr', 'basic'],
            [
                Assignment(S('CL'), S('TVCL') * sympy.exp(S('ETA_CL'))),
                Assignment(S('V'), S('TVV') * sympy.exp(S('ETA(2)'))),
            ],
            ['TVCL', 'TVV', 'IVCL', 'OMEGA(2,2)'],
            ['ETA_CL', 'ETA(2)'],
        ),
        (
            ['abbr', 'basic'],
            [
                Assignment(S('CL'), S('THETA_CL') * sympy.exp(S('ETA_CL'))),
                Assignment(S('V'), S('THETA(2)') * sympy.exp(S('ETA(2)'))),
            ],
            ['THETA_CL', 'THETA(2)', 'OMEGA(1,1)', 'OMEGA(2,2)'],
            ['ETA_CL', 'ETA(2)'],
        ),
        (
            ['basic'],
            [
                Assignment(S('CL'), S('THETA(1)') * sympy.exp(S('ETA(1)'))),
                Assignment(S('V'), S('THETA(2)') * sympy.exp(S('ETA(2)'))),
            ],
            ['THETA(1)', 'THETA(2)', 'OMEGA(1,1)', 'OMEGA(2,2)'],
            ['ETA(1)', 'ETA(2)'],
        ),
        (
            ['basic', 'comment'],
            [
                Assignment(S('CL'), S('THETA(1)') * sympy.exp(S('ETA(1)'))),
                Assignment(S('V'), S('THETA(2)') * sympy.exp(S('ETA(2)'))),
            ],
            ['THETA(1)', 'THETA(2)', 'OMEGA(1,1)', 'OMEGA(2,2)'],
            ['ETA(1)', 'ETA(2)'],
        ),
        (
            ['basic', 'abbr'],
            [
                Assignment(S('CL'), S('THETA(1)') * sympy.exp(S('ETA(1)'))),
                Assignment(S('V'), S('THETA(2)') * sympy.exp(S('ETA(2)'))),
            ],
            ['THETA(1)', 'THETA(2)', 'OMEGA(1,1)', 'OMEGA(2,2)'],
            ['ETA(1)', 'ETA(2)'],
        ),
    ],
)
def test_symbol_names_priority(testdata, parameter_names, assignments, params, etas):
    with ConfigurationContext(conf, parameter_names=parameter_names):
        model = Model(testdata / 'nonmem' / 'pheno_abbr_comments.mod')
        sset, pset, rvs = model.statements, model.parameters, model.random_variables

        assert all(str(a) in [str(s) for s in sset] for a in assignments)
        assert all(p in pset.names for p in params)
        assert all(eta in [rv.name for rv in rvs] for eta in etas)


def test_clashing_parameter_names(datadir):
    with ConfigurationContext(conf, parameter_names=['comment', 'basic']):
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


def test_missing_parameter_names_settings(pheno_path):
    with ConfigurationContext(conf, parameter_names=['comment']):
        model = Model(pheno_path)
        with pytest.raises(ValueError):
            model.statements


def test_abbr_write(pheno_path):
    with ConfigurationContext(conf, write_etas_in_abbr=True):
        model = Model(pheno_path)
        add_iiv(model, 'S1', 'add')
        model.update_source()

        assert 'ETA(S1)' in str(model)
        assert 'ETA_S1' in [rv.name for rv in model.random_variables]
        assert S('ETA_S1') in model.statements.free_symbols

        model.update_source()

        assert 'ETA(S1)' in str(model)
        assert 'ETA_S1' in [rv.name for rv in model.random_variables]
        assert S('ETA_S1') in model.statements.free_symbols

        model = Model(pheno_path)
        add_iiv(model, 'S1', 'add', eta_names='new_name')

        with pytest.warns(UserWarning, match='Not valid format of name new_name'):
            model.update_source()
            assert 'ETA(3)' in str(model)


def test_abbr_read_write(pheno_path):
    with ConfigurationContext(
        conf, parameter_names=['abbr', 'comment', 'basic'], write_etas_in_abbr=True
    ):
        model_write = Model(pheno_path)
        add_iiv(model_write, 'S1', 'add')
        model_write.update_source()
        model_read = Model(StringIO(str(model_write)))
        model_read.source.path = pheno_path

        assert str(model_read) == str(model_write)
        assert model_read.statements == model_write.statements
        assert not (
            model_read.random_variables - model_write.random_variables
        )  # Different order due to renaming in read


def test_dv_symbol(pheno_path):
    model = Model(pheno_path)
    assert model.dependent_variable.name == 'Y'


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


@pytest.mark.parametrize(
    'model_path, transformation',
    [
        ('nonmem/pheno.mod', explicit_odes),
        ('nonmem/pheno.mod', zero_order_elimination),
        ('nonmem/modeling/pheno_advan1_zero_order.mod', explicit_odes),
        ('nonmem/modeling/pheno_advan5_depot.mod', explicit_odes),
    ],
)
def test_des(testdata, model_path, transformation):
    model_ref = Model(testdata / model_path)
    transformation(model_ref)
    model_ref.update_source()

    model_des = Model(StringIO(str(model_ref)))
    model_des.source.path = model_ref.source.path  # To be able to find dataset

    assert model_ref.statements.ode_system == model_des.statements.ode_system


@pytest.mark.parametrize(
    'estcode,correct',
    [
        ('$ESTIMATION METH=COND INTERACTION', [EstimationMethod('foce', interaction=True)]),
        ('$ESTIMATION INTER METH=COND', [EstimationMethod('foce', interaction=True)]),
        ('$ESTM METH=1 INTERACTION', [EstimationMethod('foce', interaction=True)]),
        ('$ESTIM METH=1', [EstimationMethod('foce', interaction=False)]),
        ('$ESTIMA METH=0', [EstimationMethod('fo', interaction=False)]),
        ('$ESTIMA METH=ZERO', [EstimationMethod('fo', interaction=False)]),
        ('$ESTIMA INTER', [EstimationMethod('fo', interaction=True)]),
        ('$ESTIMA INTER\n$COV', [EstimationMethod('fo', interaction=True, cov=True)]),
        (
            '$ESTIMA METH=COND INTER\n$EST METH=COND',
            [
                EstimationMethod('foce', interaction=True),
                EstimationMethod('foce', interaction=False),
            ],
        ),
        ('$ESTIMATION METH=SAEM', [EstimationMethod('saem', interaction=False)]),
    ],
)
def test_estimation_steps_getter(estcode, correct):
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
    model = Model(StringIO(code))
    assert model.estimation_steps == correct


@pytest.mark.parametrize(
    'estcode,method_ref,key_ref,value_ref',
    [
        ('$ESTIMA MCETA=200', 'FO', 'MCETA', '200'),
        ('$ESTIMATION METHOD=1 MAXEVAL=9999', 'FOCE', 'MAXEVAL', '9999'),
        ('$ESTIMATION METHOD=SAEM MAXEVAL=9999', 'SAEM', 'MAXEVAL', '9999'),
    ],
)
def test_estimation_steps_getter_options(estcode, method_ref, key_ref, value_ref):
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
    model = Model(StringIO(code))
    assert model.estimation_steps[0].method == method_ref
    assert model.estimation_steps[0].options[0].key == key_ref
    assert model.estimation_steps[0].options[0].value == value_ref


@pytest.mark.parametrize(
    'estcode,method,inter,cov,rec_ref',
    [
        ('$EST METH=COND INTER', 'fo', True, False, '$ESTIMATION METHOD=ZERO INTER'),
        ('$EST METH=COND INTER', 'fo', True, True, '$COVARIANCE'),
        (
            '$EST METH=COND INTER MAXEVAL=99999',
            'fo',
            True,
            False,
            '$ESTIMATION METHOD=ZERO INTER MAXEVAL=99999',
        ),
        (
            '$EST METH=COND INTER POSTHOC',
            'fo',
            True,
            False,
            '$ESTIMATION METHOD=ZERO INTER POSTHOC',
        ),
        ('$EST METH=COND INTER', 'saem', True, False, '$ESTIMATION METHOD=SAEM INTER'),
    ],
)
def test_estimation_steps_setter(estcode, method, inter, cov, rec_ref):
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
    model = Model(StringIO(code))
    model.estimation_steps[0].method = method
    model.estimation_steps[0].interaction = inter
    model.estimation_steps[0].cov = cov
    model.update_source()
    assert str(model).split('\n')[-2] == rec_ref


def test_add_estimation_step():
    code = '''$PROBLEM base model
$INPUT ID DV TIME
$DATA file.csv IGNORE=@
$PRED
Y = THETA(1) + ETA(1) + ERR(1)
$THETA 0.1
$OMEGA 0.01
$SIGMA 1
$EST METH=COND INTER
'''
    model = Model(StringIO(code))
    model.add_estimation_step('IMP', True, False, {'saddle_reset': 1})
    model.update_source()
    assert str(model).split('\n')[-2] == '$ESTIMATION METHOD=IMP INTER SADDLE_RESET=1'
    model.add_estimation_step('SAEM', True, False, idx=0)
    model.update_source()
    assert str(model).split('\n')[-4] == '$ESTIMATION METHOD=SAEM INTER'


def test_remove_estimation_step():
    code = '''$PROBLEM base model
$INPUT ID DV TIME
$DATA file.csv IGNORE=@
$PRED
Y = THETA(1) + ETA(1) + ERR(1)
$THETA 0.1
$OMEGA 0.01
$SIGMA 1
$EST METH=COND INTER
'''
    model = Model(StringIO(code))
    model.remove_estimation_step(0)
    assert not model.estimation_steps
    model.update_source()
    assert str(model).split('\n')[-2] == '$SIGMA 1'


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
    with ConfigurationContext(conf, parameter_names=['comment', 'basic']):
        with pytest.warns(UserWarning):
            model = Model(StringIO(code))
            model.update_source()
