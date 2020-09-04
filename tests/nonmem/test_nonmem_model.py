from io import StringIO

import pytest
import sympy
from pyfakefs.fake_filesystem_unittest import Patcher

from pharmpy import Model
from pharmpy.parameter import Parameter
from pharmpy.plugins.nonmem.nmtran_parser import NMTranParser
from pharmpy.statements import Assignment, ModelStatements, ODESystem
from pharmpy.symbols import real


def S(x):
    return real(x)


def test_source(pheno_path):
    model = Model(pheno_path)
    assert model.source.code.startswith(';; 1.')


def test_update_inits(pheno_path):
    model = Model(pheno_path)
    model.update_inits()


def test_detection():
    Model(StringIO("$PROBLEM this"))
    Model(StringIO("   \t$PROBLEM skld fjl"))
    Model(StringIO(" $PRO l907"))


def test_validate(pheno_path):
    model = Model(pheno_path)
    model.validate()


def test_parameters(pheno_path):
    model = Model(pheno_path)
    params = model.parameters
    assert len(params) == 6
    assert model.parameters['THETA(1)'] == Parameter('THETA(1)', 0.00469307, lower=0, upper=1000000)
    assert model.parameters['THETA(2)'] == Parameter('THETA(2)', 1.00916, lower=0, upper=1000000)
    assert model.parameters['THETA(3)'] == Parameter('THETA(3)', 0.1, lower=-0.99, upper=1000000)
    assert model.parameters['OMEGA(1,1)'] == Parameter('OMEGA(1,1)', 0.0309626,
                                                       lower=0, upper=sympy.oo)
    assert model.parameters['OMEGA(2,2)'] == Parameter('OMEGA(2,2)', 0.031128,
                                                       lower=0, upper=sympy.oo)
    assert model.parameters['SIGMA(1,1)'] == Parameter('SIGMA(1,1)', 0.0130865,
                                                       lower=0, upper=sympy.oo)


def test_set_parameters(pheno_path):
    model = Model(pheno_path)
    params = {'THETA(1)': 0.75, 'THETA(2)': 0.5, 'THETA(3)': 0.25,
              'OMEGA(1,1)': 0.1, 'OMEGA(2,2)': 0.2, 'SIGMA(1,1)': 0.3}
    model.parameters = params
    assert model.parameters['THETA(1)'] == Parameter('THETA(1)', 0.75, lower=0, upper=1000000)
    assert model.parameters['THETA(2)'] == Parameter('THETA(2)', 0.5, lower=0, upper=1000000)
    assert model.parameters['THETA(3)'] == Parameter('THETA(3)', 0.25, lower=-0.99, upper=1000000)
    assert model.parameters['OMEGA(1,1)'] == Parameter('OMEGA(1,1)', 0.1,
                                                       lower=0, upper=sympy.oo)
    assert model.parameters['OMEGA(2,2)'] == Parameter('OMEGA(2,2)', 0.2,
                                                       lower=0, upper=sympy.oo)
    assert model.parameters['SIGMA(1,1)'] == Parameter('SIGMA(1,1)', 0.3,
                                                       lower=0, upper=sympy.oo)
    model.update_source()
    thetas = model.control_stream.get_records('THETA')
    assert str(thetas[0]) == '$THETA  (0,0.75) ; PTVCL\n'
    assert str(thetas[1]) == '$THETA  (0,0.5) ; PTVV\n'
    assert str(thetas[2]) == '$THETA  (-.99,0.25)\n'
    omegas = model.control_stream.get_records('OMEGA')
    assert str(omegas[0]) == '$OMEGA  DIAGONAL(2)\n 0.1  ;       IVCL\n 0.2  ;        IVV\n'
    sigmas = model.control_stream.get_records('SIGMA')
    assert str(sigmas[0]) == '$SIGMA  0.3\n'

    model = Model(pheno_path)
    params = model.parameters
    params['THETA(1)'].init = 18
    model.parameters = params
    assert model.parameters['THETA(1)'] == Parameter('THETA(1)', 18, lower=0, upper=1000000)
    assert model.parameters['THETA(2)'] == Parameter('THETA(2)', 1.00916, lower=0, upper=1000000)


@pytest.mark.parametrize('param_new,init_expected,buf_new', [
    (Parameter('COVEFF', 0.2), 0.2, '$THETA  0.2 ; COVEFF'),
    (Parameter('THETA', 0.1), 0.1, '$THETA  0.1 ; THETA'),
    (Parameter('THETA', 0.1, 0, fix=True), 0.1, '$THETA  (0,0.1) FIX ; THETA'),
])
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

    rec_ref = f'$THETA  (0,0.00469307) ; PTVCL\n' \
              f'$THETA  (0,1.00916) ; PTVV\n' \
              f'$THETA  (-.99,.1)\n' \
              f'{buf_new}\n'

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


@pytest.mark.parametrize('statement_new,buf_new', [
    (Assignment(S('CL'), 2), 'CL = 2'),
    (Assignment(S('Y'), S('THETA(4)') + S('THETA(5)')), 'Y = THETA(4) + THETA(5)')
])
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

    rec_ref = f'$PK\nIF(AMT.GT.0) BTIME=TIME\nTAD=TIME-BTIME\n'\
              f'      TVCL=THETA(1)*WGT\n' \
              f'      TVV=THETA(2)*WGT\n' \
              f'IF(APGR.LT.5) TVV=TVV*(1+THETA(3))\n' \
              f'      CL=TVCL*EXP(ETA(1))\n' \
              f'      V=TVV*EXP(ETA(2))\n' \
              f'      S1=V\n' \
              f'{buf_new}\n'

    rec_mod = str(model.control_stream.get_records('PK')[0])

    assert rec_ref == rec_mod


@pytest.mark.parametrize('param_new, statement_new, buf_original, buf_new', [
    (Parameter('X', 0.1), Assignment(S('Y'), S('X') + S('S1')),
     'Y = S1 + X', 'Y = S1 + THETA(4)'),
])
def test_add_parameters_and_statements(pheno_path, param_new, statement_new,
                                       buf_original, buf_new):
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

    rec = '$PK\nIF(AMT.GT.0) BTIME=TIME\nTAD=TIME-BTIME\n' \
          '      TVCL=THETA(1)*WGT\n' \
          '      TVV=THETA(2)*WGT\n' \
          'IF(APGR.LT.5) TVV=TVV*(1+THETA(3))\n' \
          '      CL=TVCL*EXP(ETA(1))\n' \
          '      V=TVV*EXP(ETA(2))\n' \
          '      S1=V\n'

    rec_original = f'{rec}{buf_original}\n'
    rec_new = f'{rec}{buf_new}\n'

    assert str(model.get_pred_pk_record()) == rec_original
    model.update_source()
    assert str(model.get_pred_pk_record()) == rec_new


def test_results(pheno_path):
    model = Model(pheno_path)
    assert len(model.modelfit_results) == 1     # A chain of one estimation


def test_minimal(datadir):
    path = datadir / 'minimal.mod'
    model = Model(path)
    assert len(model.statements) == 1
    assert model.statements[0].expression == \
        real('THETA(1)') + real('ETA(1)') + real('EPS(1)')


def test_copy(datadir):
    path = datadir / 'minimal.mod'
    model = Model(path)
    copy = model.copy()
    assert id(model) != id(copy)
    assert model.statements[0].expression == \
        real('THETA(1)') + real('ETA(1)') + real('EPS(1)')


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
        model = Model('run1.mod')
        model.name = 'run2'
        model.update_individual_estimates(model)
        model.update_source()
        with open('run2_input.phi', 'r') as fp, open('run1.phi') as op:
            assert fp.read() == op.read()
        assert str(model).endswith("""$ESTIMATION METHOD=1 INTERACTION PRINT=1 MCETA=1
$COVARIANCE UNCONDITIONAL
$TABLE      ID TIME AMT WGT APGR IPRED PRED TAD CWRES NPDE NOAPPEND
            NOPRINT ONEHEADER FILE=pheno_real.tab
$ETAS FILE=run2_input.phi""")


@pytest.mark.parametrize('buf_new, len_expected', [
    ('IF(AMT.GT.0) BTIME=TIME\nTAD=TIME-BTIME\n'
     'TVCL=THETA(1)*WGT\nTVV=THETA(2)*WGT\n'
     'IF(APGR.LT.5) TVV=TVV*(1+THETA(3))\nCL=TVCL*EXP(ETA(1))'
     '\nV=TVV*EXP(ETA(2))\nS1=V\nY=A+B', 9),
    ('IF(AMT.GT.0) BTIME=TIME\nTAD=TIME-BTIME\n'
     'TVCL=THETA(1)*WGT\nTVV=THETA(2)*WGT\n'
     'IF(APGR.LT.5) TVV=TVV*(1+THETA(3))\nCL=TVCL*EXP(ETA(1))'
     '\nV=TVV*EXP(ETA(2))\nS1=2*V', 8),
])
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
    assert str(model).split('\n')[13] == '      V = TVV*EXP(ETA(1))'
