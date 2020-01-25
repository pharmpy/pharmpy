import sympy
from sympy import Symbol

from pharmpy import Model
from pharmpy.parameter import Parameter


def test_source(pheno_path):
    model = Model(pheno_path)
    assert model.source.code.startswith(';; 1.')


def test_update_inits(pheno_path):
    model = Model(pheno_path)
    model.update_inits()


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
    assert str(thetas[0]) == '$THETA  (0,0.75) ; CL\n'
    assert str(thetas[1]) == '$THETA  (0,0.5) ; V\n'
    assert str(thetas[2]) == '$THETA  (-.99,0.25)\n'
    omegas = model.control_stream.get_records('OMEGA')
    assert str(omegas[0]) == '$OMEGA  DIAGONAL(2)\n 0.1  ;       IVCL\n 0.2  ;        IVV\n'
    sigmas = model.control_stream.get_records('SIGMA')
    assert str(sigmas[0]) == '$SIGMA  0.3\n'


def test_results(pheno_path):
    model = Model(pheno_path)
    assert len(model.modelfit_results) == 1     # A chain of one estimation


def test_minimal(datadir):
    path = datadir / 'minimal.mod'
    model = Model(path)
    assert len(model.statements) == 1
    model.control_stream.get_records('PRED')[0].root.treeprint()
    assert model.statements[0].expression == \
        Symbol('THETA(1)') + Symbol('ETA(1)') + Symbol('ERR(1)')
