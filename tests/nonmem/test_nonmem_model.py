from pharmpy import Model
from pharmpy.parameter import Parameter


def test_source(pheno_path):
    model = Model(pheno_path)
    assert model.source.code.startswith(';; 1.')


def test_parameters(pheno_path):
    model = Model(pheno_path)
    params = model.parameters
    assert len(params) == 3     # FIXME: This is correct only for now.
    assert model.parameters['THETA(1)'] == Parameter('THETA(1)', 0.00469307, lower=0, upper=1000000)
    assert model.parameters['THETA(2)'] == Parameter('THETA(2)', 1.00916, lower=0, upper=1000000)
    assert model.parameters['THETA(3)'] == Parameter('THETA(3)', 0.1, lower=-0.99, upper=1000000)
