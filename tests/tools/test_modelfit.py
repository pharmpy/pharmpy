from pharmpy import Model
from pharmpy.tools.modelfit.results import calculate_results


def test_modelfit(testdata):
    model = Model(testdata / 'nonmem' / 'pheno_real.mod')
    assert model


def test_aggregate(testdata):
    model = Model(testdata / 'nonmem' / 'pheno.mod')
    res = calculate_results([model, model])
    pe = res.parameter_estimates
    assert len(pe) == 2
    assert list(pe.index) == ['pheno', 'pheno']
    assert list(pe.columns) == ['THETA(1)', 'THETA(2)', 'OMEGA(1,1)', 'OMEGA(2,2)', 'SIGMA(1,1)']
