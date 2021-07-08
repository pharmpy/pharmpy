from io import StringIO
from pathlib import Path

import pandas as pd
import pytest
import sympy

import pharmpy.data
import pharmpy.model
import pharmpy.symbols
from pharmpy import Model

tabpath = Path(__file__).parent / 'testdata' / 'nonmem' / 'pheno_real_linbase.tab'
lincorrect = pharmpy.data.read_nonmem_dataset(
    tabpath,
    ignore_character='@',
    colnames=['ID', 'G11', 'G21', 'H11', 'CIPREDI', 'DV', 'PRED', 'RES', 'WRES'],
)


@pytest.mark.parametrize(
    'stem,force_numbering,symbol_name', [('DV', False, 'DV1'), ('X', False, 'X'), ('X', True, 'X1')]
)
def test_create_symbol(testdata, stem, force_numbering, symbol_name):
    model = Model(testdata / 'nonmem' / 'pheno_real.mod')
    symbol = model.create_symbol(stem=stem, force_numbering=force_numbering)

    assert symbol.name == symbol_name


def S(x):
    return pharmpy.symbols.symbol(x)


def test_symbolic_population_prediction(testdata):
    path = testdata / 'nonmem' / 'minimal.mod'
    model = Model(path)

    assert model.symbolic_population_prediction() == S('THETA(1)')


def test_symbolic_individual_prediction(testdata):
    path = testdata / 'nonmem' / 'minimal.mod'
    model = Model(path)

    assert model.symbolic_individual_prediction() == S('THETA(1)') + S('ETA(1)')


def test_population_prediction(testdata):
    path = testdata / 'nonmem' / 'minimal.mod'
    model = Model(path)

    dataset = pd.DataFrame({'ID': [1, 2], 'TIME': [0, 0], 'DV': [3, 4]})
    pred = model.population_prediction(dataset=dataset)

    assert list(pred) == [0.1, 0.1]

    linpath = testdata / 'nonmem' / 'pheno_real_linbase.mod'
    linmod = Model(linpath)
    pred = linmod.population_prediction()

    pd.testing.assert_series_equal(lincorrect['PRED'], pred, rtol=1e-4, check_names=False)


def test_individual_prediction(testdata):
    path = testdata / 'nonmem' / 'minimal.mod'
    model = Model(path)

    dataset = pharmpy.data.read_nonmem_dataset(
        StringIO('1 0 3\n2 0 4\n'), colnames=['ID', 'TIME', 'DV']
    )
    pred = model.individual_prediction(dataset=dataset)

    assert list(pred) == [0.1, 0.1]

    linpath = testdata / 'nonmem' / 'pheno_real_linbase.mod'
    linmod = Model(linpath)
    pred = linmod.individual_prediction(etas=linmod.modelfit_results.individual_estimates)

    pd.testing.assert_series_equal(lincorrect['CIPREDI'], pred, rtol=1e-4, check_names=False)


def test_symbolic_eta_gradient(testdata):
    path = testdata / 'nonmem' / 'minimal.mod'
    model = Model(path)

    assert model.symbolic_eta_gradient() == [1]


def test_symbolic_eps_gradient(testdata):
    path = testdata / 'nonmem' / 'minimal.mod'
    model = Model(path)

    assert model.symbolic_eps_gradient() == [1]


def test_eta_gradient(testdata):
    path = testdata / 'nonmem' / 'minimal.mod'
    model = Model(path)

    dataset = pharmpy.data.read_nonmem_dataset(
        StringIO('1 0 3\n2 0 4\n'), colnames=['ID', 'TIME', 'DV']
    )
    grad = model.eta_gradient(dataset=dataset)
    pd.testing.assert_frame_equal(grad, pd.DataFrame([1.0, 1.0], columns=['dF/dETA(1)']))

    linpath = testdata / 'nonmem' / 'pheno_real_linbase.mod'
    linmod = Model(linpath)
    grad = linmod.eta_gradient()
    pd.testing.assert_series_equal(lincorrect['G11'], grad.iloc[:, 0], rtol=1e-4, check_names=False)
    pd.testing.assert_series_equal(lincorrect['G21'], grad.iloc[:, 1], rtol=1e-4, check_names=False)


def test_eps_gradient(testdata):
    path = testdata / 'nonmem' / 'minimal.mod'
    model = Model(path)

    dataset = pharmpy.data.read_nonmem_dataset(
        StringIO('1 0 3\n2 0 4\n'), colnames=['ID', 'TIME', 'DV']
    )
    grad = model.eps_gradient(dataset=dataset)
    pd.testing.assert_frame_equal(grad, pd.DataFrame([1.0, 1.0], columns=['dY/dEPS(1)']))

    linpath = testdata / 'nonmem' / 'pheno_real_linbase.mod'
    linmod = Model(linpath)
    grad = linmod.eps_gradient(etas=linmod.modelfit_results.individual_estimates)
    pd.testing.assert_series_equal(lincorrect['H11'], grad.iloc[:, 0], rtol=1e-4, check_names=False)


def test_weighted_residuals(testdata):
    linpath = testdata / 'nonmem' / 'pheno_real_linbase.mod'
    linmod = Model(linpath)
    wres = linmod.weighted_residuals()
    pd.testing.assert_series_equal(lincorrect['WRES'], wres, rtol=1e-4, check_names=False)


def test_to_generic_model(testdata):
    path = testdata / 'nonmem' / 'pheno.mod'
    nm_model = Model(path)
    model = nm_model.to_generic_model()
    assert id(model.parameters) != id(nm_model.parameters)
    assert model.parameters == nm_model.parameters
    assert id(model.random_variables) != id(nm_model.random_variables)
    assert model.random_variables == nm_model.random_variables
    assert model.name == nm_model.name
    assert id(model.statements) != id(nm_model.statements)
    assert model.statements == nm_model.statements
    assert type(model) == pharmpy.model.Model


def test_covariates(testdata):
    model = Model(testdata / 'nonmem' / 'pheno_real.mod')
    assert set(model.covariates) == {sympy.Symbol('WGT'), sympy.Symbol('APGR')}
