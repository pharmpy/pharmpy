import pandas as pd
from sympy import Symbol

from pharmpy import Model


def test_create_symbol(testdata):
    model = Model(testdata / 'nonmem' / 'pheno_real.mod')
    symbol = model.create_symbol(stem='ETAT')
    assert symbol.name == 'ETAT1'


def S(x):
    return Symbol(x, real=True)


def test_symbolic_population_prediction(testdata):
    path = testdata / 'nonmem' / 'minimal.mod'
    model = Model(path)

    assert model.symbolic_population_prediction() == S('THETA(1)')


def test_population_prediction(testdata):
    path = testdata / 'nonmem' / 'minimal.mod'
    model = Model(path)

    dataset = pd.DataFrame({'ID': [1, 2], 'TIME': [0, 0], 'DV': [3, 4]})
    pred = model.population_prediction(dataset=dataset)

    assert list(pred) == [0.1, 0.1]


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

    dataset = pd.DataFrame({'ID': [1, 2], 'TIME': [0, 0], 'DV': [3, 4]})
    grad = model.eta_gradient(dataset=dataset)

    pd.testing.assert_frame_equal(grad, pd.DataFrame([1.0, 1.0],
                                                     columns=['dF/dETA(1)']))
