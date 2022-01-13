from io import StringIO
from pathlib import Path

import pandas as pd
import pytest

from pharmpy import Model
from pharmpy.modeling import (
    evaluate_epsilon_gradient,
    evaluate_eta_gradient,
    evaluate_expression,
    evaluate_individual_prediction,
    evaluate_population_prediction,
    evaluate_weighted_residuals,
)
from pharmpy.plugins.nonmem.dataset import read_nonmem_dataset

tabpath = Path(__file__).parent.parent / 'testdata' / 'nonmem' / 'pheno_real_linbase.tab'
lincorrect = read_nonmem_dataset(
    tabpath,
    ignore_character='@',
    colnames=['ID', 'G11', 'G21', 'H11', 'CIPREDI', 'DV', 'PRED', 'RES', 'WRES'],
)


def test_evaluate_expression(testdata):
    model = Model.create_model(testdata / 'nonmem' / 'models' / 'pheno_noifs.mod')
    ser = evaluate_expression(model, 'TVV')
    assert ser[0] == pytest.approx(1.413062)
    assert ser[743] == pytest.approx(1.110262)


def test_evaulate_population_prediction(testdata):
    path = testdata / 'nonmem' / 'minimal.mod'
    model = Model.create_model(path)

    dataset = pd.DataFrame({'ID': [1, 2], 'TIME': [0, 0], 'DV': [3, 4]})
    pred = evaluate_population_prediction(model, dataset=dataset)

    assert list(pred) == [0.1, 0.1]

    linpath = testdata / 'nonmem' / 'pheno_real_linbase.mod'
    linmod = Model.create_model(linpath)
    pred = evaluate_population_prediction(linmod)

    pd.testing.assert_series_equal(lincorrect['PRED'], pred, rtol=1e-4, check_names=False)


def test_evaluate_individual_prediction(testdata):
    path = testdata / 'nonmem' / 'minimal.mod'
    model = Model.create_model(path)

    dataset = read_nonmem_dataset(StringIO('1 0 3\n2 0 4\n'), colnames=['ID', 'TIME', 'DV'])
    pred = evaluate_individual_prediction(model, dataset=dataset)

    assert list(pred) == [0.1, 0.1]

    linpath = testdata / 'nonmem' / 'pheno_real_linbase.mod'
    linmod = Model.create_model(linpath)
    pred = evaluate_individual_prediction(
        model=linmod, etas=linmod.modelfit_results.individual_estimates
    )

    pd.testing.assert_series_equal(lincorrect['CIPREDI'], pred, rtol=1e-4, check_names=False)


def test_evaluate_eta_gradient(testdata):
    path = testdata / 'nonmem' / 'minimal.mod'
    model = Model.create_model(path)

    dataset = read_nonmem_dataset(StringIO('1 0 3\n2 0 4\n'), colnames=['ID', 'TIME', 'DV'])
    grad = evaluate_eta_gradient(model, dataset=dataset)
    pd.testing.assert_frame_equal(grad, pd.DataFrame([1.0, 1.0], columns=['dF/dETA(1)']))

    linpath = testdata / 'nonmem' / 'pheno_real_linbase.mod'
    linmod = Model.create_model(linpath)
    grad = evaluate_eta_gradient(linmod)
    pd.testing.assert_series_equal(lincorrect['G11'], grad.iloc[:, 0], rtol=1e-4, check_names=False)
    pd.testing.assert_series_equal(lincorrect['G21'], grad.iloc[:, 1], rtol=1e-4, check_names=False)


def test_evaluate_epsilon_gradient(testdata):
    path = testdata / 'nonmem' / 'minimal.mod'
    model = Model.create_model(path)

    dataset = read_nonmem_dataset(StringIO('1 0 3\n2 0 4\n'), colnames=['ID', 'TIME', 'DV'])
    grad = evaluate_epsilon_gradient(model, dataset=dataset)
    pd.testing.assert_frame_equal(grad, pd.DataFrame([1.0, 1.0], columns=['dY/dEPS(1)']))

    linpath = testdata / 'nonmem' / 'pheno_real_linbase.mod'
    linmod = Model.create_model(linpath)
    grad = evaluate_epsilon_gradient(linmod, etas=linmod.modelfit_results.individual_estimates)
    pd.testing.assert_series_equal(lincorrect['H11'], grad.iloc[:, 0], rtol=1e-4, check_names=False)


def test_evaluate_weighted_residuals(testdata):
    linpath = testdata / 'nonmem' / 'pheno_real_linbase.mod'
    linmod = Model.create_model(linpath)
    wres = evaluate_weighted_residuals(linmod)
    pd.testing.assert_series_equal(lincorrect['WRES'], wres, rtol=1e-4, check_names=False)
