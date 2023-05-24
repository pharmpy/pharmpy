from io import StringIO
from pathlib import Path

import pandas as pd
import pytest

from pharmpy.model.external.nonmem.dataset import read_nonmem_dataset
from pharmpy.modeling import (
    evaluate_epsilon_gradient,
    evaluate_eta_gradient,
    evaluate_expression,
    evaluate_individual_prediction,
    evaluate_population_prediction,
    evaluate_weighted_residuals,
)
from pharmpy.tools import read_modelfit_results

tabpath = Path(__file__).resolve().parent.parent / 'testdata' / 'nonmem' / 'pheno_real_linbase.tab'
lincorrect = read_nonmem_dataset(
    tabpath,
    ignore_character='@',
    colnames=['ID', 'G11', 'G21', 'H11', 'CIPREDI', 'DV', 'PRED', 'RES', 'WRES'],
)


def test_evaluate_expression(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'models' / 'pheno_noifs.mod')
    res = read_modelfit_results(testdata / 'nonmem' / 'models' / 'pheno_noifs.mod')
    ser = evaluate_expression(model, 'TVV', res.parameter_estimates)
    assert ser[0] == pytest.approx(1.413062)
    assert ser[743] == pytest.approx(1.110262)


def test_evaluate_population_prediction(load_model_for_test, testdata):
    path = testdata / 'nonmem' / 'minimal.mod'
    model = load_model_for_test(path)

    dataset = pd.DataFrame({'ID': [1, 2], 'TIME': [0, 0], 'DV': [3, 4]})
    pred = evaluate_population_prediction(model, dataset=dataset)

    assert list(pred) == [0.1, 0.1]

    linpath = testdata / 'nonmem' / 'pheno_real_linbase.mod'
    linmod = load_model_for_test(linpath)
    pred = evaluate_population_prediction(linmod)

    pd.testing.assert_series_equal(lincorrect['PRED'], pred, rtol=1e-4, check_names=False)


def test_evaluate_individual_prediction(load_model_for_test, testdata):
    path = testdata / 'nonmem' / 'minimal.mod'
    model = load_model_for_test(path)

    dataset = read_nonmem_dataset(StringIO('1 0 3\n2 0 4\n'), colnames=['ID', 'TIME', 'DV'])
    pred = evaluate_individual_prediction(model, dataset=dataset)

    assert list(pred) == [0.1, 0.1]

    linpath = testdata / 'nonmem' / 'pheno_real_linbase.mod'
    linmod = load_model_for_test(linpath)
    res = read_modelfit_results(linpath)
    pred = evaluate_individual_prediction(model=linmod, etas=res.individual_estimates)

    pd.testing.assert_series_equal(lincorrect['CIPREDI'], pred, rtol=1e-4, check_names=False)


def test_evaluate_eta_gradient(load_model_for_test, testdata):
    path = testdata / 'nonmem' / 'minimal.mod'
    model = load_model_for_test(path)

    dataset = read_nonmem_dataset(StringIO('1 0 3\n2 0 4\n'), colnames=['ID', 'TIME', 'DV'])
    grad = evaluate_eta_gradient(model, dataset=dataset)
    pd.testing.assert_frame_equal(grad, pd.DataFrame([1.0, 1.0], columns=['dF/dETA_1']))

    linpath = testdata / 'nonmem' / 'pheno_real_linbase.mod'
    linmod = load_model_for_test(linpath)
    grad = evaluate_eta_gradient(linmod)
    pd.testing.assert_series_equal(lincorrect['G11'], grad.iloc[:, 0], rtol=1e-4, check_names=False)
    pd.testing.assert_series_equal(lincorrect['G21'], grad.iloc[:, 1], rtol=1e-4, check_names=False)


def test_evaluate_epsilon_gradient(load_model_for_test, testdata):
    path = testdata / 'nonmem' / 'minimal.mod'
    model = load_model_for_test(path)

    dataset = read_nonmem_dataset(StringIO('1 0 3\n2 0 4\n'), colnames=['ID', 'TIME', 'DV'])
    grad = evaluate_epsilon_gradient(model, dataset=dataset)
    pd.testing.assert_frame_equal(grad, pd.DataFrame([1.0, 1.0], columns=['dY/dEPS_1']))

    linpath = testdata / 'nonmem' / 'pheno_real_linbase.mod'
    linmod = load_model_for_test(linpath)
    res = read_modelfit_results(linpath)
    grad = evaluate_epsilon_gradient(linmod, etas=res.individual_estimates)
    pd.testing.assert_series_equal(lincorrect['H11'], grad.iloc[:, 0], rtol=1e-4, check_names=False)


def test_evaluate_weighted_residuals(load_model_for_test, testdata):
    linpath = testdata / 'nonmem' / 'pheno_real_linbase.mod'
    linmod = load_model_for_test(linpath)
    res = read_modelfit_results(linpath)
    wres = evaluate_weighted_residuals(linmod, parameters=dict(res.parameter_estimates))
    pd.testing.assert_series_equal(lincorrect['WRES'], wres, rtol=1e-4, check_names=False)
