import sys

import numpy as np
import pandas as pd
import pytest

from pharmpy.modeling import read_model
from pharmpy.results import ModelfitResults
from pharmpy.tools import (
    read_modelfit_results,
    summarize_individuals,
    summarize_individuals_count_table,
)
from pharmpy.tools.funcs.summarize_individuals import dofv


def test_summarize_individuals_count_table():
    df = pd.DataFrame(
        {
            'model': ['start_model'] * 3 + ['candidate1'] * 3 + ['candidate2'] * 3,
            'ID': [1, 2, 3, 1, 2, 3, 1, 2, 3],
            'parent_model': ['start_model'] * 6 + ['candidate1'] * 3,
            'outlier_count': [0, 0, 0, 1, 0, 2, 0, 0, 1],
            'ofv': [-1, -2, -3, -4, -5, -6, -7, -8, -9],
            'dofv_vs_parent': [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            'predicted_dofv': [1.0, 1.5, 2.5, 3.0, 3.0, 4.0, 4.0, 2.0, 1.0],
            'predicted_residual': [1.0, 1.0, 1.0, 4.0, 4.0, 1.0, 1.0, 1.0, 1.0],
        }
    )
    df.set_index(['model', 'ID'], inplace=True)

    res = summarize_individuals_count_table(df=df)
    assert list(res['inf_selection']) == [0, 0, 0]
    assert list(res['inf_params']) == [0, 1, 1]
    assert list(res['out_obs']) == [0, 2, 1]
    assert list(res['out_ind']) == [0, 2, 0]
    assert list(res['inf_outlier']) == [0, 1, 0]


tflite_condition = (
    sys.version_info >= (3, 10)
    and sys.platform != 'linux'
    or sys.version_info >= (3, 11)
    and sys.platform == 'linux'
)


@pytest.mark.skipif(tflite_condition, reason="Skipping tests requiring tflite for Python 3.10")
def test_tflite_not_installed(pheno_path, monkeypatch):
    model = read_model(pheno_path)
    results = read_modelfit_results(pheno_path)
    # FIXME
    model = model.replace(modelfit_results=results)

    df = summarize_individuals([model])
    assert not df['predicted_dofv'].isnull().any().any()

    with pytest.warns(UserWarning, match='tflite is not installed'):
        monkeypatch.setitem(sys.modules, 'tflite_runtime', None)
        df = summarize_individuals([model])
        assert df['predicted_dofv'].isnull().all().all()


def test_dofv_parent_model_is_none(pheno_path):
    candidate_model = read_model(pheno_path)
    res = dofv(None, candidate_model)
    assert np.isnan(res)


def test_dofv_modelfit_results_is_none(pheno_path):
    parent_model = read_model(pheno_path)
    res = read_modelfit_results(pheno_path)
    parent_model = parent_model.replace(modelfit_results=res)
    candidate_model = parent_model.replace(modelfit_results=None)
    res = dofv(parent_model, candidate_model)
    assert res.isna().all()


def test_dofv_individual_ofv_is_none(pheno_path):
    parent_model = read_model(pheno_path)
    res = read_modelfit_results(pheno_path)
    parent_model = parent_model.replace(modelfit_results=res)
    candidate_model = parent_model.replace(modelfit_results=ModelfitResults())
    res = dofv(parent_model, candidate_model)
    assert res.isna().all()
