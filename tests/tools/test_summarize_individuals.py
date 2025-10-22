import sys

import numpy as np
import packaging
import pytest

from pharmpy.deps import pandas as pd
from pharmpy.tools.external.results import parse_modelfit_results
from pharmpy.tools.funcs import summarize_individuals, summarize_individuals_count_table
from pharmpy.tools.funcs.summarize_individuals import dofv
from pharmpy.workflows import ModelEntry
from pharmpy.workflows.results import ModelfitResults


def test_summarize_individuals_count_table():
    df = pd.DataFrame(
        {
            'model': ['start_model'] * 3 + ['candidate1'] * 3 + ['candidate2'] * 3,
            'ID': [1, 2, 3, 1, 2, 3, 1, 2, 3],
            'parent_model': [None] * 3 + ['start_model'] * 3 + ['candidate1'] * 3,
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
    sys.version_info >= (3, 12)
    and sys.platform == 'win32'
    or sys.version_info >= (3, 12)
    and sys.platform == 'darwin'
    or packaging.version.parse(np.__version__) >= packaging.version.parse("2.0.0")
)


@pytest.mark.skipif(tflite_condition, reason="Skipping tests requiring tflite for Python 3.12")
def test_tflite_not_installed(pheno, pheno_path, monkeypatch):
    results = parse_modelfit_results(pheno, pheno_path)
    me = ModelEntry(model=pheno, modelfit_results=results)

    df = summarize_individuals([me])
    assert not df['predicted_dofv'].isnull().any().any()

    monkeypatch.setitem(sys.modules, 'tflite_runtime', None)
    df = summarize_individuals([me])
    assert df['predicted_dofv'].isnull().all().all()


def test_dofv_parent_model_is_none(pheno, pheno_path):
    res = parse_modelfit_results(pheno, pheno_path)
    res = dofv(None, res)
    assert np.isnan(res)


def test_dofv_modelfit_results_is_none(pheno, pheno_path):
    parent_res = parse_modelfit_results(pheno, pheno_path)
    res = dofv(parent_res, None)
    assert res.isna().all()


def test_dofv_individual_ofv_is_none(pheno, pheno_path):
    parent_res = parse_modelfit_results(pheno, pheno_path)
    res = dofv(parent_res, ModelfitResults())
    assert res.isna().all()
