import sys

import numpy as np
import packaging
import pytest

from pharmpy.modeling import load_example_model
from pharmpy.tools import (
    load_example_modelfit_results,
    predict_influential_individuals,
    predict_outliers,
)

tflite_condition = (
    sys.version_info >= (3, 12)
    and sys.platform == 'win32'
    or sys.version_info >= (3, 12)
    and sys.platform == 'darwin'
    or packaging.version.parse(np.__version__) >= packaging.version.parse("2.0.0")
)


@pytest.mark.skipif(tflite_condition, reason="Skipping tests requiring tflite for Python 3.12")
def test_predict_outliers():
    model = load_example_model('pheno')
    results = load_example_modelfit_results('pheno')
    res = predict_outliers(model, results)
    assert len(res) == 59
    assert res['residual'][1] == pytest.approx(-0.28144291043281555)


@pytest.mark.skipif(tflite_condition, reason="Skipping tests requiring tflite for Python 3.12")
def test_predict_influential_individuals():
    model = load_example_model('pheno')
    results = load_example_modelfit_results('pheno')
    res = predict_influential_individuals(model, results)
    assert len(res) == 59
    assert res['dofv'][59] == pytest.approx(0.08806940913200378)
