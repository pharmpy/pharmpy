import pytest

from pharmpy import Model
from pharmpy.modeling import evaluate_expression


def test_evaluate_expression(testdata):
    model = Model(testdata / 'nonmem' / 'models' / 'pheno_noifs.mod')
    ser = evaluate_expression(model, 'TVV')
    assert ser[0] == pytest.approx(1.413062)
    assert ser[743] == pytest.approx(1.110262)
