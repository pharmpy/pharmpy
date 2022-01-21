import pytest

from pharmpy import Model
from pharmpy.tools.estmethod.tool import _clear_estimation_steps, _create_est_model


@pytest.mark.parametrize(
    'method, est_rec, eval_rec',
    [
        (
            'fo',
            '$ESTIMATION METHOD=ZERO INTER MAXEVAL=9999 AUTO=1 PRINT=10',
            '$ESTIMATION METHOD=IMP INTER EONLY=1 MAXEVAL=9999 ISAMPLE=10000 NITER=10 PRINT=10',
        ),
        (
            'laplace',
            '$ESTIMATION METHOD=COND LAPLACE INTER MAXEVAL=9999 AUTO=1 PRINT=10',
            '$ESTIMATION METHOD=IMP LAPLACE INTER EONLY=1 MAXEVAL=9999 ISAMPLE=10000 '
            'NITER=10 PRINT=10',
        ),
    ],
)
def test_create_est_model(pheno_path, method, est_rec, eval_rec):
    model = Model.create_model(pheno_path)
    assert len(model.estimation_steps) == 1
    est_model = _create_est_model(method, update=False, model=model)
    assert len(est_model.estimation_steps) == 2
    assert est_model.name == f'estmethod_{method.upper()}_raw_inits'
    assert est_model.model_code.split('\n')[-5] == est_rec
    assert est_model.model_code.split('\n')[-4] == eval_rec


def test_clear_estimation_steps(pheno_path):
    model = Model.create_model(pheno_path)
    assert len(model.estimation_steps) == 1
    _clear_estimation_steps(model)
    assert len(model.estimation_steps) == 0
