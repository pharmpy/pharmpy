import pytest

from pharmpy import Model
from pharmpy.covariate_effect import CovariateEffect
from pharmpy.model_transformation import ModelTransformation


@pytest.mark.parametrize('operation, added_covariate_effect', [
    ('*', 'CLWGT = exp(THETA(4)*(WGT - 1.3))\nCL = CL*CLWGT\n'),
    ('+', 'CLWGT = exp(THETA(4)*(WGT - 1.3))\nCL = CL + CLWGT\n'),
])
def test_add_covariate_effect(pheno_path, operation, added_covariate_effect):
    model = Model(pheno_path)
    model_t = ModelTransformation(model)

    ce = CovariateEffect

    model_t.add_covariate_effect('CL', 'WGT', ce.exponential, operation)

    rec_ref = f'$PK\n' \
              f'IF(AMT.GT.0) BTIME=TIME\n' \
              f'TAD=TIME-BTIME\n' \
              f'      TVCL=THETA(1)*WGT\n' \
              f'      TVV=THETA(2)*WGT\n' \
              f'IF(APGR.LT.5) TVV=TVV*(1+THETA(3))\n' \
              f'      CL=TVCL*EXP(ETA(1))\n' \
              f'      V=TVV*EXP(ETA(2))\n' \
              f'      S1=V\n' \
              f'{added_covariate_effect}'

    assert str(model_t.model.get_pred_pk_record()) == rec_ref
