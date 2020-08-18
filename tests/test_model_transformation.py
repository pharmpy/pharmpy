import pytest

from pharmpy import Model
from pharmpy.model_transformation import add_covariate_effect


@pytest.mark.parametrize('effect, operation, buf_new', [
    ('exp', '*', 'CLWGT = exp(THETA(4)*(-CL_MEDIAN + WGT))\nCL_MEDIAN = 1.3\n'
                 'CL = CLWGT*TVCL*exp(ETA(1))'),
    ('exp', '+', 'CLWGT = exp(THETA(4)*(-CL_MEDIAN + WGT))\nCL_MEDIAN = 1.3\n'
                 'CL = CLWGT + TVCL*exp(ETA(1))'),
])
def test_add_covariate_effect(pheno_path, effect, operation, buf_new):
    model = Model(pheno_path)

    add_covariate_effect(model, 'CL', 'WGT', effect, operation)
    model.update_source()

    rec_ref = f'$PK\n' \
              f'IF(AMT.GT.0) BTIME=TIME\n' \
              f'TAD=TIME-BTIME\n' \
              f'      TVCL=THETA(1)*WGT\n' \
              f'      TVV=THETA(2)*WGT\n' \
              f'IF(APGR.LT.5) TVV=TVV*(1+THETA(3))\n' \
              f'      CL=TVCL*EXP(ETA(1))\n' \
              f'      {buf_new}\n' \
              f'V=TVV*EXP(ETA(2))\n' \
              f'      S1=V\n'

    assert str(model.get_pred_pk_record()) == rec_ref
