import pytest

from pharmpy import Model
from pharmpy.modeling import absorption, add_covariate_effect, explicit_odes


@pytest.mark.parametrize('effect, operation, buf_new', [
    ('exp', '*', 'CLWGT = EXP(THETA(4)*(-CL_MEDIAN + WGT))\n      '
                 'CL_MEDIAN = 1.30000\n      '
                 'CL = CLWGT*TVCL*EXP(ETA(1))'),
    ('exp', '+', 'CLWGT = EXP(THETA(4)*(-CL_MEDIAN + WGT))\n      '
                 'CL_MEDIAN = 1.30000\n      '
                 'CL = CLWGT + TVCL*EXP(ETA(1))'),
    ('pow', '*', 'CLWGT = (WGT/CL_MEDIAN)**THETA(4)\n      '
                 'CL_MEDIAN = 1.30000\n      '
                 'CL = CLWGT*TVCL*EXP(ETA(1))'),
    ('lin_cont', '*', 'CLWGT = THETA(4)*(-CL_MEDIAN + WGT) + 1\n      '
                      'CL_MEDIAN = 1.30000\n      '
                      'CL = CLWGT*TVCL*EXP(ETA(1))'),
    ('lin_cat', '*', 'IF (WGT.EQ.1) THEN\n'
                     'CLWGT = 1\n'
                     'ELSE IF (WGT.EQ.0) THEN\n'
                     'CLWGT = THETA(4) + 1\n'
                     'END IF\n      '
                     'CL_MEDIAN = 1.30000\n      '
                     'CL = CLWGT*TVCL*EXP(ETA(1))'),
    ('piece_lin', '*', 'IF (CL_MEDIAN.GE.WGT) THEN\n'
                       'CLWGT = THETA(4)*(-CL_MEDIAN + WGT) + 1\n'
                       'ELSE\n'
                       'CLWGT = THETA(5)*(-CL_MEDIAN + WGT) + 1\n'
                       'END IF\n      '
                       'CL_MEDIAN = 1.30000\n      '
                       'CL = CLWGT*TVCL*EXP(ETA(1))'),
    ('theta - cov + median', '*', 'CLWGT = CL_MEDIAN + THETA(4) - WGT\n      '
                                  'CL_MEDIAN = 1.30000\n      '
                                  'CL = CLWGT*TVCL*EXP(ETA(1))')

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
              f'      V=TVV*EXP(ETA(2))\n' \
              f'      S1=V\n'

    assert str(model.get_pred_pk_record()) == rec_ref


def test_to_explicit_odes(pheno_path):
    model = Model(pheno_path)

    explicit_odes(model)
    model.update_source()
    lines = str(model).split('\n')
    assert lines[6] == '$MODEL TOL=3 COMPARTMENT=(CENTRAL DEFDOSE)'
    assert lines[18] == '$DES'
    assert lines[19] == 'DADT(1) = -A(1)*CL/V'


def test_aborption(testdata):
    model = Model(testdata / 'nonmem' / 'pheno_real.mod')
    first = str(model)

    absorption(model, 0)
    assert first == str(model)

    model = Model(testdata / 'nonmem' / 'ph_abs1.mod')
    absorption(model, 0)
    model.update_source()
    a = str(model).split('\n')
    assert a[3] == '$SUBROUTINE ADVAN1 TRANS2'
    assert a[13].strip() == 'S1=V'
    assert a[25] == '$OMEGA  DIAGONAL(2)'
