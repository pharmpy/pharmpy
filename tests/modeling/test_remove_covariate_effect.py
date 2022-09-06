import pytest

from pharmpy.model import Model
from pharmpy.modeling import has_covariate_effect, remove_covariate_effect


@pytest.mark.parametrize(
    ('model_path', 'effects', 'expected'),
    [
        (
            ('nonmem', 'pheno_real.mod'),
            [('CL', 'WGT')],
            '$PK\n'
            'IF(AMT.GT.0) BTIME=TIME\n'
            'TAD=TIME-BTIME\n'
            'TVCL = THETA(1)\n'
            'TVV=THETA(2)*WGT\n'
            'IF(APGR.LT.5) TVV=TVV*(1+THETA(3))\n'
            'CL=TVCL*EXP(ETA(1))\n'
            'V=TVV*EXP(ETA(2))\n'
            'S1=V\n\n',
        ),
        (
            ('nonmem', 'pheno_real.mod'),
            [('V', 'WGT')],
            '$PK\n'
            'IF(AMT.GT.0) BTIME=TIME\n'
            'TAD=TIME-BTIME\n'
            'TVCL=THETA(1)*WGT\n'
            'TVV = THETA(2)\n'
            'IF(APGR.LT.5) TVV=TVV*(1+THETA(3))\n'
            'CL=TVCL*EXP(ETA(1))\n'
            'V=TVV*EXP(ETA(2))\n'
            'S1=V\n\n',
        ),
        (
            ('nonmem', 'pheno_real.mod'),
            [('V', 'APGR')],
            '$PK\n'
            'IF(AMT.GT.0) BTIME=TIME\n'
            'TAD=TIME-BTIME\n'
            'TVCL=THETA(1)*WGT\n'
            'TVV=THETA(2)*WGT\n'
            'CL=TVCL*EXP(ETA(1))\n'
            'V=TVV*EXP(ETA(2))\n'
            'S1=V\n\n',
        ),
        (
            ('nonmem', 'pheno_real.mod'),
            [('CL', 'WGT'), ('V', 'WGT')],
            '$PK\n'
            'IF(AMT.GT.0) BTIME=TIME\n'
            'TAD=TIME-BTIME\n'
            'TVCL = THETA(1)\n'
            'TVV = THETA(2)\n'
            'IF(APGR.LT.5) TVV=TVV*(1+THETA(3))\n'
            'CL=TVCL*EXP(ETA(1))\n'
            'V=TVV*EXP(ETA(2))\n'
            'S1=V\n\n',
        ),
        (
            ('nonmem', 'pheno_real.mod'),
            [('V', 'WGT'), ('CL', 'WGT')],
            '$PK\n'
            'IF(AMT.GT.0) BTIME=TIME\n'
            'TAD=TIME-BTIME\n'
            'TVCL = THETA(1)\n'
            'TVV = THETA(2)\n'
            'IF(APGR.LT.5) TVV=TVV*(1+THETA(3))\n'
            'CL=TVCL*EXP(ETA(1))\n'
            'V=TVV*EXP(ETA(2))\n'
            'S1=V\n\n',
        ),
        (
            ('nonmem', 'pheno_real.mod'),
            [('V', 'WGT'), ('CL', 'WGT'), ('V', 'APGR')],
            '$PK\n'
            'IF(AMT.GT.0) BTIME=TIME\n'
            'TAD=TIME-BTIME\n'
            'TVCL = THETA(1)\n'
            'TVV = THETA(2)\n'
            'CL=TVCL*EXP(ETA(1))\n'
            'V=TVV*EXP(ETA(2))\n'
            'S1=V\n\n',
        ),
        (
            ('nonmem', 'models', 'fviii6.mod'),
            [('CL', 'WT')],
            '$PK\n'
            'Q1 = 0\n'
            'Q2 = 0\n'
            'Q3 = 0\n'
            'Q4 = 0\n'
            'Q5 = 0\n'
            'Q6 = 0\n'
            'Q7 = 0\n'
            'Q8 = 0\n'
            'IF(OCC.EQ.1)Q1 = 1\n'
            'IF(OCC.EQ.2)Q2 = 1\n'
            'IF(OCC.EQ.3)Q3 = 1\n'
            'IF(OCC.EQ.4)Q4 = 1\n'
            'IF(OCC.EQ.5)Q5 = 1\n'
            'IF(OCC.EQ.6)Q6 = 1\n'
            'IF(OCC.EQ.7)Q7 = 1\n'
            'IF(OCC.EQ.8)Q8 = 1\n'
            '\n'
            'IF (PREP.EQ.1) PREP2 = 1\n'
            'IF (PREP.EQ.2) PREP2 = 2\n'
            'IF (PREP.EQ.3) PREP2 = 1\n'
            'IF (PREP.EQ.4) PREP2 = 1\n'
            'IF (PREP.EQ.5) PREP2 = 1\n'
            'IF (PREP.EQ.6) PREP2 = 2\n'
            'IF (PREP.EQ.7) PREP2 = 2\n'
            ';IF (PREP.EQ.8) PREP2 = 3\n'
            'IF (PREP.EQ.9) PREP2 = 2\n'
            '\n'
            'IF (PREP2.EQ.1) TVCL = THETA(1)*(THETA(13)*(AGE - 40) + 1)\n'
            'IF (PREP2.EQ.1) TVV1 = THETA(2)*(WT/80)**THETA(8)\n'
            'IF (PREP2.EQ.2) TVCL = THETA(1)*(THETA(12) + 1)*(THETA(13)*(AGE - 40) + 1)\n'
            'IF (PREP2.EQ.2) TVV1 = THETA(2)*(WT/80)**THETA(8)*(THETA(12) + 1)\n'
            'TVQ  = THETA(5)*(WT/80)**THETA(9)\n'
            'TVV2 = THETA(6)*(WT/80)**THETA(10)\n'
            'IOVCL= Q1*ETA(4)+Q2*ETA(5)+Q3*ETA(6)+Q4*ETA(7)+Q5*ETA(8)+Q6*ETA(9)+Q7*ETA(10)+Q8*ETA(11)\n'
            ';IOVCL2 = IOVCL+Q6*ETA(9)+Q7*ETA(10)+Q8*ETA(11)\n'
            'IOVV = Q1*ETA(12)+Q2*ETA(13)+Q3*ETA(14)+Q4*ETA(15)+Q5*ETA(16)+Q6*ETA(17)+Q7*ETA(18)+Q8*ETA(19)\n'
            ';IOVV2= IOVV+Q5*ETA(16)+Q6*ETA(17)+Q7*ETA(18)+Q8*ETA(19)\n'
            'CL   = TVCL*EXP(ETA(2)+IOVCL)\n'
            'V1   = TVV1*EXP(ETA(3)+IOVV)\n'
            'Q    = TVQ\n'
            'V2   = TVV2\n'
            '\n'
            'TVBA = THETA(11)\n'
            'BASE = TVBA*EXP(ETA(1))\n'
            '\n'
            'S1 = V1\n'
            '\n'
            'K = CL/V1\n'
            'K12 = Q/V1\n'
            'K21 = Q/V2\n\n',
        ),
        (
            ('nonmem', 'models', 'fviii6.mod'),
            [('V1', 'WT')],
            '$PK\n'
            'Q1 = 0\n'
            'Q2 = 0\n'
            'Q3 = 0\n'
            'Q4 = 0\n'
            'Q5 = 0\n'
            'Q6 = 0\n'
            'Q7 = 0\n'
            'Q8 = 0\n'
            'IF(OCC.EQ.1)Q1 = 1\n'
            'IF(OCC.EQ.2)Q2 = 1\n'
            'IF(OCC.EQ.3)Q3 = 1\n'
            'IF(OCC.EQ.4)Q4 = 1\n'
            'IF(OCC.EQ.5)Q5 = 1\n'
            'IF(OCC.EQ.6)Q6 = 1\n'
            'IF(OCC.EQ.7)Q7 = 1\n'
            'IF(OCC.EQ.8)Q8 = 1\n'
            '\n'
            'IF (PREP.EQ.1) PREP2 = 1\n'
            'IF (PREP.EQ.2) PREP2 = 2\n'
            'IF (PREP.EQ.3) PREP2 = 1\n'
            'IF (PREP.EQ.4) PREP2 = 1\n'
            'IF (PREP.EQ.5) PREP2 = 1\n'
            'IF (PREP.EQ.6) PREP2 = 2\n'
            'IF (PREP.EQ.7) PREP2 = 2\n'
            ';IF (PREP.EQ.8) PREP2 = 3\n'
            'IF (PREP.EQ.9) PREP2 = 2\n'
            '\n'
            'IF (PREP2.EQ.1) TVCL = THETA(1)*(WT/80)**THETA(7)*(THETA(13)*(AGE - 40) + 1)\n'
            'IF (PREP2.EQ.1) TVV1 = THETA(2)\n'
            'IF (PREP2.EQ.2) TVCL = THETA(1)*(WT/80)**THETA(7)*(THETA(12) + 1)*(THETA(13)*(AGE - 40) + 1)\n'
            'IF (PREP2.EQ.2) TVV1 = THETA(2)*(THETA(12) + 1)\n'
            'TVQ  = THETA(5)*(WT/80)**THETA(9)\n'
            'TVV2 = THETA(6)*(WT/80)**THETA(10)\n'
            'IOVCL= Q1*ETA(4)+Q2*ETA(5)+Q3*ETA(6)+Q4*ETA(7)+Q5*ETA(8)+Q6*ETA(9)+Q7*ETA(10)+Q8*ETA(11)\n'
            ';IOVCL2 = IOVCL+Q6*ETA(9)+Q7*ETA(10)+Q8*ETA(11)\n'
            'IOVV = Q1*ETA(12)+Q2*ETA(13)+Q3*ETA(14)+Q4*ETA(15)+Q5*ETA(16)+Q6*ETA(17)+Q7*ETA(18)+Q8*ETA(19)\n'
            ';IOVV2= IOVV+Q5*ETA(16)+Q6*ETA(17)+Q7*ETA(18)+Q8*ETA(19)\n'
            'CL   = TVCL*EXP(ETA(2)+IOVCL)\n'
            'V1   = TVV1*EXP(ETA(3)+IOVV)\n'
            'Q    = TVQ\n'
            'V2   = TVV2\n'
            '\n'
            'TVBA = THETA(11)\n'
            'BASE = TVBA*EXP(ETA(1))\n'
            '\n'
            'S1 = V1\n'
            '\n'
            'K = CL/V1\n'
            'K12 = Q/V1\n'
            'K21 = Q/V2\n\n',
        ),
        (
            ('nonmem', 'models', 'fviii6.mod'),
            [('CL', 'WT'), ('V1', 'WT')],
            '$PK\n'
            'Q1 = 0\n'
            'Q2 = 0\n'
            'Q3 = 0\n'
            'Q4 = 0\n'
            'Q5 = 0\n'
            'Q6 = 0\n'
            'Q7 = 0\n'
            'Q8 = 0\n'
            'IF(OCC.EQ.1)Q1 = 1\n'
            'IF(OCC.EQ.2)Q2 = 1\n'
            'IF(OCC.EQ.3)Q3 = 1\n'
            'IF(OCC.EQ.4)Q4 = 1\n'
            'IF(OCC.EQ.5)Q5 = 1\n'
            'IF(OCC.EQ.6)Q6 = 1\n'
            'IF(OCC.EQ.7)Q7 = 1\n'
            'IF(OCC.EQ.8)Q8 = 1\n'
            '\n'
            'IF (PREP.EQ.1) PREP2 = 1\n'
            'IF (PREP.EQ.2) PREP2 = 2\n'
            'IF (PREP.EQ.3) PREP2 = 1\n'
            'IF (PREP.EQ.4) PREP2 = 1\n'
            'IF (PREP.EQ.5) PREP2 = 1\n'
            'IF (PREP.EQ.6) PREP2 = 2\n'
            'IF (PREP.EQ.7) PREP2 = 2\n'
            ';IF (PREP.EQ.8) PREP2 = 3\n'
            'IF (PREP.EQ.9) PREP2 = 2\n'
            '\n'
            'IF (PREP2.EQ.1) TVCL = THETA(1)*(THETA(13)*(AGE - 40) + 1)\n'
            'IF (PREP2.EQ.1) TVV1 = THETA(2)\n'
            'IF (PREP2.EQ.2) TVCL = THETA(1)*(THETA(12) + 1)*(THETA(13)*(AGE - 40) + 1)\n'
            'IF (PREP2.EQ.2) TVV1 = THETA(2)*(THETA(12) + 1)\n'
            'TVQ  = THETA(5)*(WT/80)**THETA(9)\n'
            'TVV2 = THETA(6)*(WT/80)**THETA(10)\n'
            'IOVCL= Q1*ETA(4)+Q2*ETA(5)+Q3*ETA(6)+Q4*ETA(7)+Q5*ETA(8)+Q6*ETA(9)+Q7*ETA(10)+Q8*ETA(11)\n'
            ';IOVCL2 = IOVCL+Q6*ETA(9)+Q7*ETA(10)+Q8*ETA(11)\n'
            'IOVV = Q1*ETA(12)+Q2*ETA(13)+Q3*ETA(14)+Q4*ETA(15)+Q5*ETA(16)+Q6*ETA(17)+Q7*ETA(18)+Q8*ETA(19)\n'
            ';IOVV2= IOVV+Q5*ETA(16)+Q6*ETA(17)+Q7*ETA(18)+Q8*ETA(19)\n'
            'CL   = TVCL*EXP(ETA(2)+IOVCL)\n'
            'V1   = TVV1*EXP(ETA(3)+IOVV)\n'
            'Q    = TVQ\n'
            'V2   = TVV2\n'
            '\n'
            'TVBA = THETA(11)\n'
            'BASE = TVBA*EXP(ETA(1))\n'
            '\n'
            'S1 = V1\n'
            '\n'
            'K = CL/V1\n'
            'K12 = Q/V1\n'
            'K21 = Q/V2\n\n',
        ),
    ],
    ids=repr,
)
def test_remove_covariate_effect(testdata, model_path, effects, expected):
    model = Model.create_model(testdata.joinpath(*model_path))
    error_record_before = ''.join(map(str, model.internals.control_stream.get_records('ERROR')))

    for effect in effects:
        assert has_covariate_effect(model, effect[0], effect[1])

    for effect in effects:
        remove_covariate_effect(model, *effect)

    for effect in effects:
        assert not has_covariate_effect(model, effect[0], effect[1])

    model.update_source()
    error_record_after = ''.join(map(str, model.internals.control_stream.get_records('ERROR')))

    assert str(model.internals.control_stream.get_pred_pk_record()) == expected
    assert error_record_after == error_record_before

    for effect in effects:
        assert f'POP_{effect[0]}{effect[1]}' not in model.model_code
