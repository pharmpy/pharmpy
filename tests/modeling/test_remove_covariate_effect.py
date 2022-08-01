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
