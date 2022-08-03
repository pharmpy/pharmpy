import pytest

from pharmpy.model import Model
from pharmpy.modeling import has_covariate_effect, remove_covariate_effect

from ..lib import diff


@pytest.mark.parametrize(
    ('model_path', 'effects', 'expected'),
    [
        (
            ('nonmem', 'pheno_real.mod'),
            [('CL', 'WGT')],
            '@@ -4 +4 @@\n' '-TVCL=THETA(1)*WGT\n' '+TVCL = THETA(1)\n',
        ),
        (
            ('nonmem', 'pheno_real.mod'),
            [('V', 'WGT')],
            '@@ -5 +5 @@\n' '-TVV=THETA(2)*WGT\n' '+TVV = THETA(2)\n',
        ),
        (
            ('nonmem', 'pheno_real.mod'),
            [('V', 'APGR')],
            '@@ -6 +5,0 @@\n' '-IF(APGR.LT.5) TVV=TVV*(1+THETA(3))\n',
        ),
        (
            ('nonmem', 'pheno_real.mod'),
            [('CL', 'WGT'), ('V', 'WGT')],
            '@@ -4,2 +4,2 @@\n'
            '-TVCL=THETA(1)*WGT\n'
            '-TVV=THETA(2)*WGT\n'
            '+TVCL = THETA(1)\n'
            '+TVV = THETA(2)\n',
        ),
        (
            ('nonmem', 'pheno_real.mod'),
            [('V', 'WGT'), ('CL', 'WGT')],
            '@@ -4,2 +4,2 @@\n'
            '-TVCL=THETA(1)*WGT\n'
            '-TVV=THETA(2)*WGT\n'
            '+TVCL = THETA(1)\n'
            '+TVV = THETA(2)\n',
        ),
        (
            ('nonmem', 'pheno_real.mod'),
            [('V', 'WGT'), ('CL', 'WGT'), ('V', 'APGR')],
            '@@ -4,3 +4,2 @@\n'
            '-TVCL=THETA(1)*WGT\n'
            '-TVV=THETA(2)*WGT\n'
            '-IF(APGR.LT.5) TVV=TVV*(1+THETA(3))\n'
            '+TVCL = THETA(1)\n'
            '+TVV = THETA(2)\n',
        ),
        (
            ('nonmem', 'models', 'fviii6.mod'),
            [('CL', 'WT')],
            '@@ -29,8 +29,4 @@\n'
            '-IF (PREP2.EQ.1) THEN\n'
            '-TVCL = THETA(1)*((WT/80)**THETA(7))*(1+THETA(13)*(AGE-40)) ;80 = Median of body weight\n'
            '-TVV1 = THETA(2)*((WT/80)**THETA(8)) \n'
            '-ENDIF\n'
            '-IF (PREP2.EQ.2) THEN\n'
            '-TVCL = THETA(1)*((WT/80)**THETA(7))*(1+THETA(12))*(1+THETA(13)*(AGE-40))\n'
            '-TVV1 = THETA(2)*((WT/80)**THETA(8))*(1+THETA(12))\n'
            '-ENDIF\n'
            '+IF (PREP2.EQ.1) TVCL = THETA(1)*(THETA(13)*(AGE - 40) + 1)\n'
            '+IF (PREP2.EQ.1) TVV1 = THETA(2)*(WT/80)**THETA(8)\n'
            '+IF (PREP2.EQ.2) TVCL = THETA(1)*(THETA(12) + 1)*(THETA(13)*(AGE - 40) + 1)\n'
            '+IF (PREP2.EQ.2) TVV1 = THETA(2)*(WT/80)**THETA(8)*(THETA(12) + 1)\n',
        ),
        (
            ('nonmem', 'models', 'fviii6.mod'),
            [('V1', 'WT')],
            '@@ -29,8 +29,4 @@\n'
            '-IF (PREP2.EQ.1) THEN\n'
            '-TVCL = THETA(1)*((WT/80)**THETA(7))*(1+THETA(13)*(AGE-40)) ;80 = Median of body weight\n'
            '-TVV1 = THETA(2)*((WT/80)**THETA(8)) \n'
            '-ENDIF\n'
            '-IF (PREP2.EQ.2) THEN\n'
            '-TVCL = THETA(1)*((WT/80)**THETA(7))*(1+THETA(12))*(1+THETA(13)*(AGE-40))\n'
            '-TVV1 = THETA(2)*((WT/80)**THETA(8))*(1+THETA(12))\n'
            '-ENDIF\n'
            '+IF (PREP2.EQ.1) TVCL = THETA(1)*(WT/80)**THETA(7)*(THETA(13)*(AGE - 40) + 1)\n'
            '+IF (PREP2.EQ.1) TVV1 = THETA(2)\n'
            '+IF (PREP2.EQ.2) TVCL = THETA(1)*(WT/80)**THETA(7)*(THETA(12) + 1)*(THETA(13)*(AGE - 40) + 1)\n'
            '+IF (PREP2.EQ.2) TVV1 = THETA(2)*(THETA(12) + 1)\n',
        ),
        (
            ('nonmem', 'models', 'fviii6.mod'),
            [('CL', 'WT'), ('V1', 'WT')],
            '@@ -29,8 +29,4 @@\n'
            '-IF (PREP2.EQ.1) THEN\n'
            '-TVCL = THETA(1)*((WT/80)**THETA(7))*(1+THETA(13)*(AGE-40)) ;80 = Median of body weight\n'
            '-TVV1 = THETA(2)*((WT/80)**THETA(8)) \n'
            '-ENDIF\n'
            '-IF (PREP2.EQ.2) THEN\n'
            '-TVCL = THETA(1)*((WT/80)**THETA(7))*(1+THETA(12))*(1+THETA(13)*(AGE-40))\n'
            '-TVV1 = THETA(2)*((WT/80)**THETA(8))*(1+THETA(12))\n'
            '-ENDIF\n'
            '+IF (PREP2.EQ.1) TVCL = THETA(1)*(THETA(13)*(AGE - 40) + 1)\n'
            '+IF (PREP2.EQ.1) TVV1 = THETA(2)\n'
            '+IF (PREP2.EQ.2) TVCL = THETA(1)*(THETA(12) + 1)*(THETA(13)*(AGE - 40) + 1)\n'
            '+IF (PREP2.EQ.2) TVV1 = THETA(2)*(THETA(12) + 1)\n',
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

    model.update_source()
    error_record_after = ''.join(map(str, model.internals.control_stream.get_records('ERROR')))

    original_model = Model.create_model(testdata.joinpath(*model_path))
    assert (
        diff(
            original_model.internals.control_stream.get_pred_pk_record(),
            model.internals.control_stream.get_pred_pk_record(),
        )
        == expected
    )
    assert error_record_after == error_record_before

    for effect in effects:
        assert not has_covariate_effect(model, effect[0], effect[1])

    for effect in effects:
        assert f'POP_{effect[0]}{effect[1]}' not in model.model_code
