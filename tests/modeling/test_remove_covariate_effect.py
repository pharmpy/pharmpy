import pytest

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
            '@@ -29,10 +29,6 @@\n'
            '-IF (PREP2.EQ.1) THEN\n'
            '-TVCL = THETA(1)*((WT/80)**THETA(7))*(1+THETA(13)*(AGE-40)) ;80 = Median of body weight\n'
            '-TVV1 = THETA(2)*((WT/80)**THETA(8)) \n'
            '-ENDIF\n'
            '-IF (PREP2.EQ.2) THEN\n'
            '-TVCL = THETA(1)*((WT/80)**THETA(7))*(1+THETA(12))*(1+THETA(13)*(AGE-40))\n'
            '-TVV1 = THETA(2)*((WT/80)**THETA(8))*(1+THETA(12))\n'
            '-ENDIF\n'
            '-TVQ  = THETA(5)*(WT/80)**THETA(9)\n'
            '-TVV2 = THETA(6)*(WT/80)**THETA(10)\n'
            '+IF (PREP2.EQ.1) TVCL = THETA(1)*(THETA(12)*(AGE - 40) + 1)\n'
            '+IF (PREP2.EQ.1) TVV1 = THETA(2)*(WT/80)**THETA(7)\n'
            '+IF (PREP2.EQ.2) TVCL = THETA(1)*(THETA(11) + 1)*(THETA(12)*(AGE - 40) + 1)\n'
            '+IF (PREP2.EQ.2) TVV1 = THETA(2)*(WT/80)**THETA(7)*(THETA(11) + 1)\n'
            '+TVQ = THETA(5)*(WT/80)**THETA(8)\n'
            '+TVV2 = THETA(6)*(WT/80)**THETA(9)\n'
            '@@ -48 +44 @@\n'
            '-TVBA = THETA(11)\n'
            '+TVBA = THETA(10)\n',
        ),
        (
            ('nonmem', 'models', 'fviii6.mod'),
            [('CL', 'WT'), ('CL', 'AGE')],
            '@@ -29,10 +29,6 @@\n'
            '-IF (PREP2.EQ.1) THEN\n'
            '-TVCL = THETA(1)*((WT/80)**THETA(7))*(1+THETA(13)*(AGE-40)) ;80 = Median of body weight\n'
            '-TVV1 = THETA(2)*((WT/80)**THETA(8)) \n'
            '-ENDIF\n'
            '-IF (PREP2.EQ.2) THEN\n'
            '-TVCL = THETA(1)*((WT/80)**THETA(7))*(1+THETA(12))*(1+THETA(13)*(AGE-40))\n'
            '-TVV1 = THETA(2)*((WT/80)**THETA(8))*(1+THETA(12))\n'
            '-ENDIF\n'
            '-TVQ  = THETA(5)*(WT/80)**THETA(9)\n'
            '-TVV2 = THETA(6)*(WT/80)**THETA(10)\n'
            '+IF (PREP2.EQ.1) TVCL = THETA(1)\n'
            '+IF (PREP2.EQ.1) TVV1 = THETA(2)*(WT/80)**THETA(7)\n'
            '+IF (PREP2.EQ.2) TVCL = THETA(1)*(THETA(11) + 1)\n'
            '+IF (PREP2.EQ.2) TVV1 = THETA(2)*(WT/80)**THETA(7)*(THETA(11) + 1)\n'
            '+TVQ = THETA(5)*(WT/80)**THETA(8)\n'
            '+TVV2 = THETA(6)*(WT/80)**THETA(9)\n'
            '@@ -48 +44 @@\n'
            '-TVBA = THETA(11)\n'
            '+TVBA = THETA(10)\n',
        ),
        (
            ('nonmem', 'models', 'fviii6.mod'),
            [('V1', 'WT')],
            '@@ -29,10 +29,6 @@\n'
            '-IF (PREP2.EQ.1) THEN\n'
            '-TVCL = THETA(1)*((WT/80)**THETA(7))*(1+THETA(13)*(AGE-40)) ;80 = Median of body weight\n'
            '-TVV1 = THETA(2)*((WT/80)**THETA(8)) \n'
            '-ENDIF\n'
            '-IF (PREP2.EQ.2) THEN\n'
            '-TVCL = THETA(1)*((WT/80)**THETA(7))*(1+THETA(12))*(1+THETA(13)*(AGE-40))\n'
            '-TVV1 = THETA(2)*((WT/80)**THETA(8))*(1+THETA(12))\n'
            '-ENDIF\n'
            '-TVQ  = THETA(5)*(WT/80)**THETA(9)\n'
            '-TVV2 = THETA(6)*(WT/80)**THETA(10)\n'
            '+IF (PREP2.EQ.1) TVCL = THETA(1)*(WT/80)**THETA(7)*(THETA(12)*(AGE - 40) + 1)\n'
            '+IF (PREP2.EQ.1) TVV1 = THETA(2)\n'
            '+IF (PREP2.EQ.2) TVCL = THETA(1)*(WT/80)**THETA(7)*(THETA(11) + 1)*(THETA(12)*(AGE - 40) + 1)\n'
            '+IF (PREP2.EQ.2) TVV1 = THETA(2)*(THETA(11) + 1)\n'
            '+TVQ = THETA(5)*(WT/80)**THETA(8)\n'
            '+TVV2 = THETA(6)*(WT/80)**THETA(9)\n'
            '@@ -48 +44 @@\n'
            '-TVBA = THETA(11)\n'
            '+TVBA = THETA(10)\n',
        ),
        (
            ('nonmem', 'models', 'fviii6.mod'),
            [('V2', 'WT')],
            '@@ -29,8 +29,4 @@\n'
            '-IF (PREP2.EQ.1) THEN\n'
            '-TVCL = THETA(1)*((WT/80)**THETA(7))*(1+THETA(13)*(AGE-40)) ;80 = Median of body weight\n'
            '-TVV1 = THETA(2)*((WT/80)**THETA(8)) \n'
            '-ENDIF\n'
            '-IF (PREP2.EQ.2) THEN\n'
            '-TVCL = THETA(1)*((WT/80)**THETA(7))*(1+THETA(12))*(1+THETA(13)*(AGE-40))\n'
            '-TVV1 = THETA(2)*((WT/80)**THETA(8))*(1+THETA(12))\n'
            '-ENDIF\n'
            '+IF (PREP2.EQ.1) TVCL = THETA(1)*(WT/80)**THETA(7)*(THETA(12)*(AGE - 40) + 1)\n'
            '+IF (PREP2.EQ.1) TVV1 = THETA(2)*(WT/80)**THETA(8)\n'
            '+IF (PREP2.EQ.2) TVCL = THETA(1)*(WT/80)**THETA(7)*(THETA(11) + 1)*(THETA(12)*(AGE - 40) + 1)\n'
            '+IF (PREP2.EQ.2) TVV1 = THETA(2)*(WT/80)**THETA(8)*(THETA(11) + 1)\n'
            '@@ -38 +34 @@\n'
            '-TVV2 = THETA(6)*(WT/80)**THETA(10)\n'
            '+TVV2 = THETA(6)\n'
            '@@ -48 +44 @@\n'
            '-TVBA = THETA(11)\n'
            '+TVBA = THETA(10)\n',
        ),
        (
            ('nonmem', 'models', 'fviii6.mod'),
            [('Q', 'WT')],
            '@@ -29,10 +29,6 @@\n'
            '-IF (PREP2.EQ.1) THEN\n'
            '-TVCL = THETA(1)*((WT/80)**THETA(7))*(1+THETA(13)*(AGE-40)) ;80 = Median of body weight\n'
            '-TVV1 = THETA(2)*((WT/80)**THETA(8)) \n'
            '-ENDIF\n'
            '-IF (PREP2.EQ.2) THEN\n'
            '-TVCL = THETA(1)*((WT/80)**THETA(7))*(1+THETA(12))*(1+THETA(13)*(AGE-40))\n'
            '-TVV1 = THETA(2)*((WT/80)**THETA(8))*(1+THETA(12))\n'
            '-ENDIF\n'
            '-TVQ  = THETA(5)*(WT/80)**THETA(9)\n'
            '-TVV2 = THETA(6)*(WT/80)**THETA(10)\n'
            '+IF (PREP2.EQ.1) TVCL = THETA(1)*(WT/80)**THETA(7)*(THETA(12)*(AGE - 40) + 1)\n'
            '+IF (PREP2.EQ.1) TVV1 = THETA(2)*(WT/80)**THETA(8)\n'
            '+IF (PREP2.EQ.2) TVCL = THETA(1)*(WT/80)**THETA(7)*(THETA(11) + 1)*(THETA(12)*(AGE - 40) + 1)\n'
            '+IF (PREP2.EQ.2) TVV1 = THETA(2)*(WT/80)**THETA(8)*(THETA(11) + 1)\n'
            '+TVQ = THETA(5)\n'
            '+TVV2 = THETA(6)*(WT/80)**THETA(9)\n'
            '@@ -48 +44 @@\n'
            '-TVBA = THETA(11)\n'
            '+TVBA = THETA(10)\n',
        ),
        (
            ('nonmem', 'models', 'fviii6.mod'),
            [('K21', 'WT')],
            '@@ -29,10 +29,6 @@\n'
            '-IF (PREP2.EQ.1) THEN\n'
            '-TVCL = THETA(1)*((WT/80)**THETA(7))*(1+THETA(13)*(AGE-40)) ;80 = Median of body weight\n'
            '-TVV1 = THETA(2)*((WT/80)**THETA(8)) \n'
            '-ENDIF\n'
            '-IF (PREP2.EQ.2) THEN\n'
            '-TVCL = THETA(1)*((WT/80)**THETA(7))*(1+THETA(12))*(1+THETA(13)*(AGE-40))\n'
            '-TVV1 = THETA(2)*((WT/80)**THETA(8))*(1+THETA(12))\n'
            '-ENDIF\n'
            '-TVQ  = THETA(5)*(WT/80)**THETA(9)\n'
            '-TVV2 = THETA(6)*(WT/80)**THETA(10)\n'
            '+IF (PREP2.EQ.1) TVCL = THETA(1)*(WT/80)**THETA(7)*(THETA(11)*(AGE - 40) + 1)\n'
            '+IF (PREP2.EQ.1) TVV1 = THETA(2)*(WT/80)**THETA(8)\n'
            '+IF (PREP2.EQ.2) TVCL = THETA(1)*(WT/80)**THETA(7)*(THETA(10) + 1)*(THETA(11)*(AGE - 40) + 1)\n'
            '+IF (PREP2.EQ.2) TVV1 = THETA(2)*(WT/80)**THETA(8)*(THETA(10) + 1)\n'
            '+TVQ = THETA(5)\n'
            '+TVV2 = THETA(6)\n'
            '@@ -48 +44 @@\n'
            '-TVBA = THETA(11)\n'
            '+TVBA = THETA(9)\n',
        ),
        (
            ('nonmem', 'models', 'fviii6.mod'),
            [('CL', 'WT'), ('V1', 'WT')],
            '@@ -29,10 +29,6 @@\n'
            '-IF (PREP2.EQ.1) THEN\n'
            '-TVCL = THETA(1)*((WT/80)**THETA(7))*(1+THETA(13)*(AGE-40)) ;80 = Median of body weight\n'
            '-TVV1 = THETA(2)*((WT/80)**THETA(8)) \n'
            '-ENDIF\n'
            '-IF (PREP2.EQ.2) THEN\n'
            '-TVCL = THETA(1)*((WT/80)**THETA(7))*(1+THETA(12))*(1+THETA(13)*(AGE-40))\n'
            '-TVV1 = THETA(2)*((WT/80)**THETA(8))*(1+THETA(12))\n'
            '-ENDIF\n'
            '-TVQ  = THETA(5)*(WT/80)**THETA(9)\n'
            '-TVV2 = THETA(6)*(WT/80)**THETA(10)\n'
            '+IF (PREP2.EQ.1) TVCL = THETA(1)*(THETA(11)*(AGE - 40) + 1)\n'
            '+IF (PREP2.EQ.1) TVV1 = THETA(2)\n'
            '+IF (PREP2.EQ.2) TVCL = THETA(1)*(THETA(10) + 1)*(THETA(11)*(AGE - 40) + 1)\n'
            '+IF (PREP2.EQ.2) TVV1 = THETA(2)*(THETA(10) + 1)\n'
            '+TVQ = THETA(5)*(WT/80)**THETA(7)\n'
            '+TVV2 = THETA(6)*(WT/80)**THETA(8)\n'
            '@@ -48 +44 @@\n'
            '-TVBA = THETA(11)\n'
            '+TVBA = THETA(9)\n',
        ),
        (
            ('nonmem', 'models', 'fviii6.mod'),
            [('V1', 'WT'), ('CL', 'AGE')],
            '@@ -29,10 +29,6 @@\n'
            '-IF (PREP2.EQ.1) THEN\n'
            '-TVCL = THETA(1)*((WT/80)**THETA(7))*(1+THETA(13)*(AGE-40)) ;80 = Median of body weight\n'
            '-TVV1 = THETA(2)*((WT/80)**THETA(8)) \n'
            '-ENDIF\n'
            '-IF (PREP2.EQ.2) THEN\n'
            '-TVCL = THETA(1)*((WT/80)**THETA(7))*(1+THETA(12))*(1+THETA(13)*(AGE-40))\n'
            '-TVV1 = THETA(2)*((WT/80)**THETA(8))*(1+THETA(12))\n'
            '-ENDIF\n'
            '-TVQ  = THETA(5)*(WT/80)**THETA(9)\n'
            '-TVV2 = THETA(6)*(WT/80)**THETA(10)\n'
            '+IF (PREP2.EQ.1) TVCL = THETA(1)*(WT/80)**THETA(7)\n'
            '+IF (PREP2.EQ.1) TVV1 = THETA(2)\n'
            '+IF (PREP2.EQ.2) TVCL = THETA(1)*(WT/80)**THETA(7)*(THETA(11) + 1)\n'
            '+IF (PREP2.EQ.2) TVV1 = THETA(2)*(THETA(11) + 1)\n'
            '+TVQ = THETA(5)*(WT/80)**THETA(8)\n'
            '+TVV2 = THETA(6)*(WT/80)**THETA(9)\n'
            '@@ -48 +44 @@\n'
            '-TVBA = THETA(11)\n'
            '+TVBA = THETA(10)\n',
        ),
        (
            ('nonmem', 'models', 'fviii6.mod'),
            [('CL', 'WT'), ('V1', 'WT'), ('V2', 'WT')],
            '@@ -29,10 +29,6 @@\n'
            '-IF (PREP2.EQ.1) THEN\n'
            '-TVCL = THETA(1)*((WT/80)**THETA(7))*(1+THETA(13)*(AGE-40)) ;80 = Median of body weight\n'
            '-TVV1 = THETA(2)*((WT/80)**THETA(8)) \n'
            '-ENDIF\n'
            '-IF (PREP2.EQ.2) THEN\n'
            '-TVCL = THETA(1)*((WT/80)**THETA(7))*(1+THETA(12))*(1+THETA(13)*(AGE-40))\n'
            '-TVV1 = THETA(2)*((WT/80)**THETA(8))*(1+THETA(12))\n'
            '-ENDIF\n'
            '-TVQ  = THETA(5)*(WT/80)**THETA(9)\n'
            '-TVV2 = THETA(6)*(WT/80)**THETA(10)\n'
            '+IF (PREP2.EQ.1) TVCL = THETA(1)*(THETA(10)*(AGE - 40) + 1)\n'
            '+IF (PREP2.EQ.1) TVV1 = THETA(2)\n'
            '+IF (PREP2.EQ.2) TVCL = THETA(1)*(THETA(9) + 1)*(THETA(10)*(AGE - 40) + 1)\n'
            '+IF (PREP2.EQ.2) TVV1 = THETA(2)*(THETA(9) + 1)\n'
            '+TVQ = THETA(5)*(WT/80)**THETA(7)\n'
            '+TVV2 = THETA(6)\n'
            '@@ -48 +44 @@\n'
            '-TVBA = THETA(11)\n'
            '+TVBA = THETA(8)\n',
        ),
        (
            ('nonmem', 'models', 'fviii6.mod'),
            [('CL', 'WT'), ('V1', 'WT'), ('Q', 'WT'), ('V2', 'WT')],
            '@@ -29,10 +29,6 @@\n'
            '-IF (PREP2.EQ.1) THEN\n'
            '-TVCL = THETA(1)*((WT/80)**THETA(7))*(1+THETA(13)*(AGE-40)) ;80 = Median of body weight\n'
            '-TVV1 = THETA(2)*((WT/80)**THETA(8)) \n'
            '-ENDIF\n'
            '-IF (PREP2.EQ.2) THEN\n'
            '-TVCL = THETA(1)*((WT/80)**THETA(7))*(1+THETA(12))*(1+THETA(13)*(AGE-40))\n'
            '-TVV1 = THETA(2)*((WT/80)**THETA(8))*(1+THETA(12))\n'
            '-ENDIF\n'
            '-TVQ  = THETA(5)*(WT/80)**THETA(9)\n'
            '-TVV2 = THETA(6)*(WT/80)**THETA(10)\n'
            '+IF (PREP2.EQ.1) TVCL = THETA(1)*(THETA(9)*(AGE - 40) + 1)\n'
            '+IF (PREP2.EQ.1) TVV1 = THETA(2)\n'
            '+IF (PREP2.EQ.2) TVCL = THETA(1)*(THETA(8) + 1)*(THETA(9)*(AGE - 40) + 1)\n'
            '+IF (PREP2.EQ.2) TVV1 = THETA(2)*(THETA(8) + 1)\n'
            '+TVQ = THETA(5)\n'
            '+TVV2 = THETA(6)\n'
            '@@ -48 +44 @@\n'
            '-TVBA = THETA(11)\n'
            '+TVBA = THETA(7)\n',
        ),
        (
            ('nonmem', 'models', 'fviii6.mod'),
            [('CL', 'WT'), ('V1', 'WT'), ('V2', 'WT'), ('CL', 'AGE')],
            '@@ -29,10 +29,6 @@\n'
            '-IF (PREP2.EQ.1) THEN\n'
            '-TVCL = THETA(1)*((WT/80)**THETA(7))*(1+THETA(13)*(AGE-40)) ;80 = Median of body weight\n'
            '-TVV1 = THETA(2)*((WT/80)**THETA(8)) \n'
            '-ENDIF\n'
            '-IF (PREP2.EQ.2) THEN\n'
            '-TVCL = THETA(1)*((WT/80)**THETA(7))*(1+THETA(12))*(1+THETA(13)*(AGE-40))\n'
            '-TVV1 = THETA(2)*((WT/80)**THETA(8))*(1+THETA(12))\n'
            '-ENDIF\n'
            '-TVQ  = THETA(5)*(WT/80)**THETA(9)\n'
            '-TVV2 = THETA(6)*(WT/80)**THETA(10)\n'
            '+IF (PREP2.EQ.1) TVCL = THETA(1)\n'
            '+IF (PREP2.EQ.1) TVV1 = THETA(2)\n'
            '+IF (PREP2.EQ.2) TVCL = THETA(1)*(THETA(9) + 1)\n'
            '+IF (PREP2.EQ.2) TVV1 = THETA(2)*(THETA(9) + 1)\n'
            '+TVQ = THETA(5)*(WT/80)**THETA(7)\n'
            '+TVV2 = THETA(6)\n'
            '@@ -48 +44 @@\n'
            '-TVBA = THETA(11)\n'
            '+TVBA = THETA(8)\n',
        ),
        (
            ('nonmem', 'models', 'fviii6.mod'),
            [('CL', 'WT'), ('V1', 'WT'), ('V2', 'WT'), ('CL', 'AGE'), ('Q', 'WT')],
            '@@ -29,10 +29,6 @@\n'
            '-IF (PREP2.EQ.1) THEN\n'
            '-TVCL = THETA(1)*((WT/80)**THETA(7))*(1+THETA(13)*(AGE-40)) ;80 = Median of body weight\n'
            '-TVV1 = THETA(2)*((WT/80)**THETA(8)) \n'
            '-ENDIF\n'
            '-IF (PREP2.EQ.2) THEN\n'
            '-TVCL = THETA(1)*((WT/80)**THETA(7))*(1+THETA(12))*(1+THETA(13)*(AGE-40))\n'
            '-TVV1 = THETA(2)*((WT/80)**THETA(8))*(1+THETA(12))\n'
            '-ENDIF\n'
            '-TVQ  = THETA(5)*(WT/80)**THETA(9)\n'
            '-TVV2 = THETA(6)*(WT/80)**THETA(10)\n'
            '+IF (PREP2.EQ.1) TVCL = THETA(1)\n'
            '+IF (PREP2.EQ.1) TVV1 = THETA(2)\n'
            '+IF (PREP2.EQ.2) TVCL = THETA(1)*(THETA(8) + 1)\n'
            '+IF (PREP2.EQ.2) TVV1 = THETA(2)*(THETA(8) + 1)\n'
            '+TVQ = THETA(5)\n'
            '+TVV2 = THETA(6)\n'
            '@@ -48 +44 @@\n'
            '-TVBA = THETA(11)\n'
            '+TVBA = THETA(7)\n',
        ),
    ],
    ids=repr,
)
def test_remove_covariate_effect(load_model_for_test, testdata, model_path, effects, expected):
    model = load_model_for_test(testdata.joinpath(*model_path))
    error_record_before = ''.join(map(str, model.internals.control_stream.get_records('ERROR')))

    for effect in effects:
        assert has_covariate_effect(model, effect[0], effect[1])

    for effect in effects:
        remove_covariate_effect(model, *effect)

    for effect in effects:
        assert not has_covariate_effect(model, effect[0], effect[1])

    model.update_source()
    error_record_after = ''.join(map(str, model.internals.control_stream.get_records('ERROR')))

    original_model = load_model_for_test(testdata.joinpath(*model_path))
    assert (
        diff(
            original_model.internals.control_stream.get_pred_pk_record(),
            model.internals.control_stream.get_pred_pk_record(),
        )
        == expected
    )
    assert error_record_after == error_record_before
