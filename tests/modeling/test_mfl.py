from functools import partial

import pytest

from pharmpy.mfl import ModelFeatures
from pharmpy.modeling import add_bioavailability, add_iiv, set_direct_effect
from pharmpy.modeling.mfl import expand_model_features


@pytest.mark.parametrize(
    'funcs, source, expected',
    (
        ([], 'COVARIATE?([CL,MAT,VC],[WT,AGE],EXP)', 'COVARIATE?([CL,MAT,VC],[AGE,WT],EXP,*)'),
        ([], 'COVARIATE?(@IIV,[WT,AGE],EXP)', 'COVARIATE?([CL,MAT,VC],[AGE,WT],EXP,*)'),
        ([], 'COVARIATE?(@PK_IIV,[WT,AGE],EXP)', 'COVARIATE?([CL,MAT,VC],[AGE,WT],EXP,*)'),
        (
            [],
            'COVARIATE?([CL,MAT,VC],@CONTINUOUS,EXP)',
            'COVARIATE?([CL,MAT,VC],[AGE,CLCR,WT],EXP,*)',
        ),
        ([], 'COVARIATE?([CL,MAT,VC],@CATEGORICAL,CAT)', 'COVARIATE?([CL,MAT,VC],SEX,CAT,*)'),
        (
            [],
            'LET(CONTINUOUS,[AGE,WT]);COVARIATE?(@IIV,@CONTINUOUS,EXP)',
            'COVARIATE?([CL,MAT,VC],[AGE,WT],EXP,*)',
        ),
        (
            [],
            'COVARIATE([CL,MAT,VC],@CONTINUOUS,EXP)\n'
            'COVARIATE([CL,MAT,VC],@CATEGORICAL,CAT2,+)\n'
            'COVARIATE([CL,MAT,VC],@CATEGORICAL,CAT,+)',
            'COVARIATE([CL,MAT,VC],[AGE,CLCR,WT],EXP,*);COVARIATE([CL,MAT,VC],SEX,[CAT,CAT2],+)',
        ),
        (
            [],
            'COVARIATE?(@PK,@CONTINUOUS,EXP);COVARIATE?(@PK,@CATEGORICAL,CAT)',
            'COVARIATE?([CL,MAT,VC],[AGE,CLCR,WT],EXP,*);COVARIATE?([CL,MAT,VC],SEX,CAT,*)',
        ),
        (
            [],
            'COVARIATE(@ABSORPTION,WT,EXP);COVARIATE(@DISTRIBUTION,AGE,EXP);COVARIATE(@ELIMINATION,SEX,CAT)',
            'COVARIATE(CL,SEX,CAT,*);COVARIATE(MAT,WT,EXP,*);COVARIATE(VC,AGE,EXP,*)',
        ),
        (
            [],
            'COVARIATE(@BIOAVAIL,WT,EXP)',
            '',
        ),
        (
            [add_bioavailability],
            'COVARIATE(@BIOAVAIL,WT,EXP)',
            'COVARIATE(F_BIO,WT,EXP,*)',
        ),
        (
            [],
            'COVARIATE(@PD,WT,EXP)',
            '',
        ),
        (
            [partial(set_direct_effect, expr='linear')],
            'COVARIATE(@PD,WT,EXP)',
            'COVARIATE([B,SLOPE],WT,EXP,*)',
        ),
        (
            [],
            'COVARIATE(@PD_IIV,WT,EXP)',
            '',
        ),
        (
            [partial(set_direct_effect, expr='linear')],
            'COVARIATE(@PD_IIV,WT,EXP)',
            '',
        ),
        (
            [
                partial(set_direct_effect, expr='linear'),
                partial(add_iiv, list_of_parameters='B', expression='exp'),
            ],
            'COVARIATE(@PD_IIV,WT,EXP)',
            'COVARIATE(B,WT,EXP,*)',
        ),
    ),
)
def test_expand_model_features(load_model_for_test, testdata, funcs, source, expected):
    mf = ModelFeatures.create(source)
    model = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox2.mod')
    for func in funcs:
        model = func(model)
    mf_expanded = expand_model_features(model, mf)
    assert repr(mf_expanded) == expected


def test_expand_model_features_raises(load_model_for_test, testdata):
    mf = ModelFeatures.create('COVARIATE?(@X,[WGT,AGE],EXP)')
    model = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox2.mod')
    with pytest.raises(ValueError):
        expand_model_features(model, mf)
