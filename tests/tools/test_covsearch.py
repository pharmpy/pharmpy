import pytest

from pharmpy.tools.covsearch.effects import Effects, parse_spec


@pytest.mark.parametrize(
    ('source', 'expected'),
    (
        ('CONTINUOUS([AGE, WT]); CATEGORICAL(SEX)', []),
        (
            'COVARIATE([CL, MAT, VC], @CONTINUOUS, EXP, *)\n'
            'COVARIATE([CL, MAT, VC], @CATEGORICAL, CAT, +)',
            [
                ('CL', 'APGR', 'cat', '+'),
                ('CL', 'WGT', 'exp', '*'),
                ('MAT', 'APGR', 'cat', '+'),
                ('MAT', 'WGT', 'exp', '*'),
                ('VC', 'APGR', 'cat', '+'),
                ('VC', 'WGT', 'exp', '*'),
            ],
        ),
        (
            'CONTINUOUS([AGE, WT]); CATEGORICAL(SEX)\n'
            'COVARIATE([CL, MAT, VC], @CONTINUOUS, EXP, *)\n'
            'COVARIATE([CL, MAT, VC], @CATEGORICAL, CAT, +)',
            [
                ('CL', 'AGE', 'exp', '*'),
                ('CL', 'SEX', 'cat', '+'),
                ('CL', 'WT', 'exp', '*'),
                ('MAT', 'AGE', 'exp', '*'),
                ('MAT', 'SEX', 'cat', '+'),
                ('MAT', 'WT', 'exp', '*'),
                ('VC', 'AGE', 'exp', '*'),
                ('VC', 'SEX', 'cat', '+'),
                ('VC', 'WT', 'exp', '*'),
            ],
        ),
        (
            'CONTINUOUS(AGE); CONTINUOUS(WT); CATEGORICAL(SEX)\n'
            'COVARIATE([CL, MAT, VC], @CONTINUOUS, [EXP])\n'
            'COVARIATE([CL, MAT, VC], @CATEGORICAL, CAT, +)',
            [
                ('CL', 'AGE', 'exp', '*'),
                ('CL', 'SEX', 'cat', '+'),
                ('CL', 'WT', 'exp', '*'),
                ('MAT', 'AGE', 'exp', '*'),
                ('MAT', 'SEX', 'cat', '+'),
                ('MAT', 'WT', 'exp', '*'),
                ('VC', 'AGE', 'exp', '*'),
                ('VC', 'SEX', 'cat', '+'),
                ('VC', 'WT', 'exp', '*'),
            ],
        ),
        (
            'CONTINUOUS(AGE); CATEGORICAL(SEX)\n'
            'COVARIATE([CL], @CONTINUOUS, *)\n'
            'COVARIATE([VC], @CATEGORICAL, CAT, +)',
            [
                ('CL', 'AGE', 'exp', '*'),
                ('CL', 'AGE', 'lin', '*'),
                ('CL', 'AGE', 'piece_lin', '*'),
                ('CL', 'AGE', 'pow', '*'),
                ('VC', 'SEX', 'cat', '+'),
            ],
        ),
        (
            'COVARIATE(@IIV, @CONTINUOUS, *);' 'COVARIATE(*, @CATEGORICAL, CAT, *)',
            [
                ('CL', 'APGR', 'cat', '*'),
                ('CL', 'WGT', 'exp', '*'),
                ('CL', 'WGT', 'lin', '*'),
                ('CL', 'WGT', 'piece_lin', '*'),
                ('CL', 'WGT', 'pow', '*'),
                ('V', 'APGR', 'cat', '*'),
                ('V', 'WGT', 'exp', '*'),
                ('V', 'WGT', 'lin', '*'),
                ('V', 'WGT', 'piece_lin', '*'),
                ('V', 'WGT', 'pow', '*'),
            ],
        ),
        (
            'COVARIATE(@ABSORPTION, APGR, CAT);'
            'COVARIATE(@DISTRIBUTION, WGT, EXP);'
            'COVARIATE(@ELIMINATION, SEX, CAT)',
            [
                ('CL', 'SEX', 'cat', '*'),
                ('V', 'WGT', 'exp', '*'),
            ],
        ),
    ),
    ids=repr,
)
def test_spec(pheno, source, expected):
    spec = Effects(source).spec(pheno)
    actual = sorted(list(parse_spec(spec)))
    assert actual == expected
