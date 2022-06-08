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
    ),
    ids=repr,
)
def test_spec(pheno, source, expected):
    spec = Effects(source).spec(pheno)
    actual = sorted(list(parse_spec(spec)))
    assert actual == expected
