from typing import Tuple

import pytest

from pharmpy.tools.mfl.helpers import all_funcs
from pharmpy.tools.mfl.parse import parse
from pharmpy.tools.mfl.statement.feature.absorption import Absorption
from pharmpy.tools.mfl.statement.feature.covariate import Covariate, Ref
from pharmpy.tools.mfl.statement.feature.elimination import Elimination
from pharmpy.tools.mfl.statement.feature.lagtime import LagTime
from pharmpy.tools.mfl.statement.feature.peripherals import Peripherals
from pharmpy.tools.mfl.statement.feature.symbols import Name, Wildcard
from pharmpy.tools.mfl.statement.feature.transits import Transits
from pharmpy.tools.mfl.statement.statement import Statement
from pharmpy.tools.mfl.stringify import stringify


@pytest.mark.parametrize(
    ('source', 'expected'),
    (
        (
            'ABSORPTION(FO)',
            (('ABSORPTION', 'FO'),),
        ),
        (
            'ABSORPTION(* )',
            (('ABSORPTION', 'FO'), ('ABSORPTION', 'ZO'), ('ABSORPTION', 'SEQ-ZO-FO')),
        ),
        (
            'ABSORPTION([ZO,FO])',
            (('ABSORPTION', 'FO'), ('ABSORPTION', 'ZO')),
        ),
        (
            'ABSORPTION([ZO,  FO])',
            (('ABSORPTION', 'FO'), ('ABSORPTION', 'ZO')),
        ),
        (
            'ABSORPTION( [   SEQ-ZO-FO,  FO   ]  )',
            (('ABSORPTION', 'FO'), ('ABSORPTION', 'SEQ-ZO-FO')),
        ),
        (
            'ABSORPTION([zo, fo])',
            (('ABSORPTION', 'FO'), ('ABSORPTION', 'ZO')),
        ),
        (
            'ABSORPTION(FO);ABSORPTION(ZO)',
            (('ABSORPTION', 'FO'), ('ABSORPTION', 'ZO')),
        ),
        (
            'ABSORPTION(FO)\nABSORPTION([FO, SEQ-ZO-FO])',
            (('ABSORPTION', 'FO'), ('ABSORPTION', 'SEQ-ZO-FO')),
        ),
        (
            'ELIMINATION(FO)',
            (('ELIMINATION', 'FO'),),
        ),
        (
            'ELIMINATION( *)',
            (
                ('ELIMINATION', 'FO'),
                ('ELIMINATION', 'ZO'),
                ('ELIMINATION', 'MM'),
                ('ELIMINATION', 'MIX-FO-MM'),
            ),
        ),
        (
            'ELIMINATION([ZO,FO])',
            (('ELIMINATION', 'FO'), ('ELIMINATION', 'ZO')),
        ),
        (
            'ELIMINATION([ZO,  FO])',
            (('ELIMINATION', 'FO'), ('ELIMINATION', 'ZO')),
        ),
        (
            'ELIMINATION( [   MIX-FO-MM,  FO   ]  )',
            (('ELIMINATION', 'FO'), ('ELIMINATION', 'MIX-FO-MM')),
        ),
        (
            'elimination([zo, fo])',
            (('ELIMINATION', 'FO'), ('ELIMINATION', 'ZO')),
        ),
        (
            'ELIMINATION(FO);ABSORPTION(ZO)',
            (
                ('ELIMINATION', 'FO'),
                ('ABSORPTION', 'ZO'),
            ),
        ),
        ('TRANSITS(0)', (('TRANSITS', 0, 'DEPOT'),)),
        (
            'TRANSITS([0, 1])',
            (
                ('TRANSITS', 0, 'DEPOT'),
                ('TRANSITS', 1, 'DEPOT'),
            ),
        ),
        (
            'TRANSITS([0, 2, 4])',
            (
                ('TRANSITS', 0, 'DEPOT'),
                ('TRANSITS', 2, 'DEPOT'),
                ('TRANSITS', 4, 'DEPOT'),
            ),
        ),
        (
            'TRANSITS(0..1)',
            (
                ('TRANSITS', 0, 'DEPOT'),
                ('TRANSITS', 1, 'DEPOT'),
            ),
        ),
        (
            'TRANSITS(1..4)',
            (
                ('TRANSITS', 1, 'DEPOT'),
                ('TRANSITS', 2, 'DEPOT'),
                ('TRANSITS', 3, 'DEPOT'),
                ('TRANSITS', 4, 'DEPOT'),
            ),
        ),
        (
            'TRANSITS(1..4); TRANSITS(5)',
            (
                ('TRANSITS', 1, 'DEPOT'),
                ('TRANSITS', 2, 'DEPOT'),
                ('TRANSITS', 3, 'DEPOT'),
                ('TRANSITS', 4, 'DEPOT'),
                ('TRANSITS', 5, 'DEPOT'),
            ),
        ),
        ('TRANSITS(0);PERIPHERALS(0)', (('TRANSITS', 0, 'DEPOT'), ('PERIPHERALS', 0))),
        (
            'TRANSITS(1..4, DEPOT)',
            (
                ('TRANSITS', 1, 'DEPOT'),
                ('TRANSITS', 2, 'DEPOT'),
                ('TRANSITS', 3, 'DEPOT'),
                ('TRANSITS', 4, 'DEPOT'),
            ),
        ),
        (
            'TRANSITS(1..4, NODEPOT)',
            (
                ('TRANSITS', 1, 'NODEPOT'),
                ('TRANSITS', 2, 'NODEPOT'),
                ('TRANSITS', 3, 'NODEPOT'),
                ('TRANSITS', 4, 'NODEPOT'),
            ),
        ),
        (
            'TRANSITS(1..4, *)',
            (
                ('TRANSITS', 1, 'DEPOT'),
                ('TRANSITS', 2, 'DEPOT'),
                ('TRANSITS', 3, 'DEPOT'),
                ('TRANSITS', 4, 'DEPOT'),
                ('TRANSITS', 1, 'NODEPOT'),
                ('TRANSITS', 2, 'NODEPOT'),
                ('TRANSITS', 3, 'NODEPOT'),
                ('TRANSITS', 4, 'NODEPOT'),
            ),
        ),
        ('PERIPHERALS(0)', (('PERIPHERALS', 0),)),
        (
            'PERIPHERALS([0, 1])',
            (
                ('PERIPHERALS', 0),
                ('PERIPHERALS', 1),
            ),
        ),
        (
            'PERIPHERALS([0, 2, 4])',
            (
                ('PERIPHERALS', 0),
                ('PERIPHERALS', 2),
                ('PERIPHERALS', 4),
            ),
        ),
        (
            'PERIPHERALS(0..1)',
            (
                ('PERIPHERALS', 0),
                ('PERIPHERALS', 1),
            ),
        ),
        (
            'PERIPHERALS(1..4)',
            (
                ('PERIPHERALS', 1),
                ('PERIPHERALS', 2),
                ('PERIPHERALS', 3),
                ('PERIPHERALS', 4),
            ),
        ),
        (
            'PERIPHERALS(1..4); PERIPHERALS(5)',
            (
                ('PERIPHERALS', 1),
                ('PERIPHERALS', 2),
                ('PERIPHERALS', 3),
                ('PERIPHERALS', 4),
                ('PERIPHERALS', 5),
            ),
        ),
        ('LAGTIME()', (('LAGTIME',),)),
        ('LAGTIME ( )', (('LAGTIME',),)),
        (
            'TRANSITS(1, *)',
            (
                ('TRANSITS', 1, 'DEPOT'),
                ('TRANSITS', 1, 'NODEPOT'),
            ),
        ),
        ('TRANSITS(1)', (('TRANSITS', 1, 'DEPOT'),)),
        ('TRANSITS(1, DEPOT)', (('TRANSITS', 1, 'DEPOT'),)),
        ('TRANSITS(1, NODEPOT)', (('TRANSITS', 1, 'NODEPOT'),)),
        ('LET(CONTINUOUS, [AGE, WT]); LET(CATEGORICAL, SEX)', []),
        (
            'COVARIATE([CL, MAT, VC], @CONTINUOUS, EXP, *)\n'
            'COVARIATE([CL, MAT, VC], @CATEGORICAL, CAT, +)',
            (
                ('COVARIATE', 'CL', 'APGR', 'cat', '+'),
                ('COVARIATE', 'CL', 'WGT', 'exp', '*'),
                ('COVARIATE', 'MAT', 'APGR', 'cat', '+'),
                ('COVARIATE', 'MAT', 'WGT', 'exp', '*'),
                ('COVARIATE', 'VC', 'APGR', 'cat', '+'),
                ('COVARIATE', 'VC', 'WGT', 'exp', '*'),
            ),
        ),
        (
            'LET(CONTINUOUS, [AGE, WT]); LET(CATEGORICAL, SEX)\n'
            'COVARIATE([CL, MAT, VC], @CONTINUOUS, EXP, *)\n'
            'COVARIATE([CL, MAT, VC], @CATEGORICAL, CAT, +)',
            (
                ('COVARIATE', 'CL', 'AGE', 'exp', '*'),
                ('COVARIATE', 'CL', 'SEX', 'cat', '+'),
                ('COVARIATE', 'CL', 'WT', 'exp', '*'),
                ('COVARIATE', 'MAT', 'AGE', 'exp', '*'),
                ('COVARIATE', 'MAT', 'SEX', 'cat', '+'),
                ('COVARIATE', 'MAT', 'WT', 'exp', '*'),
                ('COVARIATE', 'VC', 'AGE', 'exp', '*'),
                ('COVARIATE', 'VC', 'SEX', 'cat', '+'),
                ('COVARIATE', 'VC', 'WT', 'exp', '*'),
            ),
        ),
        (
            'LET(CONTINUOUS, [AGE, WT]); LET(CATEGORICAL, SEX)\n'
            'COVARIATE([CL, MAT, VC], @CONTINUOUS, [EXP])\n'
            'COVARIATE([CL, MAT, VC], @CATEGORICAL, CAT, +)',
            (
                ('COVARIATE', 'CL', 'AGE', 'exp', '*'),
                ('COVARIATE', 'CL', 'SEX', 'cat', '+'),
                ('COVARIATE', 'CL', 'WT', 'exp', '*'),
                ('COVARIATE', 'MAT', 'AGE', 'exp', '*'),
                ('COVARIATE', 'MAT', 'SEX', 'cat', '+'),
                ('COVARIATE', 'MAT', 'WT', 'exp', '*'),
                ('COVARIATE', 'VC', 'AGE', 'exp', '*'),
                ('COVARIATE', 'VC', 'SEX', 'cat', '+'),
                ('COVARIATE', 'VC', 'WT', 'exp', '*'),
            ),
        ),
        (
            'LET(CONTINUOUS, AGE); LET(CATEGORICAL, SEX)\n'
            'COVARIATE([CL], @CONTINUOUS, *)\n'
            'COVARIATE([VC], @CATEGORICAL, CAT, +)',
            (
                ('COVARIATE', 'CL', 'AGE', 'exp', '*'),
                ('COVARIATE', 'CL', 'AGE', 'lin', '*'),
                ('COVARIATE', 'CL', 'AGE', 'piece_lin', '*'),
                ('COVARIATE', 'CL', 'AGE', 'pow', '*'),
                ('COVARIATE', 'VC', 'SEX', 'cat', '+'),
            ),
        ),
        (
            'COVARIATE(@IIV, @CONTINUOUS, *);' 'COVARIATE(*, @CATEGORICAL, CAT, *)',
            (
                ('COVARIATE', 'CL', 'APGR', 'cat', '*'),
                ('COVARIATE', 'CL', 'WGT', 'exp', '*'),
                ('COVARIATE', 'CL', 'WGT', 'lin', '*'),
                ('COVARIATE', 'CL', 'WGT', 'piece_lin', '*'),
                ('COVARIATE', 'CL', 'WGT', 'pow', '*'),
                ('COVARIATE', 'V', 'APGR', 'cat', '*'),
                ('COVARIATE', 'V', 'WGT', 'exp', '*'),
                ('COVARIATE', 'V', 'WGT', 'lin', '*'),
                ('COVARIATE', 'V', 'WGT', 'piece_lin', '*'),
                ('COVARIATE', 'V', 'WGT', 'pow', '*'),
            ),
        ),
        (
            'COVARIATE(@ABSORPTION, APGR, CAT);'
            'COVARIATE(@DISTRIBUTION, WGT, EXP);'
            'COVARIATE(@ELIMINATION, SEX, CAT)',
            (
                ('COVARIATE', 'CL', 'SEX', 'cat', '*'),
                ('COVARIATE', 'V', 'WGT', 'exp', '*'),
            ),
        ),
    ),
    ids=repr,
)
def test_all_funcs(load_model_for_test, pheno_path, source, expected):
    pheno = load_model_for_test(pheno_path)
    statements = parse(source)
    funcs = all_funcs(pheno, statements)
    keys = funcs.keys()
    assert set(keys) == set(expected)


@pytest.mark.parametrize(
    'code',
    [
        ('ABSORPTION(ILLEGAL)'),
        ('ELIMINATION(ALSOILLEGAL)'),
        ('LAGTIME(0)'),
        ('TRANSITS(*)'),
    ],
)
def test_illegal_mfl(code):
    with pytest.raises(Exception):
        parse(code)


@pytest.mark.parametrize(
    ('statements', 'expected'),
    (
        (
            (
                Absorption((Name('ZO'), Name('SEQ-ZO-FO'))),
                Elimination((Name('MM'), Name('MIX-FO-MM'))),
                LagTime(),
                Transits((1, 3, 10), Wildcard()),
                Peripherals((1,)),
            ),
            'ABSORPTION([ZO,SEQ-ZO-FO]);'
            'ELIMINATION([MM,MIX-FO-MM]);'
            'LAGTIME();'
            'TRANSITS([1,3,10],*);'
            'PERIPHERALS(1)',
        ),
        (
            (
                Elimination((Name('MM'), Name('MIX-FO-MM'))),
                Peripherals((1, 2)),
            ),
            'ELIMINATION([MM,MIX-FO-MM]);' 'PERIPHERALS(1..2)',
        ),
        (
            (
                Covariate(Ref('IIV'), Ref('CONTINUOUS'), ('EXP',), '*'),
                Covariate(Ref('IIV'), Ref('CATEGORICAL'), ('CAT',), '*'),
            ),
            'COVARIATE(@IIV,@CONTINUOUS,EXP);' 'COVARIATE(@IIV,@CATEGORICAL,CAT)',
        ),
    ),
)
def test_stringify(statements: Tuple[Statement, ...], expected: str):
    result = stringify(statements)
    assert result == expected
    parsed = parse(result)
    assert tuple(parsed) == statements
