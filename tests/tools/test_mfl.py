from typing import Tuple

import pytest

from pharmpy.modeling import create_basic_pk_model, set_direct_effect
from pharmpy.tools import get_model_features
from pharmpy.tools.mfl.helpers import all_funcs
from pharmpy.tools.mfl.parse import ModelFeatures, parse
from pharmpy.tools.mfl.statement.feature.absorption import Absorption
from pharmpy.tools.mfl.statement.feature.covariate import Covariate, Ref
from pharmpy.tools.mfl.statement.feature.elimination import Elimination
from pharmpy.tools.mfl.statement.feature.lagtime import LagTime
from pharmpy.tools.mfl.statement.feature.peripherals import Peripherals
from pharmpy.tools.mfl.statement.feature.symbols import Name, Option, Wildcard
from pharmpy.tools.mfl.statement.feature.transits import Transits
from pharmpy.tools.mfl.statement.statement import Statement
from pharmpy.tools.mfl.stringify import stringify


@pytest.mark.parametrize(
    ('source', 'expected'),
    (
        (
            'ABSORPTION(INST)',
            (('ABSORPTION', 'INST'),),
        ),
        (
            'ABSORPTION(FO)',
            (('ABSORPTION', 'FO'),),
        ),
        (
            'ABSORPTION(* )',
            (
                ('ABSORPTION', 'FO'),
                ('ABSORPTION', 'ZO'),
                ('ABSORPTION', 'SEQ-ZO-FO'),
                ('ABSORPTION', 'INST'),
            ),
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
        ('LAGTIME(ON)', (('LAGTIME', 'ON'),)),
        ('LAGTIME ( ON )', (('LAGTIME', 'ON'),)),
        ('LAGTIME(OFF)', (('LAGTIME', 'OFF'),)),
        ('LAGTIME([ON, OFF])', (('LAGTIME', 'OFF'), ('LAGTIME', 'ON'))),
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
            'COVARIATE([CL, MAT, VC], @CATEGORICAL, CAT2, +)\n'
            'COVARIATE([CL, MAT, VC], @CATEGORICAL, CAT, +)',
            (
                ('COVARIATE', 'CL', 'APGR', 'cat', '+', 'ADD'),
                ('COVARIATE', 'CL', 'APGR', 'cat2', '+', 'ADD'),
                ('COVARIATE', 'CL', 'WGT', 'exp', '*', 'ADD'),
                ('COVARIATE', 'MAT', 'APGR', 'cat', '+', 'ADD'),
                ('COVARIATE', 'MAT', 'APGR', 'cat2', '+', 'ADD'),
                ('COVARIATE', 'MAT', 'WGT', 'exp', '*', 'ADD'),
                ('COVARIATE', 'VC', 'APGR', 'cat', '+', 'ADD'),
                ('COVARIATE', 'VC', 'APGR', 'cat2', '+', 'ADD'),
                ('COVARIATE', 'VC', 'WGT', 'exp', '*', 'ADD'),
            ),
        ),
        (
            'LET(CONTINUOUS, [AGE, WT]); LET(CATEGORICAL, SEX)\n'
            'COVARIATE?([CL, MAT, VC], @CONTINUOUS, EXP, *)\n'
            'COVARIATE?([CL, MAT, VC], @CATEGORICAL, CAT, +)',
            (
                ('COVARIATE', 'CL', 'AGE', 'exp', '*', 'ADD'),
                ('COVARIATE', 'CL', 'SEX', 'cat', '+', 'ADD'),
                ('COVARIATE', 'CL', 'WT', 'exp', '*', 'ADD'),
                ('COVARIATE', 'MAT', 'AGE', 'exp', '*', 'ADD'),
                ('COVARIATE', 'MAT', 'SEX', 'cat', '+', 'ADD'),
                ('COVARIATE', 'MAT', 'WT', 'exp', '*', 'ADD'),
                ('COVARIATE', 'VC', 'AGE', 'exp', '*', 'ADD'),
                ('COVARIATE', 'VC', 'SEX', 'cat', '+', 'ADD'),
                ('COVARIATE', 'VC', 'WT', 'exp', '*', 'ADD'),
                ('COVARIATE', 'CL', 'AGE', 'exp', '*', 'REMOVE'),
                ('COVARIATE', 'CL', 'SEX', 'cat', '+', 'REMOVE'),
                ('COVARIATE', 'CL', 'WT', 'exp', '*', 'REMOVE'),
                ('COVARIATE', 'MAT', 'AGE', 'exp', '*', 'REMOVE'),
                ('COVARIATE', 'MAT', 'SEX', 'cat', '+', 'REMOVE'),
                ('COVARIATE', 'MAT', 'WT', 'exp', '*', 'REMOVE'),
                ('COVARIATE', 'VC', 'AGE', 'exp', '*', 'REMOVE'),
                ('COVARIATE', 'VC', 'SEX', 'cat', '+', 'REMOVE'),
                ('COVARIATE', 'VC', 'WT', 'exp', '*', 'REMOVE'),
            ),
        ),
        (
            'LET(CONTINUOUS, [AGE, WT]); LET(CATEGORICAL, SEX)\n'
            'COVARIATE([CL, MAT, VC], @CONTINUOUS, [EXP])\n'
            'COVARIATE([CL, MAT, VC], @CATEGORICAL, CAT, +)',
            (
                ('COVARIATE', 'CL', 'AGE', 'exp', '*', 'ADD'),
                ('COVARIATE', 'CL', 'SEX', 'cat', '+', 'ADD'),
                ('COVARIATE', 'CL', 'WT', 'exp', '*', 'ADD'),
                ('COVARIATE', 'MAT', 'AGE', 'exp', '*', 'ADD'),
                ('COVARIATE', 'MAT', 'SEX', 'cat', '+', 'ADD'),
                ('COVARIATE', 'MAT', 'WT', 'exp', '*', 'ADD'),
                ('COVARIATE', 'VC', 'AGE', 'exp', '*', 'ADD'),
                ('COVARIATE', 'VC', 'SEX', 'cat', '+', 'ADD'),
                ('COVARIATE', 'VC', 'WT', 'exp', '*', 'ADD'),
            ),
        ),
        (
            'LET(CONTINUOUS, AGE); LET(CATEGORICAL, SEX)\n'
            'COVARIATE?([CL], @CONTINUOUS, *)\n'
            'COVARIATE([VC], @CATEGORICAL, CAT, +)',
            (
                ('COVARIATE', 'CL', 'AGE', 'exp', '*', 'ADD'),
                ('COVARIATE', 'CL', 'AGE', 'lin', '*', 'ADD'),
                ('COVARIATE', 'CL', 'AGE', 'piece_lin', '*', 'ADD'),
                ('COVARIATE', 'CL', 'AGE', 'pow', '*', 'ADD'),
                ('COVARIATE', 'CL', 'AGE', 'exp', '*', 'REMOVE'),
                ('COVARIATE', 'CL', 'AGE', 'lin', '*', 'REMOVE'),
                ('COVARIATE', 'CL', 'AGE', 'piece_lin', '*', 'REMOVE'),
                ('COVARIATE', 'CL', 'AGE', 'pow', '*', 'REMOVE'),
                ('COVARIATE', 'VC', 'SEX', 'cat', '+', 'ADD'),
            ),
        ),
        (
            'COVARIATE?(@IIV, @CONTINUOUS, *);' 'COVARIATE?(*, @CATEGORICAL, [CAT, CAT2], *)',
            (
                ('COVARIATE', 'CL', 'APGR', 'cat', '*', 'ADD'),
                ('COVARIATE', 'CL', 'APGR', 'cat2', '*', 'ADD'),
                ('COVARIATE', 'CL', 'WGT', 'exp', '*', 'ADD'),
                ('COVARIATE', 'CL', 'WGT', 'lin', '*', 'ADD'),
                ('COVARIATE', 'CL', 'WGT', 'piece_lin', '*', 'ADD'),
                ('COVARIATE', 'CL', 'WGT', 'pow', '*', 'ADD'),
                ('COVARIATE', 'V', 'APGR', 'cat', '*', 'ADD'),
                ('COVARIATE', 'V', 'APGR', 'cat2', '*', 'ADD'),
                ('COVARIATE', 'V', 'WGT', 'exp', '*', 'ADD'),
                ('COVARIATE', 'V', 'WGT', 'lin', '*', 'ADD'),
                ('COVARIATE', 'V', 'WGT', 'piece_lin', '*', 'ADD'),
                ('COVARIATE', 'V', 'WGT', 'pow', '*', 'ADD'),
                ('COVARIATE', 'CL', 'APGR', 'cat', '*', 'REMOVE'),
                ('COVARIATE', 'CL', 'APGR', 'cat2', '*', 'REMOVE'),
                ('COVARIATE', 'CL', 'WGT', 'exp', '*', 'REMOVE'),
                ('COVARIATE', 'CL', 'WGT', 'lin', '*', 'REMOVE'),
                ('COVARIATE', 'CL', 'WGT', 'piece_lin', '*', 'REMOVE'),
                ('COVARIATE', 'CL', 'WGT', 'pow', '*', 'REMOVE'),
                ('COVARIATE', 'V', 'APGR', 'cat', '*', 'REMOVE'),
                ('COVARIATE', 'V', 'APGR', 'cat2', '*', 'REMOVE'),
                ('COVARIATE', 'V', 'WGT', 'exp', '*', 'REMOVE'),
                ('COVARIATE', 'V', 'WGT', 'lin', '*', 'REMOVE'),
                ('COVARIATE', 'V', 'WGT', 'piece_lin', '*', 'REMOVE'),
                ('COVARIATE', 'V', 'WGT', 'pow', '*', 'REMOVE'),
            ),
        ),
        (
            'COVARIATE?(@PK, @CONTINUOUS, *);' 'COVARIATE?(@PK, @CATEGORICAL, CAT, *)',
            (
                ('COVARIATE', 'CL', 'APGR', 'cat', '*', 'ADD'),
                ('COVARIATE', 'CL', 'WGT', 'exp', '*', 'ADD'),
                ('COVARIATE', 'CL', 'WGT', 'lin', '*', 'ADD'),
                ('COVARIATE', 'CL', 'WGT', 'pow', '*', 'ADD'),
                ('COVARIATE', 'CL', 'WGT', 'piece_lin', '*', 'ADD'),
                ('COVARIATE', 'V', 'APGR', 'cat', '*', 'ADD'),
                ('COVARIATE', 'V', 'WGT', 'exp', '*', 'ADD'),
                ('COVARIATE', 'V', 'WGT', 'lin', '*', 'ADD'),
                ('COVARIATE', 'V', 'WGT', 'pow', '*', 'ADD'),
                ('COVARIATE', 'V', 'WGT', 'piece_lin', '*', 'ADD'),
                ('COVARIATE', 'CL', 'APGR', 'cat', '*', 'REMOVE'),
                ('COVARIATE', 'CL', 'WGT', 'exp', '*', 'REMOVE'),
                ('COVARIATE', 'CL', 'WGT', 'lin', '*', 'REMOVE'),
                ('COVARIATE', 'CL', 'WGT', 'pow', '*', 'REMOVE'),
                ('COVARIATE', 'CL', 'WGT', 'piece_lin', '*', 'REMOVE'),
                ('COVARIATE', 'V', 'APGR', 'cat', '*', 'REMOVE'),
                ('COVARIATE', 'V', 'WGT', 'exp', '*', 'REMOVE'),
                ('COVARIATE', 'V', 'WGT', 'lin', '*', 'REMOVE'),
                ('COVARIATE', 'V', 'WGT', 'pow', '*', 'REMOVE'),
                ('COVARIATE', 'V', 'WGT', 'piece_lin', '*', 'REMOVE'),
            ),
        ),
        (
            'COVARIATE(@ABSORPTION, APGR, CAT);'
            'COVARIATE(@DISTRIBUTION, WGT, EXP);'
            'COVARIATE(@ELIMINATION, SEX, CAT)',
            (
                ('COVARIATE', 'CL', 'SEX', 'cat', '*', 'ADD'),
                ('COVARIATE', 'V', 'WGT', 'exp', '*', 'ADD'),
            ),
        ),
        (
            'COVARIATE(@BIOAVAIL, APGR, CAT)',
            (),
        ),
        (
            'METABOLITE([BASIC, PSC]);' 'PERIPHERALS(1..2, MET)',
            (
                ('METABOLITE', 'BASIC'),
                ('METABOLITE', 'PSC'),
                ('PERIPHERALS', 1, 'METABOLITE'),
                ('PERIPHERALS', 2, 'METABOLITE'),
            ),
        ),
        (
            'METABOLITE(*)',
            (
                ('METABOLITE', 'BASIC'),
                ('METABOLITE', 'PSC'),
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
    ('source', 'expected'),
    (
        (
            'COVARIATE?(@PK, @CONTINUOUS, *);' 'COVARIATE?(@PK, @CATEGORICAL, CAT, *)',
            (
                ('COVARIATE', 'CL', 'APGR', 'cat', '*', 'ADD'),
                ('COVARIATE', 'CL', 'WGT', 'exp', '*', 'ADD'),
                ('COVARIATE', 'CL', 'WGT', 'lin', '*', 'ADD'),
                ('COVARIATE', 'CL', 'WGT', 'pow', '*', 'ADD'),
                ('COVARIATE', 'CL', 'WGT', 'piece_lin', '*', 'ADD'),
                ('COVARIATE', 'V', 'APGR', 'cat', '*', 'ADD'),
                ('COVARIATE', 'V', 'WGT', 'exp', '*', 'ADD'),
                ('COVARIATE', 'V', 'WGT', 'lin', '*', 'ADD'),
                ('COVARIATE', 'V', 'WGT', 'pow', '*', 'ADD'),
                ('COVARIATE', 'V', 'WGT', 'piece_lin', '*', 'ADD'),
                ('COVARIATE', 'CL', 'APGR', 'cat', '*', 'REMOVE'),
                ('COVARIATE', 'CL', 'WGT', 'exp', '*', 'REMOVE'),
                ('COVARIATE', 'CL', 'WGT', 'lin', '*', 'REMOVE'),
                ('COVARIATE', 'CL', 'WGT', 'pow', '*', 'REMOVE'),
                ('COVARIATE', 'CL', 'WGT', 'piece_lin', '*', 'REMOVE'),
                ('COVARIATE', 'V', 'APGR', 'cat', '*', 'REMOVE'),
                ('COVARIATE', 'V', 'WGT', 'exp', '*', 'REMOVE'),
                ('COVARIATE', 'V', 'WGT', 'lin', '*', 'REMOVE'),
                ('COVARIATE', 'V', 'WGT', 'pow', '*', 'REMOVE'),
                ('COVARIATE', 'V', 'WGT', 'piece_lin', '*', 'REMOVE'),
            ),
        ),
        (
            'COVARIATE?(@PD, @CONTINUOUS, *);' 'COVARIATE(@PD, @CATEGORICAL, CAT, *)',
            (
                ('COVARIATE', 'B', 'APGR', 'cat', '*', 'ADD'),
                ('COVARIATE', 'B', 'WGT', 'exp', '*', 'ADD'),
                ('COVARIATE', 'B', 'WGT', 'lin', '*', 'ADD'),
                ('COVARIATE', 'B', 'WGT', 'pow', '*', 'ADD'),
                ('COVARIATE', 'B', 'WGT', 'piece_lin', '*', 'ADD'),
                ('COVARIATE', 'SLOPE', 'APGR', 'cat', '*', 'ADD'),
                ('COVARIATE', 'SLOPE', 'WGT', 'exp', '*', 'ADD'),
                ('COVARIATE', 'SLOPE', 'WGT', 'lin', '*', 'ADD'),
                ('COVARIATE', 'SLOPE', 'WGT', 'pow', '*', 'ADD'),
                ('COVARIATE', 'SLOPE', 'WGT', 'piece_lin', '*', 'ADD'),
                ('COVARIATE', 'B', 'WGT', 'exp', '*', 'REMOVE'),
                ('COVARIATE', 'B', 'WGT', 'lin', '*', 'REMOVE'),
                ('COVARIATE', 'B', 'WGT', 'pow', '*', 'REMOVE'),
                ('COVARIATE', 'B', 'WGT', 'piece_lin', '*', 'REMOVE'),
                ('COVARIATE', 'SLOPE', 'WGT', 'exp', '*', 'REMOVE'),
                ('COVARIATE', 'SLOPE', 'WGT', 'lin', '*', 'REMOVE'),
                ('COVARIATE', 'SLOPE', 'WGT', 'pow', '*', 'REMOVE'),
                ('COVARIATE', 'SLOPE', 'WGT', 'piece_lin', '*', 'REMOVE'),
            ),
        ),
    ),
    ids=repr,
)
def test_all_funcs_pd(load_model_for_test, pheno_path, source, expected):
    model = load_model_for_test(pheno_path)
    model = set_direct_effect(model, 'linear')
    statements = parse(source)
    funcs = all_funcs(model, statements)
    keys = funcs.keys()
    assert set(keys) == set(expected)


@pytest.mark.parametrize(
    ('source', 'expected'),
    (
        (
            'COVARIATE?(@PD_IIV, @CONTINUOUS, *);' 'COVARIATE(@PD_IIV, @CATEGORICAL, CAT, *)',
            (
                ('COVARIATE', 'SLOPE', 'APGR', 'cat', '*', 'ADD'),
                ('COVARIATE', 'SLOPE', 'WGT', 'exp', '*', 'ADD'),
                ('COVARIATE', 'SLOPE', 'WGT', 'lin', '*', 'ADD'),
                ('COVARIATE', 'SLOPE', 'WGT', 'pow', '*', 'ADD'),
                ('COVARIATE', 'SLOPE', 'WGT', 'piece_lin', '*', 'ADD'),
                ('COVARIATE', 'SLOPE', 'WGT', 'exp', '*', 'REMOVE'),
                ('COVARIATE', 'SLOPE', 'WGT', 'lin', '*', 'REMOVE'),
                ('COVARIATE', 'SLOPE', 'WGT', 'pow', '*', 'REMOVE'),
                ('COVARIATE', 'SLOPE', 'WGT', 'piece_lin', '*', 'REMOVE'),
            ),
        ),
        (
            'COVARIATE?(@PK_IIV, @CONTINUOUS, *);' 'COVARIATE(@PK_IIV, @CATEGORICAL, CAT, *)',
            (
                ('COVARIATE', 'CL', 'APGR', 'cat', '*', 'ADD'),
                ('COVARIATE', 'CL', 'WGT', 'exp', '*', 'ADD'),
                ('COVARIATE', 'CL', 'WGT', 'lin', '*', 'ADD'),
                ('COVARIATE', 'CL', 'WGT', 'pow', '*', 'ADD'),
                ('COVARIATE', 'CL', 'WGT', 'piece_lin', '*', 'ADD'),
                ('COVARIATE', 'V', 'APGR', 'cat', '*', 'ADD'),
                ('COVARIATE', 'V', 'WGT', 'exp', '*', 'ADD'),
                ('COVARIATE', 'V', 'WGT', 'lin', '*', 'ADD'),
                ('COVARIATE', 'V', 'WGT', 'pow', '*', 'ADD'),
                ('COVARIATE', 'V', 'WGT', 'piece_lin', '*', 'ADD'),
                ('COVARIATE', 'CL', 'WGT', 'exp', '*', 'REMOVE'),
                ('COVARIATE', 'CL', 'WGT', 'lin', '*', 'REMOVE'),
                ('COVARIATE', 'CL', 'WGT', 'pow', '*', 'REMOVE'),
                ('COVARIATE', 'CL', 'WGT', 'piece_lin', '*', 'REMOVE'),
                ('COVARIATE', 'V', 'WGT', 'exp', '*', 'REMOVE'),
                ('COVARIATE', 'V', 'WGT', 'lin', '*', 'REMOVE'),
                ('COVARIATE', 'V', 'WGT', 'pow', '*', 'REMOVE'),
                ('COVARIATE', 'V', 'WGT', 'piece_lin', '*', 'REMOVE'),
            ),
        ),
        (
            'COVARIATE?(@IIV, @CONTINUOUS, *);' 'COVARIATE(@IIV, @CATEGORICAL, CAT, *)',
            (
                ('COVARIATE', 'SLOPE', 'APGR', 'cat', '*', 'ADD'),
                ('COVARIATE', 'SLOPE', 'WGT', 'exp', '*', 'ADD'),
                ('COVARIATE', 'SLOPE', 'WGT', 'lin', '*', 'ADD'),
                ('COVARIATE', 'SLOPE', 'WGT', 'pow', '*', 'ADD'),
                ('COVARIATE', 'SLOPE', 'WGT', 'piece_lin', '*', 'ADD'),
                ('COVARIATE', 'CL', 'APGR', 'cat', '*', 'ADD'),
                ('COVARIATE', 'CL', 'WGT', 'exp', '*', 'ADD'),
                ('COVARIATE', 'CL', 'WGT', 'lin', '*', 'ADD'),
                ('COVARIATE', 'CL', 'WGT', 'pow', '*', 'ADD'),
                ('COVARIATE', 'CL', 'WGT', 'piece_lin', '*', 'ADD'),
                ('COVARIATE', 'V', 'APGR', 'cat', '*', 'ADD'),
                ('COVARIATE', 'V', 'WGT', 'exp', '*', 'ADD'),
                ('COVARIATE', 'V', 'WGT', 'lin', '*', 'ADD'),
                ('COVARIATE', 'V', 'WGT', 'pow', '*', 'ADD'),
                ('COVARIATE', 'V', 'WGT', 'piece_lin', '*', 'ADD'),
                ('COVARIATE', 'SLOPE', 'WGT', 'exp', '*', 'REMOVE'),
                ('COVARIATE', 'SLOPE', 'WGT', 'lin', '*', 'REMOVE'),
                ('COVARIATE', 'SLOPE', 'WGT', 'pow', '*', 'REMOVE'),
                ('COVARIATE', 'SLOPE', 'WGT', 'piece_lin', '*', 'REMOVE'),
                ('COVARIATE', 'CL', 'WGT', 'exp', '*', 'REMOVE'),
                ('COVARIATE', 'CL', 'WGT', 'lin', '*', 'REMOVE'),
                ('COVARIATE', 'CL', 'WGT', 'pow', '*', 'REMOVE'),
                ('COVARIATE', 'CL', 'WGT', 'piece_lin', '*', 'REMOVE'),
                ('COVARIATE', 'V', 'WGT', 'exp', '*', 'REMOVE'),
                ('COVARIATE', 'V', 'WGT', 'lin', '*', 'REMOVE'),
                ('COVARIATE', 'V', 'WGT', 'pow', '*', 'REMOVE'),
                ('COVARIATE', 'V', 'WGT', 'piece_lin', '*', 'REMOVE'),
            ),
        ),
    ),
    ids=repr,
)
def test_all_funcs_pd_iiv(load_model_for_test, pheno_path, source, expected):
    from pharmpy.modeling import add_iiv

    model = load_model_for_test(pheno_path)
    model = set_direct_effect(model, 'linear')
    model = add_iiv(model, 'SLOPE', 'exp')
    statements = parse(source)
    funcs = all_funcs(model, statements)
    keys = funcs.keys()
    assert set(keys) == set(expected)


@pytest.mark.parametrize(
    ('source', 'expected'),
    (
        (
            'COVARIATE(@BIOAVAIL, APGR, CAT, +)',
            (('COVARIATE', 'BIO', 'APGR', 'cat', '+', 'ADD'),),
        ),
    ),
    ids=repr,
)
def test_funcs_ivoral(source, expected):
    model = create_basic_pk_model(administration='ivoral')
    statements = parse(source)
    funcs = all_funcs(model, statements)
    keys = funcs.keys()
    assert set(keys) == set(expected)


@pytest.mark.parametrize(
    ('code'),
    (
        [
            ('ABSORPTION(ILLEGAL)'),
            ('ELIMINATION(ALSOILLEGAL)'),
            ('LAGTIME(0)'),
            ('TRANSITS(*)'),
        ],
        [
            ('COVARIATE([V], WGT, *, +)'),
            ('COVARIATE?(V, WGT, *, +)'),
        ],
    ),
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
                LagTime((Name('ON'),)),
                Transits((1, 3, 10), Wildcard()),
                Peripherals((1,)),
            ),
            'ABSORPTION([ZO,SEQ-ZO-FO]);'
            'ELIMINATION([MM,MIX-FO-MM]);'
            'LAGTIME(ON);'
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
        (
            (Covariate(('CL',), ('WGT',), Wildcard(), '+', Option(True)),),
            'COVARIATE?(CL,WGT,*,+)',
        ),
    ),
)
def test_stringify(statements: Tuple[Statement, ...], expected: str):
    result = stringify(statements)
    assert result == expected
    parsed = parse(result)
    assert tuple(parsed) == statements


def test_get_model_features(load_model_for_test, pheno_path):
    model = load_model_for_test(pheno_path)
    assert (
        'ABSORPTION(INST);ELIMINATION(FO);COVARIATE([CL, V],WGT,CUSTOM,*);COVARIATE([V],APGR,CUSTOM,*)'
        == get_model_features(model)
    )


@pytest.mark.parametrize(
    ('source', 'expected'),
    (
        (
            'ABSORPTION(INST)',
            (
                ('ABSORPTION', 'INST'),
                ('ELIMINATION', 'FO'),
                ('TRANSITS', 0, 'DEPOT'),
                ('PERIPHERALS', 0),
                ('LAGTIME', 'OFF'),
            ),
        ),
        (
            'ABSORPTION(FO)',
            (
                ('ABSORPTION', 'FO'),
                ('ELIMINATION', 'FO'),
                ('TRANSITS', 0, 'DEPOT'),
                ('PERIPHERALS', 0),
                ('LAGTIME', 'OFF'),
            ),
        ),
        (
            'ABSORPTION(* )',
            (
                ('ABSORPTION', 'FO'),
                ('ABSORPTION', 'ZO'),
                ('ABSORPTION', 'SEQ-ZO-FO'),
                ('ABSORPTION', 'INST'),
                ('ELIMINATION', 'FO'),
                ('TRANSITS', 0, 'DEPOT'),
                ('PERIPHERALS', 0),
                ('LAGTIME', 'OFF'),
            ),
        ),
        (
            'ABSORPTION([ZO,FO])',
            (
                ('ABSORPTION', 'FO'),
                ('ABSORPTION', 'ZO'),
                ('ELIMINATION', 'FO'),
                ('TRANSITS', 0, 'DEPOT'),
                ('PERIPHERALS', 0),
                ('LAGTIME', 'OFF'),
            ),
        ),
        (
            'ABSORPTION([ZO,  FO])',
            (
                ('ABSORPTION', 'FO'),
                ('ABSORPTION', 'ZO'),
                ('ELIMINATION', 'FO'),
                ('TRANSITS', 0, 'DEPOT'),
                ('PERIPHERALS', 0),
                ('LAGTIME', 'OFF'),
            ),
        ),
        (
            'ABSORPTION( [   SEQ-ZO-FO,  FO   ]  )',
            (
                ('ABSORPTION', 'FO'),
                ('ABSORPTION', 'SEQ-ZO-FO'),
                ('ELIMINATION', 'FO'),
                ('TRANSITS', 0, 'DEPOT'),
                ('PERIPHERALS', 0),
                ('LAGTIME', 'OFF'),
            ),
        ),
        (
            'ABSORPTION([zo, fo])',
            (
                ('ABSORPTION', 'FO'),
                ('ABSORPTION', 'ZO'),
                ('ELIMINATION', 'FO'),
                ('TRANSITS', 0, 'DEPOT'),
                ('PERIPHERALS', 0),
                ('LAGTIME', 'OFF'),
            ),
        ),
        (
            'ABSORPTION(FO);ABSORPTION(ZO)',
            (
                ('ABSORPTION', 'FO'),
                ('ABSORPTION', 'ZO'),
                ('ELIMINATION', 'FO'),
                ('TRANSITS', 0, 'DEPOT'),
                ('PERIPHERALS', 0),
                ('LAGTIME', 'OFF'),
            ),
        ),
        (
            'ABSORPTION(FO)\nABSORPTION([FO, SEQ-ZO-FO])',
            (
                ('ABSORPTION', 'FO'),
                ('ABSORPTION', 'SEQ-ZO-FO'),
                ('ELIMINATION', 'FO'),
                ('TRANSITS', 0, 'DEPOT'),
                ('PERIPHERALS', 0),
                ('LAGTIME', 'OFF'),
            ),
        ),
        (
            'ELIMINATION(FO)',
            (
                ('ELIMINATION', 'FO'),
                ('ABSORPTION', 'INST'),
                ('TRANSITS', 0, 'DEPOT'),
                ('PERIPHERALS', 0),
                ('LAGTIME', 'OFF'),
            ),
        ),
        (
            'ELIMINATION( *)',
            (
                ('ELIMINATION', 'FO'),
                ('ELIMINATION', 'ZO'),
                ('ELIMINATION', 'MM'),
                ('ELIMINATION', 'MIX-FO-MM'),
                ('ABSORPTION', 'INST'),
                ('TRANSITS', 0, 'DEPOT'),
                ('PERIPHERALS', 0),
                ('LAGTIME', 'OFF'),
            ),
        ),
        (
            'ELIMINATION([ZO,FO])',
            (
                ('ELIMINATION', 'FO'),
                ('ELIMINATION', 'ZO'),
                ('ABSORPTION', 'INST'),
                ('TRANSITS', 0, 'DEPOT'),
                ('PERIPHERALS', 0),
                ('LAGTIME', 'OFF'),
            ),
        ),
        (
            'ELIMINATION([ZO,  FO])',
            (
                ('ELIMINATION', 'FO'),
                ('ELIMINATION', 'ZO'),
                ('ABSORPTION', 'INST'),
                ('TRANSITS', 0, 'DEPOT'),
                ('PERIPHERALS', 0),
                ('LAGTIME', 'OFF'),
            ),
        ),
        (
            'ELIMINATION( [   MIX-FO-MM,  FO   ]  )',
            (
                ('ELIMINATION', 'FO'),
                ('ELIMINATION', 'MIX-FO-MM'),
                ('ABSORPTION', 'INST'),
                ('TRANSITS', 0, 'DEPOT'),
                ('PERIPHERALS', 0),
                ('LAGTIME', 'OFF'),
            ),
        ),
        (
            'elimination([zo, fo])',
            (
                ('ELIMINATION', 'FO'),
                ('ELIMINATION', 'ZO'),
                ('ABSORPTION', 'INST'),
                ('TRANSITS', 0, 'DEPOT'),
                ('PERIPHERALS', 0),
                ('LAGTIME', 'OFF'),
            ),
        ),
        (
            'ELIMINATION(FO);ABSORPTION(ZO)',
            (
                ('ELIMINATION', 'FO'),
                ('ABSORPTION', 'ZO'),
                ('TRANSITS', 0, 'DEPOT'),
                ('PERIPHERALS', 0),
                ('LAGTIME', 'OFF'),
            ),
        ),
        (
            'TRANSITS(0)',
            (
                ('TRANSITS', 0, 'DEPOT'),
                ('ABSORPTION', 'INST'),
                ('ELIMINATION', 'FO'),
                ('PERIPHERALS', 0),
                ('LAGTIME', 'OFF'),
            ),
        ),
        (
            'TRANSITS([0, 1])',
            (
                ('TRANSITS', 0, 'DEPOT'),
                ('TRANSITS', 1, 'DEPOT'),
                ('ABSORPTION', 'INST'),
                ('ELIMINATION', 'FO'),
                ('PERIPHERALS', 0),
                ('LAGTIME', 'OFF'),
            ),
        ),
        (
            'TRANSITS([0, 2, 4])',
            (
                ('TRANSITS', 0, 'DEPOT'),
                ('TRANSITS', 2, 'DEPOT'),
                ('TRANSITS', 4, 'DEPOT'),
                ('ABSORPTION', 'INST'),
                ('ELIMINATION', 'FO'),
                ('PERIPHERALS', 0),
                ('LAGTIME', 'OFF'),
            ),
        ),
        (
            'TRANSITS(0..1)',
            (
                ('TRANSITS', 0, 'DEPOT'),
                ('TRANSITS', 1, 'DEPOT'),
                ('ABSORPTION', 'INST'),
                ('ELIMINATION', 'FO'),
                ('PERIPHERALS', 0),
                ('LAGTIME', 'OFF'),
            ),
        ),
        (
            'TRANSITS(1..4)',
            (
                ('TRANSITS', 1, 'DEPOT'),
                ('TRANSITS', 2, 'DEPOT'),
                ('TRANSITS', 3, 'DEPOT'),
                ('TRANSITS', 4, 'DEPOT'),
                ('ABSORPTION', 'INST'),
                ('ELIMINATION', 'FO'),
                ('PERIPHERALS', 0),
                ('LAGTIME', 'OFF'),
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
                ('ABSORPTION', 'INST'),
                ('ELIMINATION', 'FO'),
                ('PERIPHERALS', 0),
                ('LAGTIME', 'OFF'),
            ),
        ),
        (
            'TRANSITS(0);PERIPHERALS(0)',
            (
                ('TRANSITS', 0, 'DEPOT'),
                ('PERIPHERALS', 0),
                ('ABSORPTION', 'INST'),
                ('ELIMINATION', 'FO'),
                ('LAGTIME', 'OFF'),
            ),
        ),
        (
            'TRANSITS(1..4, DEPOT)',
            (
                ('TRANSITS', 1, 'DEPOT'),
                ('TRANSITS', 2, 'DEPOT'),
                ('TRANSITS', 3, 'DEPOT'),
                ('TRANSITS', 4, 'DEPOT'),
                ('ABSORPTION', 'INST'),
                ('ELIMINATION', 'FO'),
                ('PERIPHERALS', 0),
                ('LAGTIME', 'OFF'),
            ),
        ),
        (
            'TRANSITS(1..4, NODEPOT)',
            (
                ('TRANSITS', 1, 'NODEPOT'),
                ('TRANSITS', 2, 'NODEPOT'),
                ('TRANSITS', 3, 'NODEPOT'),
                ('TRANSITS', 4, 'NODEPOT'),
                ('ABSORPTION', 'INST'),
                ('ELIMINATION', 'FO'),
                ('PERIPHERALS', 0),
                ('LAGTIME', 'OFF'),
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
                ('ABSORPTION', 'INST'),
                ('ELIMINATION', 'FO'),
                ('PERIPHERALS', 0),
                ('LAGTIME', 'OFF'),
            ),
        ),
        (
            'PERIPHERALS(0)',
            (
                ('PERIPHERALS', 0),
                ('ABSORPTION', 'INST'),
                ('ELIMINATION', 'FO'),
                ('TRANSITS', 0, 'DEPOT'),
                ('LAGTIME', 'OFF'),
            ),
        ),
        (
            'PERIPHERALS([0, 1])',
            (
                ('PERIPHERALS', 0),
                ('PERIPHERALS', 1),
                ('ABSORPTION', 'INST'),
                ('ELIMINATION', 'FO'),
                ('TRANSITS', 0, 'DEPOT'),
                ('LAGTIME', 'OFF'),
            ),
        ),
        (
            'PERIPHERALS([0, 2, 4])',
            (
                ('PERIPHERALS', 0),
                ('PERIPHERALS', 2),
                ('PERIPHERALS', 4),
                ('ABSORPTION', 'INST'),
                ('ELIMINATION', 'FO'),
                ('TRANSITS', 0, 'DEPOT'),
                ('LAGTIME', 'OFF'),
            ),
        ),
        (
            'PERIPHERALS(0..1)',
            (
                ('PERIPHERALS', 0),
                ('PERIPHERALS', 1),
                ('ABSORPTION', 'INST'),
                ('ELIMINATION', 'FO'),
                ('TRANSITS', 0, 'DEPOT'),
                ('LAGTIME', 'OFF'),
            ),
        ),
        (
            'PERIPHERALS(1..4)',
            (
                ('PERIPHERALS', 1),
                ('PERIPHERALS', 2),
                ('PERIPHERALS', 3),
                ('PERIPHERALS', 4),
                ('ABSORPTION', 'INST'),
                ('ELIMINATION', 'FO'),
                ('TRANSITS', 0, 'DEPOT'),
                ('LAGTIME', 'OFF'),
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
                ('ABSORPTION', 'INST'),
                ('ELIMINATION', 'FO'),
                ('TRANSITS', 0, 'DEPOT'),
                ('LAGTIME', 'OFF'),
            ),
        ),
        (
            'LAGTIME(ON)',
            (
                ('LAGTIME', 'ON'),
                ('ABSORPTION', 'INST'),
                ('ELIMINATION', 'FO'),
                ('TRANSITS', 0, 'DEPOT'),
                ('PERIPHERALS', 0),
            ),
        ),
        (
            'LAGTIME ( ON )',
            (
                ('LAGTIME', 'ON'),
                ('ABSORPTION', 'INST'),
                ('ELIMINATION', 'FO'),
                ('TRANSITS', 0, 'DEPOT'),
                ('PERIPHERALS', 0),
            ),
        ),
        (
            'LAGTIME(OFF)',
            (
                ('LAGTIME', 'OFF'),
                ('ABSORPTION', 'INST'),
                ('ELIMINATION', 'FO'),
                ('TRANSITS', 0, 'DEPOT'),
                ('PERIPHERALS', 0),
            ),
        ),
        (
            'LAGTIME([ON, OFF])',
            (
                ('LAGTIME', 'OFF'),
                ('LAGTIME', 'ON'),
                ('ABSORPTION', 'INST'),
                ('ELIMINATION', 'FO'),
                ('TRANSITS', 0, 'DEPOT'),
                ('PERIPHERALS', 0),
            ),
        ),
        (
            'TRANSITS(1, *)',
            (
                ('TRANSITS', 1, 'DEPOT'),
                ('TRANSITS', 1, 'NODEPOT'),
                ('ABSORPTION', 'INST'),
                ('ELIMINATION', 'FO'),
                ('PERIPHERALS', 0),
                ('LAGTIME', 'OFF'),
            ),
        ),
        (
            'TRANSITS(1)',
            (
                ('TRANSITS', 1, 'DEPOT'),
                ('ABSORPTION', 'INST'),
                ('ELIMINATION', 'FO'),
                ('PERIPHERALS', 0),
                ('LAGTIME', 'OFF'),
            ),
        ),
        (
            'TRANSITS(1, DEPOT)',
            (
                ('TRANSITS', 1, 'DEPOT'),
                ('ABSORPTION', 'INST'),
                ('ELIMINATION', 'FO'),
                ('PERIPHERALS', 0),
                ('LAGTIME', 'OFF'),
            ),
        ),
        (
            'TRANSITS(1, NODEPOT)',
            (
                ('TRANSITS', 1, 'NODEPOT'),
                ('ABSORPTION', 'INST'),
                ('ELIMINATION', 'FO'),
                ('PERIPHERALS', 0),
                ('LAGTIME', 'OFF'),
            ),
        ),
        ('LET(CONTINUOUS, [AGE, WT]); LET(CATEGORICAL, SEX)', []),
        (
            'COVARIATE([CL, MAT, VC], @CONTINUOUS, EXP, *)\n'
            'COVARIATE([CL, MAT, VC], @CATEGORICAL, CAT, +)',
            (
                ('COVARIATE', 'CL', 'APGR', 'cat', '+', 'ADD'),
                ('COVARIATE', 'CL', 'WGT', 'exp', '*', 'ADD'),
                ('COVARIATE', 'MAT', 'APGR', 'cat', '+', 'ADD'),
                ('COVARIATE', 'MAT', 'WGT', 'exp', '*', 'ADD'),
                ('COVARIATE', 'VC', 'APGR', 'cat', '+', 'ADD'),
                ('COVARIATE', 'VC', 'WGT', 'exp', '*', 'ADD'),
            ),
        ),
        (
            'LET(CONTINUOUS, [AGE, WT]); LET(CATEGORICAL, SEX)\n'
            'COVARIATE?([CL, MAT, VC], @CONTINUOUS, EXP, *)\n'
            'COVARIATE?([CL, MAT, VC], @CATEGORICAL, CAT, +)',
            (
                ('COVARIATE', 'CL', 'AGE', 'exp', '*', 'ADD'),
                ('COVARIATE', 'CL', 'SEX', 'cat', '+', 'ADD'),
                ('COVARIATE', 'CL', 'WT', 'exp', '*', 'ADD'),
                ('COVARIATE', 'MAT', 'AGE', 'exp', '*', 'ADD'),
                ('COVARIATE', 'MAT', 'SEX', 'cat', '+', 'ADD'),
                ('COVARIATE', 'MAT', 'WT', 'exp', '*', 'ADD'),
                ('COVARIATE', 'VC', 'AGE', 'exp', '*', 'ADD'),
                ('COVARIATE', 'VC', 'SEX', 'cat', '+', 'ADD'),
                ('COVARIATE', 'VC', 'WT', 'exp', '*', 'ADD'),
                ('COVARIATE', 'CL', 'AGE', 'exp', '*', 'REMOVE'),
                ('COVARIATE', 'CL', 'SEX', 'cat', '+', 'REMOVE'),
                ('COVARIATE', 'CL', 'WT', 'exp', '*', 'REMOVE'),
                ('COVARIATE', 'MAT', 'AGE', 'exp', '*', 'REMOVE'),
                ('COVARIATE', 'MAT', 'SEX', 'cat', '+', 'REMOVE'),
                ('COVARIATE', 'MAT', 'WT', 'exp', '*', 'REMOVE'),
                ('COVARIATE', 'VC', 'AGE', 'exp', '*', 'REMOVE'),
                ('COVARIATE', 'VC', 'SEX', 'cat', '+', 'REMOVE'),
                ('COVARIATE', 'VC', 'WT', 'exp', '*', 'REMOVE'),
            ),
        ),
        (
            'LET(CONTINUOUS, [AGE, WT]); LET(CATEGORICAL, SEX)\n'
            'COVARIATE([CL, MAT, VC], @CONTINUOUS, [EXP])\n'
            'COVARIATE([CL, MAT, VC], @CATEGORICAL, CAT, +)',
            (
                ('COVARIATE', 'CL', 'AGE', 'exp', '*', 'ADD'),
                ('COVARIATE', 'CL', 'SEX', 'cat', '+', 'ADD'),
                ('COVARIATE', 'CL', 'WT', 'exp', '*', 'ADD'),
                ('COVARIATE', 'MAT', 'AGE', 'exp', '*', 'ADD'),
                ('COVARIATE', 'MAT', 'SEX', 'cat', '+', 'ADD'),
                ('COVARIATE', 'MAT', 'WT', 'exp', '*', 'ADD'),
                ('COVARIATE', 'VC', 'AGE', 'exp', '*', 'ADD'),
                ('COVARIATE', 'VC', 'SEX', 'cat', '+', 'ADD'),
                ('COVARIATE', 'VC', 'WT', 'exp', '*', 'ADD'),
            ),
        ),
        (
            'LET(CONTINUOUS, AGE); LET(CATEGORICAL, SEX)\n'
            'COVARIATE?([CL], @CONTINUOUS, *)\n'
            'COVARIATE([VC], @CATEGORICAL, CAT, +)',
            (
                ('COVARIATE', 'CL', 'AGE', 'exp', '*', 'ADD'),
                ('COVARIATE', 'CL', 'AGE', 'lin', '*', 'ADD'),
                ('COVARIATE', 'CL', 'AGE', 'piece_lin', '*', 'ADD'),
                ('COVARIATE', 'CL', 'AGE', 'pow', '*', 'ADD'),
                ('COVARIATE', 'CL', 'AGE', 'exp', '*', 'REMOVE'),
                ('COVARIATE', 'CL', 'AGE', 'lin', '*', 'REMOVE'),
                ('COVARIATE', 'CL', 'AGE', 'piece_lin', '*', 'REMOVE'),
                ('COVARIATE', 'CL', 'AGE', 'pow', '*', 'REMOVE'),
                ('COVARIATE', 'VC', 'SEX', 'cat', '+', 'ADD'),
            ),
        ),
        (
            'COVARIATE?(@IIV, @CONTINUOUS, *);' 'COVARIATE?(*, @CATEGORICAL, CAT, *)',
            (
                ('COVARIATE', 'CL', 'APGR', 'cat', '*', 'ADD'),
                ('COVARIATE', 'CL', 'WGT', 'exp', '*', 'ADD'),
                ('COVARIATE', 'CL', 'WGT', 'lin', '*', 'ADD'),
                ('COVARIATE', 'CL', 'WGT', 'piece_lin', '*', 'ADD'),
                ('COVARIATE', 'CL', 'WGT', 'pow', '*', 'ADD'),
                ('COVARIATE', 'V', 'APGR', 'cat', '*', 'ADD'),
                ('COVARIATE', 'V', 'WGT', 'exp', '*', 'ADD'),
                ('COVARIATE', 'V', 'WGT', 'lin', '*', 'ADD'),
                ('COVARIATE', 'V', 'WGT', 'piece_lin', '*', 'ADD'),
                ('COVARIATE', 'V', 'WGT', 'pow', '*', 'ADD'),
                ('COVARIATE', 'CL', 'APGR', 'cat', '*', 'REMOVE'),
                ('COVARIATE', 'CL', 'WGT', 'exp', '*', 'REMOVE'),
                ('COVARIATE', 'CL', 'WGT', 'lin', '*', 'REMOVE'),
                ('COVARIATE', 'CL', 'WGT', 'piece_lin', '*', 'REMOVE'),
                ('COVARIATE', 'CL', 'WGT', 'pow', '*', 'REMOVE'),
                ('COVARIATE', 'V', 'APGR', 'cat', '*', 'REMOVE'),
                ('COVARIATE', 'V', 'WGT', 'exp', '*', 'REMOVE'),
                ('COVARIATE', 'V', 'WGT', 'lin', '*', 'REMOVE'),
                ('COVARIATE', 'V', 'WGT', 'piece_lin', '*', 'REMOVE'),
                ('COVARIATE', 'V', 'WGT', 'pow', '*', 'REMOVE'),
            ),
        ),
        (
            'COVARIATE?(@PK, @CONTINUOUS, *);' 'COVARIATE?(@PK, @CATEGORICAL, [CAT, CAT2], *)',
            (
                ('COVARIATE', 'CL', 'APGR', 'cat', '*', 'ADD'),
                ('COVARIATE', 'CL', 'APGR', 'cat2', '*', 'ADD'),
                ('COVARIATE', 'CL', 'WGT', 'exp', '*', 'ADD'),
                ('COVARIATE', 'CL', 'WGT', 'lin', '*', 'ADD'),
                ('COVARIATE', 'CL', 'WGT', 'pow', '*', 'ADD'),
                ('COVARIATE', 'CL', 'WGT', 'piece_lin', '*', 'ADD'),
                ('COVARIATE', 'V', 'APGR', 'cat', '*', 'ADD'),
                ('COVARIATE', 'V', 'APGR', 'cat2', '*', 'ADD'),
                ('COVARIATE', 'V', 'WGT', 'exp', '*', 'ADD'),
                ('COVARIATE', 'V', 'WGT', 'lin', '*', 'ADD'),
                ('COVARIATE', 'V', 'WGT', 'pow', '*', 'ADD'),
                ('COVARIATE', 'V', 'WGT', 'piece_lin', '*', 'ADD'),
                ('COVARIATE', 'CL', 'APGR', 'cat', '*', 'REMOVE'),
                ('COVARIATE', 'CL', 'APGR', 'cat2', '*', 'REMOVE'),
                ('COVARIATE', 'CL', 'WGT', 'exp', '*', 'REMOVE'),
                ('COVARIATE', 'CL', 'WGT', 'lin', '*', 'REMOVE'),
                ('COVARIATE', 'CL', 'WGT', 'pow', '*', 'REMOVE'),
                ('COVARIATE', 'CL', 'WGT', 'piece_lin', '*', 'REMOVE'),
                ('COVARIATE', 'V', 'APGR', 'cat', '*', 'REMOVE'),
                ('COVARIATE', 'V', 'APGR', 'cat2', '*', 'REMOVE'),
                ('COVARIATE', 'V', 'WGT', 'exp', '*', 'REMOVE'),
                ('COVARIATE', 'V', 'WGT', 'lin', '*', 'REMOVE'),
                ('COVARIATE', 'V', 'WGT', 'pow', '*', 'REMOVE'),
                ('COVARIATE', 'V', 'WGT', 'piece_lin', '*', 'REMOVE'),
            ),
        ),
        (
            'COVARIATE(@ABSORPTION, APGR, CAT);'
            'COVARIATE(@DISTRIBUTION, WGT, EXP);'
            'COVARIATE(@ELIMINATION, SEX, CAT)',
            (
                ('COVARIATE', 'CL', 'SEX', 'cat', '*', 'ADD'),
                ('COVARIATE', 'V', 'WGT', 'exp', '*', 'ADD'),
            ),
        ),
        (
            'COVARIATE(@BIOAVAIL, APGR, CAT)',
            (),
        ),
        (
            'METABOLITE([BASIC, PSC]);' 'PERIPHERALS(1..2, MET)',
            (
                ('METABOLITE', 'BASIC'),
                ('METABOLITE', 'PSC'),
                ('PERIPHERALS', 1, 'METABOLITE'),
                ('PERIPHERALS', 2, 'METABOLITE'),
                ('ABSORPTION', 'INST'),
                ('ELIMINATION', 'FO'),
                ('TRANSITS', 0, 'DEPOT'),
                ('LAGTIME', 'OFF'),
            ),
        ),
        (
            'METABOLITE(*)',
            (
                ('METABOLITE', 'BASIC'),
                ('METABOLITE', 'PSC'),
                ('ABSORPTION', 'INST'),
                ('ELIMINATION', 'FO'),
                ('TRANSITS', 0, 'DEPOT'),
                ('PERIPHERALS', 0),
                ('LAGTIME', 'OFF'),
            ),
        ),
    ),
    ids=repr,
)
def test_ModelFeatures(load_model_for_test, pheno_path, source, expected):
    pheno = load_model_for_test(pheno_path)
    model_mfl = parse(source, True)
    model_mfl_funcs = model_mfl.convert_to_funcs(model=pheno)

    assert set(model_mfl_funcs.keys()) == set(expected)
    assert model_mfl.get_number_of_features(pheno) == len(expected)


def test_ModelFeatures_eq(load_model_for_test, pheno_path):
    model = load_model_for_test(pheno_path)
    model_string = get_model_features(model)
    model_mfl = ModelFeatures.create_from_mfl_string(model_string)
    mfl = parse(
        "ABSORPTION(INST);"
        "ELIMINATION(FO);"
        "TRANSITS(0,DEPOT);"
        "PERIPHERALS(0);"
        "LAGTIME(OFF)",
        True,
    )
    mfl = mfl.replace(covariate=model_mfl.covariate)
    assert mfl == model_mfl


def test_ModelFeatures_add(load_model_for_test, pheno_path):
    model = load_model_for_test(pheno_path)
    model_string = get_model_features(model)
    model_mfl = ModelFeatures.create_from_mfl_string(model_string)
    mfl = parse("ABSORPTION([FO,ZO]);PERIPHERALS(1)", True)

    expected_mfl = parse(
        "ABSORPTION([INST,FO,ZO]);"
        "ELIMINATION(FO);"
        "TRANSITS(0,DEPOT);"
        "PERIPHERALS(0..1);"
        "LAGTIME(OFF)",
        True,
    )
    expected_mfl = expected_mfl.replace(
        covariate=(
            Covariate(parameter=('CL', 'V'), covariate=('WGT',), fp=('CUSTOM',), op=('*')),
            Covariate(parameter=('V',), covariate=('APGR',), fp=('CUSTOM',), op=('*')),
        )
    )

    assert mfl + model_mfl == expected_mfl

    m1 = ModelFeatures.create_from_mfl_string("COVARIATE?(V, WGT, *)")
    m2 = ModelFeatures.create_from_mfl_string("COVARIATE?(@ELIMINATION, WGT, *)")

    with pytest.raises(
        ValueError,
        match=r'Cannot be performed with reference value. Try using .expand\(model\) first.',
    ):
        m1 + m2

    m2 = m2.expand(model)
    m3 = m1 + m2
    assert set(m3.covariate[0].parameter) == {'CL', 'V'}
    assert set(m3.covariate[0].covariate) == {'WGT'}
    assert set(m3.covariate[0].fp) == {'LIN', 'PIECE_LIN', 'EXP', 'POW'}
    assert set(m3.covariate[0].op) == {'*'}


def test_ModelFeatures_sub(load_model_for_test, pheno_path):
    model = load_model_for_test(pheno_path)
    model_string = get_model_features(model)
    model_mfl = ModelFeatures.create_from_mfl_string(model_string)
    mfl = parse("ABSORPTION([INST,ZO]);PERIPHERALS(0..1)", True)

    expected_mfl = parse(
        "ABSORPTION(ZO);" "ELIMINATION(FO);" "TRANSITS(0,DEPOT);" "PERIPHERALS(1);" "LAGTIME(OFF)",
        True,
    )

    res_mfl = mfl - model_mfl
    assert res_mfl == expected_mfl

    mfl = parse("ABSORPTION([INST,ZO]);PERIPHERALS(0..1)", True)

    expected_mfl = parse(
        "ABSORPTION(INST);"
        "ELIMINATION(FO);"
        "TRANSITS(0,DEPOT);"
        "PERIPHERALS(0);"
        "LAGTIME(OFF)",
        True,
    )
    expected_mfl = expected_mfl.replace(
        covariate=(
            Covariate(parameter=('CL', 'V'), covariate=('WGT',), fp=('CUSTOM',), op=('*')),
            Covariate(parameter=('V',), covariate=('APGR',), fp=('CUSTOM',), op=('*')),
        )
    )

    res_mfl = model_mfl - mfl
    assert res_mfl == expected_mfl

    m1 = ModelFeatures.create_from_mfl_string("COVARIATE?([CL,V], WGT, *)")
    m2 = ModelFeatures.create_from_mfl_string("COVARIATE?(@ELIMINATION, WGT, *)")

    with pytest.raises(
        ValueError,
        match=r'Cannot be performed with reference value. Try using .expand\(model\) first.',
    ):
        m1 - m2

    m2 = m2.expand(model)
    m3 = m1 - m2

    assert set(m3.covariate[0].parameter) == {'V'}
    assert set(m3.covariate[0].covariate) == {'WGT'}
    assert set(m3.covariate[0].fp) == {'LIN', 'PIECE_LIN', 'EXP', 'POW'}
    assert set(m3.covariate[0].op) == {'*'}


def test_least_number_of_transformations(load_model_for_test, pheno_path):
    model = load_model_for_test(pheno_path)
    model_string = get_model_features(model)
    model_mfl = ModelFeatures.create_from_mfl_string(model_string)
    ss = "ABSORPTION([FO,ZO]);ELIMINATION(ZO);PERIPHERALS(1);TRANSITS(1)"
    ss_mfl = parse(ss, mfl_class=True)

    lnt = model_mfl.least_number_of_transformations(ss_mfl, model)
    assert ('ABSORPTION', 'FO') in lnt
    assert ('ELIMINATION', 'ZO') in lnt
    assert ('PERIPHERALS', 1) in lnt
    assert ('TRANSITS', 1, 'DEPOT') in lnt
    assert ('COVARIATE', 'CL', 'WGT', 'custom', '*', 'REMOVE') in lnt
    assert ('COVARIATE', 'V', 'WGT', 'custom', '*', 'REMOVE') in lnt
    assert ('COVARIATE', 'V', 'APGR', 'custom', '*', 'REMOVE') in lnt

    assert len(lnt) == 7


def test_mfl_function_filtration(load_model_for_test, pheno_path):
    model = load_model_for_test(pheno_path)
    search_space = (
        "ABSORPTION(FO);"
        "ELIMINATION(ZO);"
        "LAGTIME(ON);"
        "PERIPHERALS(2);"
        "TRANSITS(1);"
        "COVARIATE(CL,WT,EXP);"
        "DIRECTEFFECT(LINEAR);"
        "INDIRECTEFFECT(LINEAR,PRODUCTION);"
        "EFFECTCOMP(LINEAR);"
        "PERIPHERALS(1,MET);"
        "METABOLITE(BASIC)"
    )
    ss_mfl = parse(search_space, True)

    funcs = ss_mfl.convert_to_funcs()
    assert len(funcs) == 11
    pk_funcs = ss_mfl.convert_to_funcs(model=model, subset_features="pk")
    assert len(pk_funcs) == 5
    pd_funcs = ss_mfl.convert_to_funcs(model=model, subset_features="pd")
    assert len(pd_funcs) == 3
    metabolite_funcs = ss_mfl.convert_to_funcs(model=model, subset_features="metabolite")
    assert len(metabolite_funcs) == 2


@pytest.mark.parametrize(
    ('search_space', 'expected'),
    (
        (
            'ABSORPTION(ZO);PERIPHERALS(1)',
            {
                'absorption': Absorption((Name('ZO'),)),
                'elimination': Elimination((Name('FO'),)),
                'peripherals': (Peripherals((1,)),),
            },
        ),
        (
            'PERIPHERALS(2);PERIPHERALS(1, MET)',
            {'peripherals': (Peripherals((2,)), Peripherals((1,), (Name('MET'),)))},
        ),
        (
            'COVARIATE(CL,WT,[EXP,POW]);COVARIATE(V,WT,CAT)',
            {
                'covariate': (
                    Covariate(
                        parameter=('CL',),
                        covariate=('WT',),
                        fp=('EXP', 'POW'),
                    ),
                    Covariate(
                        parameter=('V',),
                        covariate=('WT',),
                        fp=('CAT',),
                    ),
                )
            },
        ),
    ),
)
def test_replace_features(load_model_for_test, pheno_path, search_space, expected):
    mfl_original = parse("ABSORPTION(FO)", mfl_class=True)
    mfl_new = mfl_original.replace_features(search_space)

    for attr, attr_expected in expected.items():
        assert getattr(mfl_new, attr) == attr_expected


@pytest.mark.parametrize(
    ('source', 'expected'),
    (
        (
            'DIRECTEFFECT(*); EFFECTCOMP(*); INDIRECTEFFECT(*, *)',
            (
                ('DIRECT', 'LINEAR'),
                ('DIRECT', 'EMAX'),
                ('DIRECT', 'SIGMOID'),
                ('EFFECTCOMP', 'LINEAR'),
                ('EFFECTCOMP', 'EMAX'),
                ('EFFECTCOMP', 'SIGMOID'),
                ('INDIRECT', 'LINEAR', 'PRODUCTION'),
                ('INDIRECT', 'LINEAR', 'DEGRADATION'),
                ('INDIRECT', 'EMAX', 'PRODUCTION'),
                ('INDIRECT', 'EMAX', 'DEGRADATION'),
                ('INDIRECT', 'SIGMOID', 'PRODUCTION'),
                ('INDIRECT', 'SIGMOID', 'DEGRADATION'),
            ),
        ),
    ),
    ids=repr,
)
def test_mfl_structsearch(load_model_for_test, pheno_path, source, expected):
    model = load_model_for_test(pheno_path)
    statements = parse(source)
    funcs = all_funcs(model, statements)
    keys = funcs.keys()
    assert set(keys) == set(expected)
