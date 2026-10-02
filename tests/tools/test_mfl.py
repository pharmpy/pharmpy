import pytest

from pharmpy.modeling import create_basic_pk_model, set_direct_effect
from pharmpy.tools.mfl.helpers import all_funcs
from pharmpy.tools.mfl.parse import get_model_features, parse
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
            'METABOLITE([BASIC, PSC]);PERIPHERALS(1..2, MET)',
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
            'COVARIATE?(@PK, @CONTINUOUS, *);COVARIATE?(@PK, @CATEGORICAL, CAT, *)',
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
            'COVARIATE?(@PD, @CONTINUOUS, *);COVARIATE(@PD, @CATEGORICAL, CAT, *)',
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
            'COVARIATE?(@PD_IIV, @CONTINUOUS, *);COVARIATE(@PD_IIV, @CATEGORICAL, CAT, *)',
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
            'COVARIATE?(@PK_IIV, @CONTINUOUS, *);COVARIATE(@PK_IIV, @CATEGORICAL, CAT, *)',
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
            'COVARIATE?(@IIV, @CONTINUOUS, *);COVARIATE(@IIV, @CATEGORICAL, CAT, *)',
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
            (
                'ABSORPTION([ZO,SEQ-ZO-FO]);'
                'ELIMINATION([MM,MIX-FO-MM]);'
                'LAGTIME(ON);'
                'TRANSITS([1,3,10],*);'
                'PERIPHERALS(1)'
            ),
        ),
        (
            (
                Elimination((Name('MM'), Name('MIX-FO-MM'))),
                Peripherals((1, 2)),
            ),
            'ELIMINATION([MM,MIX-FO-MM]);PERIPHERALS(1..2)',
        ),
        (
            (
                Covariate(Ref('IIV'), Ref('CONTINUOUS'), ('EXP',), '*'),
                Covariate(Ref('IIV'), Ref('CATEGORICAL'), ('CAT',), '*'),
            ),
            'COVARIATE(@IIV,@CONTINUOUS,EXP);COVARIATE(@IIV,@CATEGORICAL,CAT)',
        ),
        (
            (Covariate(('CL',), ('WGT',), Wildcard(), '+', Option(True)),),
            'COVARIATE?(CL,WGT,*,+)',
        ),
    ),
)
def test_stringify(statements: tuple[Statement, ...], expected: str):
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
        ('LET(CONTINUOUS, [AGE, WT]); LET(CATEGORICAL, SEX)', []),
        (
            (
                'COVARIATE([CL, MAT, VC], @CONTINUOUS, EXP, *)\n'
                'COVARIATE([CL, MAT, VC], @CATEGORICAL, CAT, +)'
            ),
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
            (
                'LET(CONTINUOUS, [AGE, WT]); LET(CATEGORICAL, SEX)\n'
                'COVARIATE?([CL, MAT, VC], @CONTINUOUS, EXP, *)\n'
                'COVARIATE?([CL, MAT, VC], @CATEGORICAL, CAT, +)'
            ),
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
            (
                'LET(CONTINUOUS, [AGE, WT]); LET(CATEGORICAL, SEX)\n'
                'COVARIATE([CL, MAT, VC], @CONTINUOUS, [EXP])\n'
                'COVARIATE([CL, MAT, VC], @CATEGORICAL, CAT, +)'
            ),
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
            (
                'LET(CONTINUOUS, AGE); LET(CATEGORICAL, SEX)\n'
                'COVARIATE?([CL], @CONTINUOUS, *)\n'
                'COVARIATE([VC], @CATEGORICAL, CAT, +)'
            ),
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
            'COVARIATE?(@IIV, @CONTINUOUS, *);COVARIATE?(*, @CATEGORICAL, CAT, *)',
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
            'COVARIATE?(@PK, @CONTINUOUS, *);COVARIATE?(@PK, @CATEGORICAL, [CAT, CAT2], *)',
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
            (
                'COVARIATE(@ABSORPTION, APGR, CAT);'
                'COVARIATE(@DISTRIBUTION, WGT, EXP);'
                'COVARIATE(@ELIMINATION, SEX, CAT)'
            ),
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
            'METABOLITE([BASIC, PSC]);PERIPHERALS(1..2, MET)',
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


@pytest.mark.parametrize(
    ('source', 'expected'),
    (
        (
            'DIRECTEFFECT(*); EFFECTCOMP(*); INDIRECTEFFECT(*, *)',
            (
                ('DIRECT', 'LINEAR'),
                ('DIRECT', 'EMAX'),
                ('DIRECT', 'SIGMOID'),
                ('DIRECT', 'STEP'),
                ('DIRECT', 'LOGLIN'),
                ('EFFECTCOMP', 'LINEAR'),
                ('EFFECTCOMP', 'EMAX'),
                ('EFFECTCOMP', 'SIGMOID'),
                ('EFFECTCOMP', 'STEP'),
                ('EFFECTCOMP', 'LOGLIN'),
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
