import pytest

from pharmpy.search.mfl import ModelFeatures


@pytest.mark.parametrize(
    'code,args',
    [
        ('ABSORPTION(FO)', {'FO'}),
        ('ABSORPTION(* )', {'FO', 'ZO', 'SEQ-ZO-FO'}),
        ('ABSORPTION([ZO,FO])', {'FO', 'ZO'}),
        ('ABSORPTION([ZO,  FO])', {'FO', 'ZO'}),
        ('ABSORPTION( [   SEQ-ZO-FO,  FO   ]  )', {'SEQ-ZO-FO', 'FO'}),
        ('absorption([zo, fo])', {'FO', 'ZO'}),
        ('ABSORPTION(FO);ABSORPTION(ZO)', {'FO', 'ZO'}),
        ('ABSORPTION(FO)\nABSORPTION([FO, SEQ-ZO-FO])', {'FO', 'SEQ-ZO-FO'}),
    ],
)
def test_mfl_absorption(code, args):
    mfl = ModelFeatures(code)
    assert mfl.absorption.args == args


@pytest.mark.parametrize(
    'code,args',
    [
        ('ELIMINATION(FO)', {'FO'}),
        ('ELIMINATION( *)', {'FO', 'ZO', 'MM', 'MIX-FO-MM'}),
        ('ELIMINATION([ZO,FO])', {'FO', 'ZO'}),
        ('ELIMINATION([ZO,  FO])', {'FO', 'ZO'}),
        ('ELIMINATION( [   MIX-FO-MM,  FO   ]  )', {'MIX-FO-MM', 'FO'}),
        ('elimination([zo, fo])', {'FO', 'ZO'}),
        ('ELIMINATION(FO);ABSORPTION(ZO)', {'FO'}),
    ],
)
def test_mfl_elimination(code, args):
    mfl = ModelFeatures(code)
    assert mfl.elimination.args == args


@pytest.mark.parametrize(
    'code,args',
    [
        ('TRANSITS(0)', {0}),
        ('TRANSITS([0, 1])', {0, 1}),
        ('TRANSITS([0, 2, 4])', {0, 2, 4}),
        ('TRANSITS(0..1)', {0, 1}),
        ('TRANSITS(1..4)', {1, 2, 3, 4}),
        ('TRANSITS(1..4); TRANSITS(5)', {1, 2, 3, 4, 5}),
    ],
)
def test_mfl_transits(code, args):
    mfl = ModelFeatures(code)
    assert mfl.transits.args == args
