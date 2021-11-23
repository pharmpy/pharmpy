import pytest
from pyfakefs.fake_filesystem_unittest import Patcher

import pharmpy.tools.modelsearch as ms
from pharmpy import Model
from pharmpy.plugins.nonmem import conf
from pharmpy.tools.modelsearch.algorithms import exhaustive
from pharmpy.tools.modelsearch.mfl import ModelFeatures
from pharmpy.tools.rankfuncs import ofv


def test_modelsearch(datadir):
    assert conf  # Make linter happy. Don't know why conf needs to be imported
    with Patcher(additional_skip_names=['pkgutil']) as patcher:
        fs = patcher.fs
        fs.add_real_file(datadir / 'pheno_real.mod', target_path='run1.mod')
        fs.add_real_file(datadir / 'pheno_real.phi', target_path='run1.phi')
        fs.add_real_file(datadir / 'pheno_real.lst', target_path='run1.lst')
        fs.add_real_file(datadir / 'pheno_real.ext', target_path='run1.ext')
        fs.add_real_file(datadir / 'pheno.dta', target_path='pheno.dta')
        model = Model('run1.mod')

        tool = ms.ModelSearch(model, 'stepwise', 'ABSORPTION(FO)')
        assert tool


def test_exhaustive(testdata):
    base = Model(testdata / 'nonmem' / 'pheno.mod')

    def do_nothing(model):
        model[0].modelfit_results = base.modelfit_results
        return model

    trans = 'ABSORPTION(FO)'
    res = exhaustive(base, trans, do_nothing, ofv)
    assert len(res) == 1

    res = exhaustive(base, trans, do_nothing, ofv)
    assert len(res) == 1
    assert list(res['dofv']) == [0.0]


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
        ('TRANSITS(0);PERIPHERALS(0)', {0}),
    ],
)
def test_mfl_transits(code, args):
    mfl = ModelFeatures(code)
    assert mfl.transits.args == args


@pytest.mark.parametrize(
    'code,args',
    [
        ('PERIPHERALS(0)', {0}),
        ('PERIPHERALS([0, 1])', {0, 1}),
        ('PERIPHERALS([0, 2, 4])', {0, 2, 4}),
        ('PERIPHERALS(0..1)', {0, 1}),
        ('PERIPHERALS(1..4)', {1, 2, 3, 4}),
        ('PERIPHERALS(1..4); PERIPHERALS(5)', {1, 2, 3, 4, 5}),
    ],
)
def test_mfl_peripherals(code, args):
    mfl = ModelFeatures(code)
    assert mfl.peripherals.args == args


@pytest.mark.parametrize(
    'code,args',
    [
        ('LAGTIME()', None),
    ],
)
def test_mfl_lagtime(code, args):
    mfl = ModelFeatures(code)
    assert mfl.lagtime.args == args


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
        ModelFeatures(code)
