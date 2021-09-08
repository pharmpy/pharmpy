import pytest
from pyfakefs.fake_filesystem_unittest import Patcher

import pharmpy.tools.modelsearch as ms
from pharmpy import Model
from pharmpy.plugins.nonmem import conf
from pharmpy.tools.modelsearch.algorithms import exhaustive
from pharmpy.tools.modelsearch.mfl import ModelFeatures
from pharmpy.tools.modelsearch.rankfuncs import aic, bic, ofv
from pharmpy.tools.workflows import Task, Workflow


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


class DummyResults:
    def __init__(self, ofv=None, aic=None, bic=None, parameter_estimates=None):
        self.ofv = ofv
        self.aic = aic
        self.bic = bic
        self.parameter_estimates = parameter_estimates

    def __bool__(self):
        return bool(self.ofv) and bool(self.parameter_estimates)


class DummyModel:
    def __init__(self, name, **kwargs):
        self.name = name
        self.modelfit_results = DummyResults(**kwargs)


@pytest.fixture
def wf_run():
    def run(model):
        return model

    return Workflow([Task('run', run)])


def test_ofv():
    run1 = DummyModel("run1", ofv=0)
    run2 = DummyModel("run2", ofv=-1)
    run3 = DummyModel("run3", ofv=-14)
    res = ofv(run1, [run2, run3])
    assert [run3] == res

    run4 = DummyModel("run4", ofv=2)
    run5 = DummyModel("run5", ofv=-2)
    res = ofv(run1, [run2, run3, run4, run5], cutoff=2)
    assert [run3, run5] == res


def test_aic():
    run1 = DummyModel("run1", aic=0)
    run2 = DummyModel("run2", aic=-1)
    run3 = DummyModel("run3", aic=-14)
    res = aic(run1, [run2, run3])
    assert [run3] == res


def test_bic():
    run1 = DummyModel("run1", bic=0)
    run2 = DummyModel("run2", bic=-1)
    run3 = DummyModel("run3", bic=-14)
    res = bic(run1, [run2, run3])
    assert [run3] == res


def test_exhaustive(testdata):
    base = Model(testdata / 'nonmem' / 'pheno.mod')

    def do_nothing(model):
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
