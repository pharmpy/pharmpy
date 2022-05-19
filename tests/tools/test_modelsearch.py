import functools
from io import StringIO

import pytest

from pharmpy import Model
from pharmpy.modeling import (
    set_peripheral_compartments,
    set_transit_compartments,
    set_zero_order_absorption,
)
from pharmpy.tools.modelsearch.algorithms import (
    _is_allowed,
    exhaustive,
    exhaustive_stepwise,
    reduced_stepwise,
)
from pharmpy.tools.modelsearch.mfl import ModelFeatures
from pharmpy.tools.modelsearch.tool import check_input


def test_exhaustive_algorithm():
    mfl = 'ABSORPTION(ZO);PERIPHERALS(1)'
    wf, _ = exhaustive(mfl, iiv_strategy=0)
    fit_tasks = [task.name for task in wf.tasks if task.name.startswith('run')]

    assert len(fit_tasks) == 3


@pytest.mark.parametrize(
    'mfl, iiv_strategy, no_of_models',
    [
        (
            'ABSORPTION(ZO)\nPERIPHERALS(1)',
            0,
            4,
        ),
        ('ABSORPTION(ZO);TRANSITS(1)', False, 2),
        (
            'ABSORPTION([ZO,SEQ-ZO-FO]);PERIPHERALS(1)',
            0,
            7,
        ),
        (
            'ABSORPTION(ZO);PERIPHERALS([1, 2])',
            0,
            8,
        ),
        (
            'ABSORPTION(SEQ-ZO-FO);LAGTIME()',
            0,
            2,
        ),
        (
            'ABSORPTION(ZO);LAGTIME();PERIPHERALS(1)',
            0,
            15,
        ),
        (
            'ABSORPTION(ZO);LAGTIME();PERIPHERALS([1,2]);ELIMINATION(ZO)',
            0,
            170,
        ),
        (
            'LAGTIME();TRANSITS(1);PERIPHERALS(1)',
            1,
            7,
        ),
        (
            'ABSORPTION([ZO,SEQ-ZO-FO]);ELIMINATION(MM)',
            0,
            7,
        ),
        (
            'ABSORPTION([ZO,SEQ-ZO-FO]);LAGTIME();TRANSITS([1,3,10],*);'
            'PERIPHERALS(1);ELIMINATION([MM,MIX-FO-MM])',
            0,
            246,
        ),
    ],
)
def test_exhaustive_stepwise_algorithm(mfl, iiv_strategy, no_of_models):
    wf, _ = exhaustive_stepwise(mfl, iiv_strategy=iiv_strategy)
    fit_tasks = [task.name for task in wf.tasks if task.name.startswith('run')]

    assert len(fit_tasks) == no_of_models


@pytest.mark.parametrize(
    'mfl, no_of_models',
    [
        (
            'ABSORPTION(ZO);LAGTIME();PERIPHERALS(1)',
            12,
        ),
        (
            'ABSORPTION(ZO);LAGTIME();PERIPHERALS([1,2]);ELIMINATION(ZO)',
            52,
        ),
        ('ABSORPTION([ZO,SEQ-ZO-FO]);ELIMINATION(MM)', 7),
        (
            'ABSORPTION([ZO,SEQ-ZO-FO]);LAGTIME();TRANSITS([1,3,10],*);'
            'PERIPHERALS(1);ELIMINATION([MM,MIX-FO-MM])',
            159,
        ),
    ],
)
def test_reduced_stepwise_algorithm(mfl, no_of_models):
    wf, _ = reduced_stepwise(mfl, iiv_strategy=0)
    fit_tasks = [task.name for task in wf.tasks if task.name.startswith('run')]

    assert len(fit_tasks) == no_of_models
    assert all(task.name == 'run0' for task in wf.output_tasks)


def test_check_input(testdata):
    model_code = '''$PROBLEM
$INPUT ID VISI XAT2=DROP DGRP DOSE FLAG=DROP ONO=DROP
       XIME=DROP NEUY SCR AGE SEX NYH=DROP WT DROP ACE
       DIG DIU NUMB=DROP TAD TIME VIDD=DROP CRCL AMT SS II DROP
       CMT CONO=DROP DV EVID=DROP OVID=DROP
$DATA mox_simulated_normal.csv IGNORE=@
$SUBROUTINE ADVAN1 TRANS2
$PK
CL=THETA(1)*EXP(ETA(1))
V=THETA(2)*EXP(ETA(2))
S1=V
$ERROR
Y=F+F*EPS(1)
$THETA (0,0.00469307) ; TVCL
$THETA (0,1.00916) ; TVV
$OMEGA 0.0309626  ; IVCL
$OMEGA 0.031128  ; IVV
$SIGMA 0.013241

$ESTIMATION METHOD=1 INTERACTION
'''
    model = Model.create_model(StringIO(model_code))
    model.datainfo.path = testdata / 'nonmem' / 'models' / 'mox_simulated_normal.csv'

    with pytest.raises(ValueError):
        check_input(model)


def test_is_allowed():
    features = ModelFeatures('ABSORPTION(ZO);PERIPHERALS(1)')
    feat_previous = []
    feat_current, func_current = 'ABSORPTION(ZO)', set_zero_order_absorption
    assert _is_allowed(feat_current, func_current, feat_previous, features)
    assert _is_allowed(feat_current, func_current, feat_current, features) is False

    features = ModelFeatures('ABSORPTION([ZO,SEQ-ZO-FO])')
    feat_previous = ['ABSORPTION(SEQ-ZO-FO)']
    feat_current, func_current = 'ABSORPTION(ZO)', set_zero_order_absorption
    assert _is_allowed(feat_current, func_current, feat_previous, features) is False

    features = ModelFeatures('PERIPHERALS([1,2])')
    feat_previous = []
    feat_current, func_current = 'PERIPHERALS(1)', functools.partial(
        set_peripheral_compartments, n=1
    )
    assert _is_allowed(feat_current, func_current, feat_previous, features)
    feat_previous = [feat_current]
    feat_current, func_current = 'PERIPHERALS(2)', functools.partial(
        set_peripheral_compartments, n=2
    )
    assert _is_allowed(feat_current, func_current, feat_previous, features)

    features = ModelFeatures('PERIPHERALS([1,2])')
    feat_previous = ['PERIPHERALS(1)']
    feat_current, func_current = 'PERIPHERALS(1)', functools.partial(
        set_peripheral_compartments, n=1
    )
    assert _is_allowed(feat_current, func_current, feat_previous, features) is False

    features = ModelFeatures('PERIPHERALS([1,2])')
    feat_previous = []
    feat_current, func_current = 'PERIPHERALS(2)', functools.partial(
        set_peripheral_compartments, n=2
    )
    assert _is_allowed(feat_current, func_current, feat_previous, features) is False

    features = ModelFeatures('PERIPHERALS(2)')
    feat_previous = []
    feat_current, func_current = 'PERIPHERALS(2)', functools.partial(
        set_peripheral_compartments, n=2
    )
    assert _is_allowed(feat_current, func_current, feat_previous, features)


class DummyModel:
    def __init__(self, name, parent_model):
        self.name = name
        self.parent_model = parent_model


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
    'code,args,depot',
    [
        ('TRANSITS(0)', {0}, 'DEPOT'),
        ('TRANSITS([0, 1])', {0, 1}, 'DEPOT'),
        ('TRANSITS([0, 2, 4])', {0, 2, 4}, 'DEPOT'),
        ('TRANSITS(0..1)', {0, 1}, 'DEPOT'),
        ('TRANSITS(1..4)', {1, 2, 3, 4}, 'DEPOT'),
        ('TRANSITS(1..4); TRANSITS(5)', {1, 2, 3, 4, 5}, 'DEPOT'),
        ('TRANSITS(0);PERIPHERALS(0)', {0}, 'DEPOT'),
        ('TRANSITS(1..4, DEPOT)', {1, 2, 3, 4}, 'DEPOT'),
        ('TRANSITS(1..4, NODEPOT)', {1, 2, 3, 4}, 'NODEPOT'),
        ('TRANSITS(1..4, *)', {1, 2, 3, 4}, '*'),
    ],
)
def test_mfl_transits(code, args, depot):
    mfl = ModelFeatures(code)
    assert mfl.transits.args == args
    assert mfl.transits.depot == depot


def test_mfl_transits_depot():
    mfl = ModelFeatures('TRANSITS(1, *)')
    func_depot = functools.partial(set_transit_compartments, n=1)
    assert mfl.transits._funcs['TRANSITS(1)'].keywords == func_depot.keywords
    func_nodepot = functools.partial(set_transit_compartments, n=2, keep_depot=False)
    assert mfl.transits._funcs['TRANSITS(1, NODEPOT)'].keywords == func_nodepot.keywords


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
