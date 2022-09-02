import functools

import pytest

from pharmpy.modeling import set_peripheral_compartments, set_zero_order_absorption
from pharmpy.tools.mfl.parse import parse
from pharmpy.tools.modelsearch.algorithms import (
    _is_allowed,
    exhaustive,
    exhaustive_stepwise,
    reduced_stepwise,
)
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
            143,
        ),
    ],
)
def test_reduced_stepwise_algorithm(mfl, no_of_models):
    wf, _ = reduced_stepwise(mfl, iiv_strategy=0)
    fit_tasks = [task.name for task in wf.tasks if task.name.startswith('run')]

    assert len(fit_tasks) == no_of_models
    assert all(task.name == 'run0' for task in wf.output_tasks)


def test_check_input(create_model_for_test, testdata):
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
    model = create_model_for_test(model_code)
    model.datainfo = model.datainfo.derive(
        path=testdata / 'nonmem' / 'models' / 'mox_simulated_normal.csv'
    )

    with pytest.raises(ValueError):
        check_input(model)


def test_is_allowed():
    features = parse('ABSORPTION(ZO);PERIPHERALS(1)')
    feat_previous = []
    feat_current, func_current = 'ABSORPTION(ZO)', set_zero_order_absorption
    assert _is_allowed(feat_current, func_current, feat_previous, features)
    assert _is_allowed(feat_current, func_current, feat_current, features) is False

    features = parse('ABSORPTION([ZO,SEQ-ZO-FO])')
    feat_previous = [
        (
            'ABSORPTION',
            'SEQ-ZO-FO',
        )
    ]
    feat_current, func_current = ('ABSORPTION', 'ZO'), set_zero_order_absorption
    assert _is_allowed(feat_current, func_current, feat_previous, features) is False

    features = parse('PERIPHERALS([1,2])')
    feat_previous = []
    feat_current, func_current = ('PERIPHERALS', 1), functools.partial(
        set_peripheral_compartments, n=1
    )
    assert _is_allowed(feat_current, func_current, feat_previous, features)
    feat_previous = [feat_current]
    feat_current, func_current = ('PERIPHERALS', 2), functools.partial(
        set_peripheral_compartments, n=2
    )
    assert _is_allowed(feat_current, func_current, feat_previous, features)

    features = parse('PERIPHERALS([1,2])')
    feat_previous = [('PERIPHERALS', 1)]
    feat_current, func_current = ('PERIPHERALS', 1), functools.partial(
        set_peripheral_compartments, n=1
    )
    assert _is_allowed(feat_current, func_current, feat_previous, features) is False

    features = parse('PERIPHERALS([1,2])')
    feat_previous = []
    feat_current, func_current = ('PERIPHERALS', 2), functools.partial(
        set_peripheral_compartments, n=2
    )
    assert _is_allowed(feat_current, func_current, feat_previous, features) is False

    features = parse('PERIPHERALS(2)')
    feat_previous = []
    feat_current, func_current = ('PERIPHERALS', 2), functools.partial(
        set_peripheral_compartments, n=2
    )
    assert _is_allowed(feat_current, func_current, feat_previous, features)
