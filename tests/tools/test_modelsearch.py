import functools

import pytest

from pharmpy.modeling import (
    add_peripheral_compartment,
    set_mixed_mm_fo_elimination,
    set_peripheral_compartments,
    set_zero_order_absorption,
    set_zero_order_elimination,
)
from pharmpy.tools.mfl.parse import parse
from pharmpy.tools.mfl.parse import parse as mfl_parse
from pharmpy.tools.modelsearch.algorithms import (
    _add_iiv_to_func,
    _is_allowed,
    exhaustive,
    exhaustive_stepwise,
    reduced_stepwise,
)
from pharmpy.tools.modelsearch.tool import create_workflow, validate_input
from pharmpy.workflows import Workflow

MINIMAL_INVALID_MFL_STRING = ''
MINIMAL_VALID_MFL_STRING = 'LAGTIME()'


def test_exhaustive_algorithm():
    mfl = 'ABSORPTION(ZO);PERIPHERALS(1)'
    search_space = mfl_parse(mfl)
    wf, _ = exhaustive(search_space, iiv_strategy='no_add')
    fit_tasks = [task.name for task in wf.tasks if task.name.startswith('run')]

    assert len(fit_tasks) == 3


@pytest.mark.parametrize(
    'mfl, iiv_strategy, no_of_models',
    [
        (
            'ABSORPTION(ZO);PERIPHERALS(1)',
            'no_add',
            4,
        ),
        ('ABSORPTION(ZO);TRANSITS(1)', False, 2),
        (
            'ABSORPTION([ZO,SEQ-ZO-FO]);PERIPHERALS(1)',
            'no_add',
            7,
        ),
        (
            'ABSORPTION(ZO);PERIPHERALS([1, 2])',
            'no_add',
            8,
        ),
        (
            'ABSORPTION(SEQ-ZO-FO);LAGTIME()',
            'no_add',
            2,
        ),
        (
            'ABSORPTION(ZO);LAGTIME();PERIPHERALS(1)',
            'no_add',
            15,
        ),
        (
            'ABSORPTION(ZO);LAGTIME();PERIPHERALS([1,2]);ELIMINATION(ZO)',
            'no_add',
            170,
        ),
        (
            'LAGTIME();TRANSITS(1);PERIPHERALS(1)',
            'diagonal',
            7,
        ),
        (
            'ABSORPTION([ZO,SEQ-ZO-FO]);ELIMINATION(MM)',
            'no_add',
            7,
        ),
        ('ABSORPTION([ZO,SEQ-ZO-FO]);PERIPHERALS(1)', 0, 7),
        (
            'LAGTIME();TRANSITS(1)',
            'no_add',
            2,
        ),
        (
            'ABSORPTION(ZO);TRANSITS(3, *)',
            'no_add',
            3,
        ),
        (
            'ABSORPTION([ZO,SEQ-ZO-FO]);LAGTIME();TRANSITS([1,3,10],*);'
            'PERIPHERALS(1);ELIMINATION([MM,MIX-FO-MM])',
            'no_add',
            246,
        ),
    ],
)
def test_exhaustive_stepwise_algorithm(mfl: str, iiv_strategy: str, no_of_models: int):
    search_space = mfl_parse(mfl)
    wf, _ = exhaustive_stepwise(search_space, iiv_strategy=iiv_strategy)
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
def test_reduced_stepwise_algorithm(mfl: str, no_of_models: int):
    search_space = mfl_parse(mfl)
    wf, _ = reduced_stepwise(search_space, iiv_strategy='no_add')
    fit_tasks = [task.name for task in wf.tasks if task.name.startswith('run')]

    assert len(fit_tasks) == no_of_models
    assert all(task.name == 'run0' for task in wf.output_tasks)


def test_validate_input_model_validation(create_model_for_test, testdata):
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
    model = model.replace(
        datainfo=model.datainfo.replace(
            path=testdata / 'nonmem' / 'models' / 'mox_simulated_normal.csv'
        )
    )

    with pytest.raises(ValueError, match='Invalid `model`'):
        validate_input(MINIMAL_VALID_MFL_STRING, 'exhaustive', model=model)


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


@pytest.mark.parametrize(
    'transform_funcs, no_of_added_etas',
    [
        (
            [set_zero_order_absorption, add_peripheral_compartment],
            2,
        ),
        (
            [set_zero_order_absorption, set_zero_order_elimination],
            1,
        ),
        (
            [set_zero_order_absorption, set_mixed_mm_fo_elimination],
            2,
        ),
        (
            [set_zero_order_absorption, functools.partial(set_peripheral_compartments, n=2)],
            4,
        ),
    ],
)
def test_add_iiv_to_func(load_model_for_test, testdata, transform_funcs, no_of_added_etas):
    model = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox2.mod')
    no_of_etas_start = len(model.random_variables)
    for func in transform_funcs:
        model = func(model)
    model = _add_iiv_to_func('add_diagonal', model)
    assert len(model.random_variables) - no_of_etas_start == no_of_added_etas


def test_create_workflow():
    assert isinstance(create_workflow(MINIMAL_VALID_MFL_STRING, 'exhaustive'), Workflow)


def test_create_workflow_with_model(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    assert isinstance(
        create_workflow(MINIMAL_VALID_MFL_STRING, 'exhaustive', model=model), Workflow
    )


def test_validate_input():
    validate_input(MINIMAL_VALID_MFL_STRING, 'exhaustive')


def test_validate_input_with_model(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    validate_input(MINIMAL_VALID_MFL_STRING, 'exhaustive', model=model)


@pytest.mark.parametrize(
    (
        'model_path',
        'arguments',
        'exception',
        'match',
    ),
    [
        (
            None,
            dict(search_space=1),
            TypeError,
            'Invalid `search_space`',
        ),
        (
            None,
            dict(search_space=MINIMAL_INVALID_MFL_STRING),
            ValueError,
            'Invalid `search_space`',
        ),
        (
            None,
            dict(search_space='LET(x, 0)'),
            ValueError,
            'Invalid `search_space`',
        ),
        (
            None,
            dict(algorithm=1),
            TypeError,
            'Invalid `algorithm`',
        ),
        (
            None,
            dict(algorithm='brute_force'),
            ValueError,
            'Invalid `algorithm`',
        ),
        (
            None,
            dict(iiv_strategy=1),
            TypeError,
            'Invalid `iiv_strategy`',
        ),
        (
            None,
            dict(iiv_strategy='delay'),
            ValueError,
            'Invalid `iiv_strategy`',
        ),
        (
            None,
            dict(rank_type=1),
            TypeError,
            'Invalid `rank_type`',
        ),
        (
            None,
            dict(rank_type='bi'),
            ValueError,
            'Invalid `rank_type`',
        ),
        (
            None,
            dict(cutoff='1'),
            TypeError,
            'Invalid `cutoff`',
        ),
        (
            None,
            dict(model=1),
            TypeError,
            'Invalid `model`',
        ),
    ],
)
def test_validate_input_raises(
    load_model_for_test,
    testdata,
    model_path,
    arguments,
    exception,
    match,
):
    model = load_model_for_test(testdata.joinpath(*model_path)) if model_path else None

    harmless_arguments = dict(
        search_space=MINIMAL_VALID_MFL_STRING,
        algorithm='exhaustive',
    )

    kwargs = {**harmless_arguments, 'model': model, **arguments}

    with pytest.raises(exception, match=match):
        validate_input(**kwargs)
