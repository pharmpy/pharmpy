import functools

import pytest

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
from pharmpy.tools.modelsearch.tool import _update_model_features


def test_exhaustive_algorithm():
    mfl = 'ABSORPTION(ZO);PERIPHERALS(1)'
    wf, _, model_features = exhaustive(mfl, iiv_strategy=0)
    fit_tasks = [task.name for task in wf.tasks if task.name.startswith('run')]

    assert len(fit_tasks) == 3
    assert list(model_features.values())[-1] == ('ABSORPTION(ZO)', 'PERIPHERALS(1)')


@pytest.mark.parametrize(
    'mfl, iiv_strategy, no_of_models, last_model_features',
    [
        (
            'ABSORPTION(ZO)\nPERIPHERALS(1)',
            0,
            4,
            ('PERIPHERALS(1)', 'ABSORPTION(ZO)'),
        ),
        ('ABSORPTION(ZO);TRANSITS(1)', False, 2, ('TRANSITS(1)',)),
        (
            'ABSORPTION([ZO,SEQ-ZO-FO]);PERIPHERALS(1)',
            0,
            7,
            ('PERIPHERALS(1)', 'ABSORPTION(ZO)'),
        ),
        (
            'ABSORPTION(ZO);PERIPHERALS([1, 2])',
            0,
            8,
            ('PERIPHERALS(1)', 'PERIPHERALS(2)', 'ABSORPTION(ZO)'),
        ),
        (
            'ABSORPTION(SEQ-ZO-FO);LAGTIME()',
            0,
            2,
            ('LAGTIME()',),
        ),
        (
            'ABSORPTION(ZO);LAGTIME();PERIPHERALS(1)',
            0,
            15,
            ('LAGTIME()', 'PERIPHERALS(1)', 'ABSORPTION(ZO)'),
        ),
        (
            'ABSORPTION(ZO);LAGTIME();PERIPHERALS([1,2]);ELIMINATION(ZO)',
            0,
            170,
            ('LAGTIME()', 'PERIPHERALS(1)', 'PERIPHERALS(2)', 'ELIMINATION(ZO)', 'ABSORPTION(ZO)'),
        ),
        (
            'LAGTIME();TRANSITS(1);PERIPHERALS(1)',
            1,
            7,
            ('LAGTIME()', 'PERIPHERALS(1)'),
        ),
        (
            'ABSORPTION([ZO,SEQ-ZO-FO]);ELIMINATION(MM)',
            0,
            7,
            ('ELIMINATION(MM)', 'ABSORPTION(ZO)'),
        ),
        (
            'ABSORPTION([ZO,SEQ-ZO-FO]);LAGTIME();TRANSITS([1,3,10],*);'
            'PERIPHERALS(1);ELIMINATION([MM,MIX-FO-MM])',
            0,
            246,
            ('LAGTIME()', 'PERIPHERALS(1)', 'ELIMINATION(MM)', 'ABSORPTION(ZO)'),
        ),
    ],
)
def test_exhaustive_stepwise_algorithm(mfl, iiv_strategy, no_of_models, last_model_features):
    wf, _, model_features = exhaustive_stepwise(mfl, iiv_strategy=iiv_strategy)
    fit_tasks = [task.name for task in wf.tasks if task.name.startswith('run')]

    assert len(fit_tasks) == no_of_models
    assert list(model_features.values())[-1] == last_model_features


@pytest.mark.parametrize(
    'mfl, no_of_models, last_model_features',
    [
        (
            'ABSORPTION(ZO);LAGTIME();PERIPHERALS(1)',
            12,
            'ABSORPTION(ZO)',
        ),
        (
            'ABSORPTION(ZO);LAGTIME();PERIPHERALS([1,2]);ELIMINATION(ZO)',
            52,
            'PERIPHERALS(2)',
        ),
        ('ABSORPTION([ZO,SEQ-ZO-FO]);ELIMINATION(MM)', 7, 'ABSORPTION(ZO)'),
        (
            'ABSORPTION([ZO,SEQ-ZO-FO]);LAGTIME();TRANSITS([1,3,10],*);'
            'PERIPHERALS(1);ELIMINATION([MM,MIX-FO-MM])',
            159,
            'ABSORPTION(ZO)',
        ),
    ],
)
def test_reduced_stepwise_algorithm(mfl, no_of_models, last_model_features):
    wf, _, model_features = reduced_stepwise(mfl, iiv_strategy=0)
    fit_tasks = [task.name for task in wf.tasks if task.name.startswith('run')]

    assert len(fit_tasks) == no_of_models
    assert all(task.name == 'run0' for task in wf.output_tasks)
    assert list(model_features.values())[-1] == last_model_features


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


def test_update_model_features():
    start_model = DummyModel('start', None)
    res_models = []
    model_features = dict()
    prev_model = start_model
    for i in range(3):
        res_model = DummyModel(f'candidate{i}', prev_model.name)
        res_models.append(res_model)
        model_features[res_model.name] = f'{i}'
        prev_model = res_model

    model_features_new = _update_model_features(start_model, res_models, model_features)
    assert model_features_new['candidate2'] == ('0', '1', '2')


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
