import functools

import pytest

from pharmpy.model import Model
from pharmpy.modeling import (
    add_lag_time,
    add_peripheral_compartment,
    has_first_order_absorption,
    has_seq_zo_fo_absorption,
    set_mixed_mm_fo_elimination,
    set_peripheral_compartments,
    set_seq_zo_fo_absorption,
    set_zero_order_absorption,
    set_zero_order_elimination,
)
from pharmpy.tools import read_modelfit_results
from pharmpy.tools.mfl.helpers import funcs, modelsearch_features
from pharmpy.tools.mfl.parse import parse
from pharmpy.tools.mfl.parse import parse as mfl_parse
from pharmpy.tools.modelsearch.algorithms import (
    _add_iiv_to_func,
    _is_allowed,
    create_candidate_exhaustive,
    create_candidate_stepwise,
    exhaustive,
    exhaustive_stepwise,
    get_best_model,
    reduced_stepwise,
)
from pharmpy.tools.modelsearch.tool import (
    categorize_model_entries,
    clear_description,
    create_base_model,
    create_workflow,
    filter_mfl_statements,
    validate_input,
)
from pharmpy.workflows import ModelEntry, Workflow

MINIMAL_INVALID_MFL_STRING = ''
MINIMAL_VALID_MFL_STRING = 'LAGTIME(ON)'


def test_exhaustive_algorithm():
    mfl = 'ABSORPTION(ZO);PERIPHERALS(1)'
    search_space = mfl_parse(mfl)
    search_space = funcs(Model(), search_space, modelsearch_features)
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
            'ABSORPTION(SEQ-ZO-FO);LAGTIME(ON)',
            'no_add',
            2,
        ),
        (
            'ABSORPTION(ZO);LAGTIME(ON);PERIPHERALS(1)',
            'no_add',
            15,
        ),
        (
            'ABSORPTION(ZO);LAGTIME(ON);PERIPHERALS([1,2]);ELIMINATION(ZO)',
            'no_add',
            170,
        ),
        (
            'LAGTIME(ON);TRANSITS(1);PERIPHERALS(1)',
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
            'LAGTIME(ON);TRANSITS(1)',
            'no_add',
            2,
        ),
        (
            'ABSORPTION(ZO);TRANSITS(3, *)',
            'no_add',
            3,
        ),
        (
            'ABSORPTION([ZO,SEQ-ZO-FO]);LAGTIME(ON);TRANSITS([1,3,10],*);'
            'PERIPHERALS(1);ELIMINATION([MM,MIX-FO-MM])',
            'no_add',
            246,
        ),
    ],
)
def test_exhaustive_stepwise_algorithm(mfl: str, iiv_strategy: str, no_of_models: int):
    search_space = mfl_parse(mfl)
    search_space = funcs(Model(), search_space, modelsearch_features)
    wf, _ = exhaustive_stepwise(search_space, iiv_strategy=iiv_strategy)
    fit_tasks = [task.name for task in wf.tasks if task.name.startswith('run')]

    assert len(fit_tasks) == no_of_models


@pytest.mark.parametrize(
    'mfl, no_of_models',
    [
        (
            'ABSORPTION(ZO);LAGTIME(ON);PERIPHERALS(1)',
            12,
        ),
        (
            'ABSORPTION(ZO);LAGTIME(ON);PERIPHERALS([1,2]);ELIMINATION(ZO)',
            52,
        ),
        ('ABSORPTION([ZO,SEQ-ZO-FO]);ELIMINATION(MM)', 7),
        (
            'ABSORPTION([ZO,SEQ-ZO-FO]);LAGTIME(ON);TRANSITS([1,3,10],*);'
            'PERIPHERALS(1);ELIMINATION([MM,MIX-FO-MM])',
            143,
        ),
    ],
)
def test_reduced_stepwise_algorithm(mfl: str, no_of_models: int):
    search_space = mfl_parse(mfl)
    search_space = funcs(Model(), search_space, modelsearch_features)
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

    # TODO: CMT column now supported, add some other test?
    # with pytest.raises(ValueError, match='Invalid `model`'):
    #    validate_input(MINIMAL_VALID_MFL_STRING, 'exhaustive', model=model)


def test_is_allowed():
    features = parse('ABSORPTION(ZO);PERIPHERALS(1)')
    features = funcs(Model(), features, modelsearch_features)
    feat_previous = []
    feat_current, func_current = 'ABSORPTION(ZO)', set_zero_order_absorption
    assert _is_allowed(feat_current, func_current, feat_previous, features)
    assert _is_allowed(feat_current, func_current, feat_current, features) is False

    features = parse('ABSORPTION([ZO,SEQ-ZO-FO])')
    features = funcs(Model(), features, modelsearch_features)
    feat_previous = [
        (
            'ABSORPTION',
            'SEQ-ZO-FO',
        )
    ]
    feat_current, func_current = ('ABSORPTION', 'ZO'), set_zero_order_absorption
    assert _is_allowed(feat_current, func_current, feat_previous, features) is False

    features = parse('PERIPHERALS([1,2])')
    features = funcs(Model(), features, modelsearch_features)
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
    features = funcs(Model(), features, modelsearch_features)
    feat_previous = [('PERIPHERALS', 1)]
    feat_current, func_current = ('PERIPHERALS', 1), functools.partial(
        set_peripheral_compartments, n=1
    )
    assert _is_allowed(feat_current, func_current, feat_previous, features) is False

    features = parse('PERIPHERALS([1,2])')
    features = funcs(Model(), features, modelsearch_features)
    feat_previous = []
    feat_current, func_current = ('PERIPHERALS', 2), functools.partial(
        set_peripheral_compartments, n=2
    )
    assert _is_allowed(feat_current, func_current, feat_previous, features) is False

    features = parse('PERIPHERALS(2)')
    features = funcs(Model(), features, modelsearch_features)
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
    res = read_modelfit_results(testdata / 'nonmem' / 'models' / 'mox2.mod')
    model_entry = ModelEntry.create(model, modelfit_results=res)
    no_of_etas_start = len(model.random_variables)
    for func in transform_funcs:
        model = func(model)
    model = _add_iiv_to_func('add_diagonal', model, model_entry)
    assert len(model.random_variables) - no_of_etas_start == no_of_added_etas


def test_get_best_model(load_model_for_test, testdata, model_entry_factory):
    model_start = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox2.mod')
    res_start = read_modelfit_results(testdata / 'nonmem' / 'models' / 'mox2.mod')
    me_start = ModelEntry.create(model_start, modelfit_results=res_start)

    model_candidate = model_start.replace(name='cand')
    me_candidate = model_entry_factory([model_candidate], ref_val=res_start.ofv)[0]

    assert get_best_model(me_start, me_candidate).model.name == 'cand'

    me_no_res_1 = ModelEntry.create(model_start)
    me_no_res_2 = ModelEntry.create(model_candidate)
    assert get_best_model(me_no_res_1, me_no_res_2).model.name == 'mox2'
    assert get_best_model(me_no_res_2, me_no_res_1).model.name == 'cand'
    assert get_best_model(me_no_res_2, me_start).model.name == 'mox2'


def test_create_base_model(load_model_for_test, testdata):
    model_start = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox2.mod')
    res_start = read_modelfit_results(testdata / 'nonmem' / 'models' / 'mox2.mod')
    me_start = ModelEntry.create(model_start, modelfit_results=res_start)
    search_space = mfl_parse('ABSORPTION([SEQ-ZO-FO])', mfl_class=True)
    assert has_first_order_absorption(model_start)
    model_base = create_base_model(search_space, None, me_start).model
    assert has_seq_zo_fo_absorption(model_base)
    assert model_base.description == 'ABSORPTION(SEQ-ZO-FO)'

    search_space = mfl_parse('ABSORPTION([FO]);PERIPHERALS(1)', mfl_class=True)
    mfl_allometry = mfl_parse('ALLOMETRY(WT, 70)', mfl_class=True).allometry
    model_base = create_base_model(search_space, mfl_allometry, me_start).model
    assert len([p for p in model_base.parameters if p.name.startswith('ALLO')]) == 4
    assert model_base.description == 'PERIPHERALS(1)'


def test_clear_description(load_model_for_test, testdata):
    model_start = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    res_start = read_modelfit_results(testdata / 'nonmem' / 'pheno.mod')
    me_start = ModelEntry.create(model_start, modelfit_results=res_start)
    model_no_desc = clear_description(me_start).model
    assert model_start.description != ''
    assert model_no_desc.description == ''


@pytest.mark.parametrize(
    ('iiv_strategy', 'allometry', 'params_added'),
    [
        ('no_add', None, {'POP_VP1', 'POP_QP1'}),
        ('add_diagonal', None, {'POP_VP1', 'POP_QP1', 'IIV_QP1', 'IIV_VP1'}),
        (
            'no_add',
            'ALLOMETRY(WT, 70)',
            {'POP_VP1', 'POP_QP1', 'ALLO_QP1', 'ALLO_CL', 'ALLO_VC', 'ALLO_VP1'},
        ),
    ],
)
def test_create_candidate(load_model_for_test, testdata, iiv_strategy, allometry, params_added):
    model_start = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox2.mod')
    res_start = read_modelfit_results(testdata / 'nonmem' / 'models' / 'mox2.mod')
    me_start = ModelEntry.create(model_start, modelfit_results=res_start)

    search_space_exhaustive = 'ABSORPTION(ZO);PERIPHERALS(1)'
    mfl = mfl_parse(search_space_exhaustive, mfl_class=True)
    mfl_funcs = mfl.convert_to_funcs()
    feats, funcs = mfl_funcs.keys(), mfl_funcs.values()

    if allometry:
        allometry = mfl_parse('ALLOMETRY(WT, 70)', mfl_class=True).allometry

    me_cand_exhaustive = create_candidate_exhaustive(
        'cand', feats, funcs, iiv_strategy, allometry, me_start
    )
    model_exhaustive = me_cand_exhaustive.model
    assert model_exhaustive.name == 'cand'
    assert len(model_exhaustive.statements.ode_system.find_peripheral_compartments()) == 1
    assert me_cand_exhaustive.modelfit_results is None
    assert me_cand_exhaustive.parent == model_start
    params_start = model_start.parameters.names
    params_cand = model_exhaustive.parameters.names
    assert set(params_cand) - set(params_start) == params_added
    assert model_exhaustive.parameters['POP_QP1'].upper == 999999
    assert model_exhaustive.parameters['POP_VP1'].upper == 999999

    model_base = set_zero_order_absorption(model_start)
    me_base = ModelEntry.create(model_base, modelfit_results=res_start)
    feats = ('PERIPHERALS', 1)
    funcs = mfl_funcs[feats]
    me_stepwise = create_candidate_stepwise('cand', feats, funcs, iiv_strategy, allometry, me_base)
    assert me_stepwise.model == model_exhaustive


@pytest.mark.parametrize(
    ('funcs', 'search_space', 'mfl_funcs'),
    [
        (
            [],
            'ABSORPTION([FO,ZO,SEQ-ZO-FO])',
            {('ABSORPTION', 'ZO'), ('ABSORPTION', 'SEQ-ZO-FO')},
        ),
        (
            [set_zero_order_absorption],
            'ABSORPTION([FO,ZO,SEQ-ZO-FO])',
            {('ABSORPTION', 'FO'), ('ABSORPTION', 'SEQ-ZO-FO')},
        ),
        (
            [set_zero_order_absorption, add_peripheral_compartment],
            'ABSORPTION([FO,ZO,SEQ-ZO-FO]);PERIPHERALS(1..2)',
            {('ABSORPTION', 'FO'), ('ABSORPTION', 'SEQ-ZO-FO'), ('PERIPHERALS', 2)},
        ),
    ],
)
def test_filter_mfl_statements(load_model_for_test, testdata, funcs, search_space, mfl_funcs):
    model_start = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox2.mod')
    res_start = read_modelfit_results(testdata / 'nonmem' / 'models' / 'mox2.mod')
    for func in funcs:
        model_start = func(model_start)
    me_start = ModelEntry.create(model_start, modelfit_results=res_start)
    search_space = mfl_parse(search_space, mfl_class=True)
    assert set(filter_mfl_statements(search_space, me_start).keys()) == mfl_funcs


def test_categorize_model_entries(load_model_for_test, testdata, model_entry_factory):
    model_start = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox2.mod')
    res_start = read_modelfit_results(testdata / 'nonmem' / 'models' / 'mox2.mod')
    me_start = ModelEntry.create(model_start, modelfit_results=res_start)
    model_base = set_zero_order_absorption(model_start).replace(name='base')
    me_base = ModelEntry.create(model_base)
    funcs = [set_seq_zo_fo_absorption, add_peripheral_compartment, add_lag_time]
    candidates = []
    for i, func in enumerate(funcs):
        model_cand = func(model_base).replace(name=f'modelsearch_run{i}')
        me_cand = ModelEntry.create(model_cand)
        candidates.append(me_cand)

    model_entries = [me_start, me_base] + candidates
    input_model_entry, base_model_entry, res_model_entries = categorize_model_entries(model_entries)
    assert input_model_entry.model == model_start
    assert base_model_entry.model == model_base
    assert len(res_model_entries) == len(funcs)

    with pytest.raises(ValueError):
        categorize_model_entries(candidates)


def test_create_workflow_with_model(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    results = read_modelfit_results(testdata / 'nonmem' / 'pheno.mod')
    assert isinstance(
        create_workflow(model, results, MINIMAL_VALID_MFL_STRING, 'exhaustive'), Workflow
    )


def test_validate_input_with_model(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    results = read_modelfit_results(testdata / 'nonmem' / 'pheno.mod')
    validate_input(model, results, MINIMAL_VALID_MFL_STRING, 'exhaustive')


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
            dict(search_space='ABSORPTION(FO);ALLOMETRY(X,70)'),
            ValueError,
            'Invalid `search_space`',
        ),
        (
            None,
            dict(algorithm=1),
            ValueError,
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
            ValueError,
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
            ValueError,
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
        (
            ('nonmem/ruvsearch/mox3.mod',),
            dict(strictness='rse'),
            ValueError,
            '`parameter_uncertainty_method` not set',
        ),
        (None, {'rank_type': 'ofv', 'E': 1.0}, ValueError, 'E can only be provided'),
        (None, {'rank_type': 'mbic'}, ValueError, 'Value `E` must be provided when using mbic'),
        (None, {'rank_type': 'mbic', 'E': 0.0}, ValueError, 'Value `E` must be more than 0'),
        (None, {'rank_type': 'mbic', 'E': '10'}, ValueError, 'Value `E` must be denoted with `%`'),
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
    if not model_path:
        model_path = ('nonmem/pheno.mod',)
    path = testdata.joinpath(*model_path)
    model = load_model_for_test(path)
    results = read_modelfit_results(path)

    harmless_arguments = dict(
        search_space=MINIMAL_VALID_MFL_STRING,
        algorithm='exhaustive',
    )

    kwargs = {'model': model, 'results': results, **harmless_arguments, **arguments}

    with pytest.raises(exception, match=match):
        validate_input(**kwargs)
