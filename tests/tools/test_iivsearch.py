from functools import partial

import pytest

from pharmpy.mfl import ModelFeatures
from pharmpy.modeling import (
    add_iiv,
    add_iov,
    add_lag_time,
    add_peripheral_compartment,
    add_pk_iiv,
    create_joint_distribution,
    find_clearance_parameters,
    fix_parameters,
    remove_iiv,
    set_direct_effect,
    set_iiv_on_ruv,
    set_name,
)
from pharmpy.modeling.mfl import get_model_features
from pharmpy.tools.external.results import parse_modelfit_results
from pharmpy.tools.iivsearch.algorithms import (
    _create_param_dict,
    _extract_clearance_parameter,
    _is_rv_block_structure,
    _rv_block_structures,
    create_block_structure_candidate_entry,
    create_candidate_linearized,
    create_description,
    create_description_mfl,
    create_eta_blocks,
    create_no_of_etas_candidate_entry,
    get_covariance_combinations,
    get_eta_names,
    get_iiv_combinations,
    td_exhaustive_block_structure,
    td_exhaustive_no_of_etas,
)
from pharmpy.tools.iivsearch.tool import add_iiv as iivsearch_add_iiv
from pharmpy.tools.iivsearch.tool import (
    categorize_model_entries,
    create_base_model_entry,
    create_param_mapping,
    create_param_mapping_mfl,
    create_workflow,
    get_mbic_search_space,
    get_ref_model,
    needs_base_model,
    prepare_algorithms,
    prepare_base_model,
    prepare_input_model,
    prepare_input_model_entry,
    prepare_rank_options,
    update_input_model_description,
    update_linearized_base_model,
    update_linearized_base_model_mfl,
    validate_input,
)
from pharmpy.tools.linearize.tool import create_derivative_model, create_linearized_model
from pharmpy.tools.run import read_modelfit_results
from pharmpy.workflows import ModelEntry, Workflow


def test_prepare_input_model(load_model_for_test, testdata):
    path = testdata / 'nonmem' / 'models' / 'mox2.mod'
    model = load_model_for_test(path)
    res = parse_modelfit_results(model, path)
    model_input, me_input = prepare_input_model(model, res)

    assert model_input.description == '[CL]+[VC]+[MAT]'
    assert me_input.modelfit_results is not None


@pytest.mark.parametrize(
    'iiv_strategy, linearize, no_of_params_added, description, has_mfr',
    [
        ('no_add', False, 0, '[CL]+[VC]+[MAT]', True),
        ('add_diagonal', False, 2, '[CL]+[VC]+[MAT]+[QP1]+[VP1]', False),
        ('fullblock', False, 12, '[CL,VC,MAT,QP1,VP1]', False),
        ('add_diagonal', True, 0, '[CL]+[VC]+[MAT]+[QP1]+[VP1]', True),
    ],
)
def test_prepare_base_model(
    load_model_for_test, testdata, iiv_strategy, linearize, no_of_params_added, description, has_mfr
):
    path = testdata / 'nonmem' / 'models' / 'mox2.mod'
    model_start = load_model_for_test(path)
    res_input = parse_modelfit_results(model_start, path)
    model_input = add_peripheral_compartment(model_start)
    me_input = ModelEntry.create(model_input, modelfit_results=res_input)
    model_base, me_base = prepare_base_model(me_input, iiv_strategy, linearize)

    has_updated_initial_ests = model_input.parameters['POP_CL'] != model_base.parameters['POP_CL']

    if iiv_strategy != 'no_add':
        assert has_updated_initial_ests
    else:
        assert not has_updated_initial_ests

    no_of_params_input = len(model_input.random_variables.parameter_names)
    param_names = model_base.random_variables.parameter_names
    no_of_params_base = len(model_base.parameters[param_names].nonfixed.names)
    assert no_of_params_base - no_of_params_input == no_of_params_added
    assert model_base.description == description
    assert bool(me_base.modelfit_results) is has_mfr


@pytest.mark.parametrize(
    'iiv_strategy, param_mapping, description',
    [
        ('no_add', {'ETA_1': 'CL', 'ETA_2': 'VC'}, ''),
        ('add_diagonal', {'ETA_1': 'CL', 'ETA_2': 'VC', 'ETA_MAT': 'MAT'}, '[CL]+[VC]+[MAT]'),
        ('fullblock', {'ETA_1': 'CL', 'ETA_2': 'VC', 'ETA_MAT': 'MAT'}, '[CL,VC,MAT]'),
    ],
)
def test_update_linearized_base_model(
    load_model_for_test, testdata, iiv_strategy, param_mapping, description
):
    path = testdata / 'nonmem' / 'models' / 'mox2.mod'
    model_start = load_model_for_test(path)
    res_start = parse_modelfit_results(model_start, path)
    model_start = remove_iiv(model_start, ['ETA_3'])

    if iiv_strategy != 'no_add':
        model_base = iivsearch_add_iiv(iiv_strategy, model_start, res_start, linearize=True)
    else:
        model_base = model_start
    me_base = ModelEntry.create(model_base, modelfit_results=res_start)
    me_updated = update_linearized_base_model(me_base, model_start, iiv_strategy, param_mapping)
    assert me_updated.model.description == description
    if iiv_strategy != 'no_add':
        assert not me_updated.modelfit_results
        assert len(model_base.parameters.fixed) > 0
    else:
        assert me_updated.modelfit_results
    assert len(me_updated.model.parameters.fixed) == 0


def test_update_linearized_base_model_mfl(load_model_for_test, testdata, model_entry_factory):
    model_start = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox2.mod')
    model_start = remove_iiv(model_start, ['ETA_3'])
    me_start = model_entry_factory([model_start])[0]

    param_mapping = {'ETA_1': 'CL', 'ETA_2': 'VC', 'ETA_MAT': 'MAT'}
    model_updated = update_linearized_base_model_mfl(False, param_mapping, me_start, me_start)
    assert len(model_updated.parameters) == len(model_start.parameters)
    assert model_updated.description == '[CL]+[VC]'

    model_base = add_iiv(model_start, ['MAT'], 'exp')
    model_base = fix_parameters(model_base, parameter_names=['IIV_MAT'])
    me_base = model_entry_factory([model_base])[0]
    model_updated = update_linearized_base_model_mfl(False, param_mapping, me_start, me_base)
    assert len(model_updated.parameters) > len(model_start.parameters)
    assert len(model_base.parameters.fixed) > len(model_updated.parameters.fixed)
    assert len(model_updated.parameters) == len(model_base.parameters)
    assert model_updated.description == '[CL]+[VC]+[MAT]'

    model_fullblock = update_linearized_base_model_mfl(True, param_mapping, me_start, me_base)
    assert len(model_fullblock.parameters) > len(model_updated.parameters)
    assert len(model_fullblock.random_variables.iiv) == 1
    assert model_fullblock.description == '[CL,VC,MAT]'


def test_prepare_input_model_entry(load_model_for_test, testdata):
    model_start = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox2.mod')
    res_start = read_modelfit_results(testdata / 'nonmem' / 'models' / 'mox2.mod')

    me = prepare_input_model_entry(model_start, res_start)
    assert me.model.name == 'input'
    assert me.model.description == '[CL]+[MAT]+[VC]'
    assert me.modelfit_results is not None
    assert me.parent is None


@pytest.mark.parametrize(
    'algorithm, correlation_algorithm, list_of_algorithms',
    [
        ('top_down_exhaustive', 'skip', ['td_exhaustive_no_of_etas']),
        ('bottom_up_stepwise', 'skip', ['bu_stepwise_no_of_etas']),
        (
            'top_down_exhaustive',
            'top_down_exhaustive',
            ['td_exhaustive_no_of_etas', 'td_exhaustive_block_structure'],
        ),
        ('skip', 'top_down_exhaustive', ['td_exhaustive_block_structure']),
        (
            'top_down_exhaustive',
            None,
            ['td_exhaustive_no_of_etas', 'td_exhaustive_block_structure'],
        ),
        ('bottom_up_stepwise', None, ['bu_stepwise_no_of_etas', 'td_exhaustive_block_structure']),
    ],
)
def test_prepare_algorithms(algorithm, correlation_algorithm, list_of_algorithms):
    assert prepare_algorithms(algorithm, correlation_algorithm) == list_of_algorithms


def test_create_param_mapping(load_model_for_test, testdata):
    path = testdata / 'nonmem' / 'models' / 'mox2.mod'
    model = load_model_for_test(path)
    res = parse_modelfit_results(model, path)
    me_input = ModelEntry.create(model, modelfit_results=res)

    param_mapping = create_param_mapping(me_input, linearize=False)
    assert param_mapping is None

    param_mapping = create_param_mapping(me_input, linearize=True)
    assert param_mapping == {'ETA_1': 'CL', 'ETA_2': 'VC', 'ETA_3': 'MAT'}

    param_mapping = create_param_mapping_mfl(me_input)
    assert param_mapping == {'ETA_1': 'CL', 'ETA_2': 'VC', 'ETA_3': 'MAT'}


@pytest.mark.parametrize(
    'iiv_strategy, linearize, no_of_added_params',
    [
        ('add_diagonal', False, 2),
        ('fullblock', False, 12),
        ('add_diagonal', True, 0),
        ('pd_add_diagonal', False, 2),
        ('pd_fullblock', False, 3),
    ],
)
def test_add_iiv(load_model_for_test, testdata, iiv_strategy, linearize, no_of_added_params):
    if not iiv_strategy.startswith('pd'):
        path = testdata / 'nonmem' / 'models' / 'mox2.mod'
        model_start = load_model_for_test(path)
        res_input = parse_modelfit_results(model_start, path)
        model_input = add_peripheral_compartment(model_start)
    else:
        path = testdata / 'nonmem' / 'pheno_pd.mod'
        model_start = load_model_for_test(path)
        res_input = parse_modelfit_results(model_start, path)
        model_start = fix_parameters(model_start, model_start.parameters.names)
        model_input = set_direct_effect(model_start, expr='linear')

    model_iiv = iivsearch_add_iiv(iiv_strategy, model_input, res_input, linearize=linearize)
    assert (
        len(model_iiv.parameters.nonfixed) - len(model_input.parameters.nonfixed)
        == no_of_added_params
    )


@pytest.mark.parametrize(
    'algorithm, base_model_name',
    [
        ('td_exhaustive_no_of_etas', 'pheno'),
        ('bu_stepwise_no_of_etas', 'model_2'),
    ],
)
def test_categorize_model_entries(
    load_model_for_test, testdata, model_entry_factory, algorithm, base_model_name
):
    path = testdata / 'nonmem' / 'pheno.mod'
    model_start = load_model_for_test(path)
    res_start = parse_modelfit_results(model_start, path)
    model_start = add_peripheral_compartment(model_start)
    model_start = add_pk_iiv(model_start)
    me_start = ModelEntry.create(model_start, modelfit_results=res_start)

    model_1 = remove_iiv(model_start, 'QP1')
    model_1 = model_1.replace(name='model_1')
    model_2 = remove_iiv(model_1, 'VP1')
    model_2 = model_2.replace(name='model_2')

    candidate_entries = model_entry_factory([model_1, model_2])
    model_entries = [me_start] + candidate_entries

    base_model_entry, res_model_entries = categorize_model_entries(model_entries, algorithm)

    assert len(res_model_entries) == 2
    assert base_model_entry.model.name == base_model_name


def test_update_input_model_description(load_model_for_test, testdata):
    path = testdata / 'nonmem' / 'pheno.mod'
    model_start = load_model_for_test(path)
    res_start = parse_modelfit_results(model_start, path)
    model_start = add_peripheral_compartment(model_start)
    model_start = add_pk_iiv(model_start)
    me_start = ModelEntry.create(model_start, modelfit_results=res_start)

    me = update_input_model_description(me_start)
    assert me.model.description == '[CL]+[V1]+[QP1]+[VP1]'


@pytest.mark.parametrize(
    'to_remove, no_of_etas, description',
    [
        (('ETA_1',), 2, '[VC]+[MAT]'),
        (('ETA_1', 'ETA_2'), 1, '[MAT]'),
    ],
)
def test_create_no_of_etas_candidate_entry(
    load_model_for_test, testdata, to_remove, no_of_etas, description
):
    path = testdata / 'nonmem' / 'models' / 'mox2.mod'
    model_start = load_model_for_test(path)
    res_start = parse_modelfit_results(model_start, path)
    me_start = ModelEntry.create(model_start, modelfit_results=res_start)

    me_candidate = create_no_of_etas_candidate_entry(
        'cand', to_remove, tuple(), tuple(), False, None, me_start
    )
    assert len(me_candidate.model.random_variables.iiv.names) == no_of_etas
    assert me_candidate.model.description == description
    assert me_candidate.model.parameters['POP_CL'].init == res_start.parameter_estimates['POP_CL']
    assert me_candidate.modelfit_results is None
    assert me_candidate.parent == model_start

    etas = ['ETA_1', 'ETA_2', 'ETA_3']
    param_names = ['CL', 'VC', 'MAT']
    me_candidate_param_mapping = create_no_of_etas_candidate_entry(
        'cand', to_remove, etas, param_names, False, None, me_start
    )
    assert me_candidate.model == me_candidate_param_mapping.model

    me_candidate_base = create_no_of_etas_candidate_entry(
        'cand', to_remove, tuple(), tuple(), True, me_candidate, me_start
    )
    assert me_candidate.model.statements == me_candidate_base.model.statements
    assert me_candidate_base.parent == model_start


@pytest.mark.parametrize(
    'block_structure, no_of_dists, description',
    [
        ((('ETA_1', 'ETA_2', 'ETA_3'),), 1, '[CL,VC,MAT]'),
        (
            (
                ('ETA_1',),
                ('ETA_2', 'ETA_3'),
            ),
            2,
            '[CL]+[VC,MAT]',
        ),
        (
            (
                ('ETA_1',),
                ('ETA_2',),
                ('ETA_3',),
            ),
            3,
            '[CL]+[VC]+[MAT]',
        ),
    ],
)
def test_create_block_structure_candidate_entry(
    load_model_for_test, testdata, block_structure, no_of_dists, description
):
    path = testdata / 'nonmem' / 'models' / 'mox2.mod'
    model_start = load_model_for_test(path)
    res_start = parse_modelfit_results(model_start, path)
    me_start = ModelEntry.create(model_start, modelfit_results=res_start)

    me_candidate = create_block_structure_candidate_entry(
        'cand', block_structure, tuple(), tuple(), me_start
    )
    assert len(me_candidate.model.random_variables.iiv) == no_of_dists
    assert me_candidate.model.description == description
    assert me_candidate.model.parameters['POP_CL'].init == res_start.parameter_estimates['POP_CL']
    assert me_candidate.modelfit_results is None
    assert me_candidate.parent == model_start

    etas = ['ETA_1', 'ETA_2', 'ETA_3']
    param_names = ['CL', 'VC', 'MAT']
    me_candidate_param_mapping = create_block_structure_candidate_entry(
        'cand', block_structure, etas, param_names, me_start
    )
    assert me_candidate.model == me_candidate_param_mapping.model


@pytest.mark.parametrize(
    'iivs_to_remove, param_dict, description',
    [
        (('ETA_1',), None, '[VC]+[MAT]'),
        (
            (
                'ETA_1',
                'ETA_2',
                'ETA_3',
            ),
            None,
            '[]',
        ),
        (
            (
                'ETA_1',
                'ETA_2',
            ),
            {'ETA_3': 'MAT'},
            '[MAT]',
        ),
    ],
)
def test_create_description(load_model_for_test, testdata, iivs_to_remove, param_dict, description):
    model = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox2.mod')
    model = remove_iiv(model, iivs_to_remove)
    assert create_description(model, False, param_dict) == description


def test_create_description_iov(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox2.mod')
    model = add_iov(model, 'VISI')
    assert create_description(model, True, None) == '[CL]+[VC]+[MAT]'


@pytest.mark.parametrize(
    'mfl, type, expected',
    [
        ('IIV([CL,MAT,VC],EXP)', 'iiv', '[CL]+[MAT]+[VC]'),
        ('IIV(CL,EXP);IIV([MAT,VC],ADD)', 'iiv', '[CL]+[MAT]+[VC] (ADD:MAT,VC)'),
        ('IIV(CL,EXP);IIV(MAT,ADD);IIV(VC,LOG)', 'iiv', '[CL]+[MAT]+[VC] (ADD:MAT;LOG:VC)'),
        ('IIV([CL,MAT,VC],EXP);COVARIANCE(IIV,[CL,VC])', 'iiv', '[CL,VC]+[MAT]'),
        ('IIV(CL,EXP);IIV(MAT,ADD);IIV(VC,LOG)', 'covariance', '[CL]+[MAT]+[VC]'),
        ('IIV([CL,MAT,VC],EXP);COVARIANCE(IIV,[CL,MAT,VC])', 'iiv', '[CL,MAT,VC]'),
    ],
)
def test_create_description_mfl(mfl, type, expected):
    mfl = ModelFeatures.create(mfl)
    assert create_description_mfl(mfl, type) == expected


@pytest.mark.parametrize(
    'mfl, base_features, expected',
    [
        ('IIV(CL,EXP);IIV?([MAT,VC],EXP)', 'IIV([CL,MAT,VC],EXP)', 3),
        ('IIV?([CL,MAT,VC],EXP)', 'IIV([CL,MAT,VC],EXP)', 7),
        ('IIV?([CL,MAT,VC],EXP)', [], 7),
        ('IIV(CL,EXP);IIV?([MAT,VC],[EXP,ADD])', 'IIV([CL,MAT,VC],EXP)', 8),
    ],
)
def test_get_iiv_combinations(mfl, base_features, expected):
    mfl = ModelFeatures.create(mfl)
    mfl_base = ModelFeatures.create(base_features)
    combinations = get_iiv_combinations(mfl, mfl_base)
    assert len(combinations) == expected
    assert mfl_base not in combinations


@pytest.mark.parametrize(
    'mfl, base_features, expected',
    [
        ('COVARIANCE?(IIV,[CL,MAT,VC])', 'COVARIANCE(IIV,[CL,MAT,VC])', 4),
        ('COVARIANCE?(IIV,[CL,MAT,VC,MDT])', 'COVARIANCE(IIV,[CL,MAT,VC,MDT])', 11),
        ('COVARIANCE?(IIV,[CL,MAT,VC,MDT])', 'COVARIANCE(IIV,[CL,MAT,VC])', 11),
        ('COVARIANCE?(IIV,[CL,MAT,VC])', [], 4),
    ],
)
def test_get_covariance_combinations(mfl, base_features, expected):
    mfl = ModelFeatures.create(mfl)
    mfl_base = ModelFeatures.create(base_features)
    combinations = get_covariance_combinations(mfl, mfl_base)
    mf_empty = ModelFeatures(tuple())
    assert mf_empty in combinations if base_features else mf_empty not in combinations
    assert len(combinations) == expected
    assert mfl_base not in combinations


@pytest.mark.parametrize(
    'func, mfl, type, description, no_of_params',
    [
        (None, 'IIV([CL,MAT],exp)', 'iiv', '[CL]+[MAT]', 2),
        (None, 'IIV([CL],exp)', 'iiv', '[CL]', 1),
        (None, '', 'iiv', '', 1),
        (add_lag_time, 'IIV([CL,VC,MDT],exp)', 'iiv', '[CL]+[MDT]+[VC]', 3),
        (None, 'COVARIANCE(IIV,[CL,VC,MAT])', 'covariance', '[CL,MAT,VC]', 6),
        (None, 'COVARIANCE(IIV,[CL,VC])', 'covariance', '[CL,VC]+[MAT]', 4),
        (add_lag_time, 'COVARIANCE(IIV,[CL,VC,MAT])', 'covariance', '[CL,MAT,VC]+[MDT]', 7),
        (
            add_lag_time,
            'COVARIANCE(IIV,[CL,VC]);COVARIANCE(IIV,[MAT,MDT])',
            'covariance',
            '[CL,VC]+[MAT,MDT]',
            6,
        ),
    ],
)
def test_create_candidate_linearized(
    load_model_for_test, model_entry_factory, testdata, func, mfl, type, description, no_of_params
):
    input_model = load_model_for_test(testdata / "nonmem" / "models" / "mox2.mod")
    if func:
        input_model = func(input_model)
        input_model = add_pk_iiv(input_model)
    input_model_entry = model_entry_factory([input_model])[0]

    derivative_model_entry = create_derivative_model(input_model_entry)
    derivative_model_entry = model_entry_factory([derivative_model_entry.model])[0]

    linbase_model_entry = create_linearized_model(
        "linbase", "", input_model, derivative_model_entry
    )
    linbase_model_entry = model_entry_factory([linbase_model_entry.model])[0]

    param_mapping = {'ETA_1': 'CL', 'ETA_2': 'VC', 'ETA_3': 'MAT', 'ETA_MDT': 'MDT'}

    mfl = ModelFeatures.create(mfl)
    candidate_model_entry = create_candidate_linearized(
        'cand1', mfl, type, param_mapping, linbase_model_entry
    )
    assert candidate_model_entry.parent == linbase_model_entry.model
    assert candidate_model_entry.modelfit_results is None

    candidate_model = candidate_model_entry.model
    assert candidate_model.description == description
    assert len(candidate_model.random_variables.iiv.parameter_names) == no_of_params
    if mfl:
        assert (
            candidate_model.parameters.inits['IIV_CL']
            != linbase_model_entry.model.parameters.inits['IIV_CL']
        )


@pytest.mark.parametrize(
    'funcs, mfl, as_fullblock, algorithm, expected',
    [
        ([], 'IIV([CL,MAT,VC],EXP)', False, 'td', False),
        ([], 'IIV?([CL,MAT,VC],EXP)', False, 'td', False),
        ([], 'IIV?([CL,MAT,VC],EXP);COVARIANCE(IIV,[CL,VC])', False, 'td', True),
        ([add_peripheral_compartment], 'IIV?([CL,MAT,VC],EXP)', False, 'td', False),
        ([add_peripheral_compartment], 'IIV?([CL,MAT,VC,QP1],EXP)', False, 'td', False),
        (
            [
                add_peripheral_compartment,
                partial(add_iiv, list_of_parameters=['QP1'], expression='exp'),
            ],
            'IIV?([CL,MAT,VC,QP1],EXP)',
            False,
            'td',
            False,
        ),
        ([], 'IIV([CL,MAT,VC],EXP);COVARIANCE?(IIV,[CL,VC])', False, 'td', False),
        ([], 'IIV([CL,MAT,VC],EXP)', True, 'td', True),
        ([create_joint_distribution], 'IIV([CL,MAT,VC],EXP)', True, 'td', False),
        ([], 'IIV(CL,EXP);IIV?([MAT,VC],[EXP,ADD])', False, 'td', False),
        ([], 'IIV([CL,MAT,VC],EXP)', False, 'bu', True),
    ],
)
def test_needs_base_model(
    load_model_for_test, testdata, funcs, mfl, as_fullblock, algorithm, expected
):
    model = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox2.mod')
    for func in funcs:
        model = func(model)

    mfl = ModelFeatures.create(mfl)
    assert needs_base_model(model, mfl, as_fullblock, algorithm) == expected


@pytest.mark.parametrize(
    'type, mfl, as_fullblock, iiv_expected, cov_expected',
    [
        ('td', 'IIV?([CL,VC],EXP)', False, 'IIV([CL,VC],EXP)', ''),
        ('td', 'IIV?([CL,MAT,VC],EXP)', False, 'IIV([CL,MAT,VC],EXP)', ''),
        (
            'td',
            'IIV?([CL,MAT,VC],EXP)',
            True,
            'IIV([CL,MAT,VC],EXP)',
            'COVARIANCE(IIV,[CL,MAT,VC])',
        ),
        (
            'td',
            'IIV(CL,EXP);IIV?([CL,MAT,VC],EXP);COVARIANCE?(IIV,[CL,MAT,VC])',
            False,
            'IIV([CL,MAT,VC],EXP)',
            '',
        ),
        ('td', 'IIV(CL,EXP);IIV?([MAT,VC],ADD)', False, 'IIV(CL,EXP);IIV([MAT,VC],ADD)', ''),
        ('td', 'IIV(CL,[EXP,ADD]);IIV?([MAT,VC],[EXP,ADD])', False, 'IIV([CL,MAT,VC],EXP)', ''),
        (
            'td',
            'IIV(CL,[EXP,ADD]);IIV?([MAT,VC],[EXP,ADD]);COVARIANCE?(IIV,[CL,MAT,VC])',
            True,
            'IIV([CL,MAT,VC],EXP)',
            'COVARIANCE(IIV,[CL,MAT,VC])',
        ),
        ('bu', 'IIV?([CL,MAT,VC],EXP)', False, '', ''),
        ('bu', 'IIV(CL,EXP);IIV?([MAT,VC],EXP)', False, 'IIV(CL,EXP)', ''),
        ('bu', 'IIV([CL,MAT],EXP);IIV?(VC,EXP)', False, 'IIV([CL,MAT],EXP)', ''),
        ('bu', 'IIV(CL,EXP);IIV?([MAT,VC],EXP)', True, 'IIV(CL,EXP)', ''),
        (
            'bu',
            'IIV([CL,MAT],EXP);IIV?(VC,EXP)',
            True,
            'IIV([CL,MAT],EXP)',
            'COVARIANCE(IIV,[CL,MAT])',
        ),
        ('linearize', 'IIV?([CL,VC],EXP)', False, 'IIV([CL,VC],EXP)', ''),
        ('linearize', 'IIV?([CL,MAT,VC],EXP)', False, 'IIV([CL,MAT,VC],EXP)', ''),
    ],
)
def test_create_base_model(
    load_model_for_test,
    testdata,
    model_entry_factory,
    type,
    mfl,
    as_fullblock,
    iiv_expected,
    cov_expected,
):
    model = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox2.mod')
    model = remove_iiv(model, to_remove=['ETA_3'])
    model_entry = model_entry_factory([model])[0]
    mfl = ModelFeatures.create(mfl)
    base_model_entry = create_base_model_entry(type, mfl, as_fullblock, model_entry)
    base_model = base_model_entry.model
    assert repr(get_model_features(base_model, type='iiv')) == iiv_expected
    assert repr(get_model_features(base_model, type='covariance')) == cov_expected
    assert base_model_entry.modelfit_results is None
    assert base_model_entry.parent == model

    if type == 'linearize':
        params_new = base_model.parameters - model.parameters
        assert all(p.init == 0.000001 for p in params_new)


def test_create_base_model_raises(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox2.mod')
    res = read_modelfit_results(testdata / 'nonmem' / 'models' / 'mox2.mod')
    model_entry = ModelEntry.create(model=model, modelfit_results=res)
    mfl = ModelFeatures.create('')
    with pytest.raises(ValueError):
        create_base_model_entry('x', mfl, True, model_entry)


@pytest.mark.parametrize(
    'kwargs, expected',
    [
        (
            {
                'rank_type': 'ofv',
                'cutoff': 3.84,
                'strictness': 'minimization_successful',
                'parameter_uncertainty_method': None,
                'E_p': None,
                'E_q': None,
            },
            {
                'rank_type': 'ofv',
                'cutoff': 3.84,
                'strictness': 'minimization_successful',
                'parameter_uncertainty_method': None,
                'E': None,
            },
        ),
        (
            {
                'rank_type': 'bic',
                'cutoff': None,
                'strictness': 'minimization_successful',
                'parameter_uncertainty_method': None,
                'E_p': None,
                'E_q': None,
            },
            {
                'rank_type': 'bic_iiv',
                'cutoff': None,
                'strictness': 'minimization_successful',
                'parameter_uncertainty_method': None,
                'E': None,
            },
        ),
        (
            {
                'rank_type': 'mbic',
                'cutoff': None,
                'strictness': 'minimization_successful',
                'parameter_uncertainty_method': None,
                'E_p': 0.5,
                'E_q': 0.5,
            },
            {
                'rank_type': 'mbic_iiv',
                'cutoff': None,
                'strictness': 'minimization_successful',
                'parameter_uncertainty_method': None,
                'E': (0.5, 0.5),
            },
        ),
    ],
)
def test_prepare_rank_options(kwargs, expected):
    rank_options = prepare_rank_options(**kwargs)
    for key, value in expected.items():
        assert getattr(rank_options, key) == value


@pytest.mark.parametrize(
    'list_of_parameters, expected_values',
    [([], 4), (['IIV_CL'], 1), (["IIV_CL", "IIV_VC"], 0)],
)
def test_td_exhaustive_block_structure_ignore_fixed_params(
    load_model_for_test, testdata, list_of_parameters, expected_values
):
    model = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox2.mod')
    model = fix_parameters(model, list_of_parameters)
    wf = td_exhaustive_block_structure(model)
    fit_tasks = [task.name for task in wf.tasks if task.name.startswith('run')]
    assert len(fit_tasks) == expected_values


@pytest.mark.parametrize(
    'list_of_parameters, expected_values',
    [([], 3), (['CL'], 1), (["CL", "V"], 0)],
)
def test_brute_force_no_of_etas_keep(
    load_model_for_test, testdata, list_of_parameters, expected_values
):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno_real.mod')
    wf = td_exhaustive_no_of_etas(model, keep=list_of_parameters)
    fit_tasks = [task.name for task in wf.tasks if task.name.startswith('run')]
    assert len(fit_tasks) == expected_values


def test_brute_force_no_of_etas_fixed(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno_real.mod')
    model = fix_parameters(model, 'IVCL')
    wf = td_exhaustive_no_of_etas(model)
    fit_tasks = [task.name for task in wf.tasks if task.name.startswith('run')]
    assert len(fit_tasks) == 1


@pytest.mark.parametrize(
    'list_of_parameters, no_of_models',
    [([], 7), (['QP1'], 15)],
)
def test_brute_force_no_of_etas(load_model_for_test, testdata, list_of_parameters, no_of_models):
    model = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox2.mod')
    model = add_peripheral_compartment(model)
    model = add_iiv(model, list_of_parameters, 'add')
    wf = td_exhaustive_no_of_etas(model)
    fit_tasks = [task.name for task in wf.tasks if task.name.startswith('run')]

    assert len(fit_tasks) == no_of_models


def test_td_no_of_etas_linearized(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno_real_linbase.mod')
    param_mapping = {"ETA_1": "CL", "ETA_2": "V"}
    wf = td_exhaustive_no_of_etas(model, param_mapping=param_mapping)
    fit_tasks = [task.name for task in wf.tasks if task.name.startswith('run')]

    assert len(fit_tasks) == 3


def test_extract_base_parameter(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox2.mod')
    clearance_parameter = str(find_clearance_parameters(model)[0])
    iiv_names = model.random_variables.iiv.names
    param_mapping = None

    base_parameter = _extract_clearance_parameter(
        model, param_mapping, clearance_parameter, iiv_names
    )
    assert base_parameter == "ETA_1"

    param_mapping = {"ETA_3": clearance_parameter}
    base_parameter = _extract_clearance_parameter(
        model, param_mapping, clearance_parameter, iiv_names
    )
    assert base_parameter == "ETA_3"

    param_mapping = None
    clearance_parameter = ""
    base_parameter = _extract_clearance_parameter(
        model, param_mapping, clearance_parameter, iiv_names
    )
    assert base_parameter == ""


@pytest.mark.parametrize(
    'list_of_parameters, block_structure, no_of_models',
    [([], [], 4), (['QP1'], [], 14), ([], ['ETA_1', 'ETA_2'], 4)],
)
def test_brute_force_block_structure(
    load_model_for_test, testdata, list_of_parameters, block_structure, no_of_models
):
    path = testdata / 'nonmem' / 'models' / 'mox2.mod'
    model = load_model_for_test(path)
    res = parse_modelfit_results(model, path)
    model = add_peripheral_compartment(model)
    model = add_iiv(model, list_of_parameters, 'add')
    if block_structure:
        model = create_joint_distribution(
            model, block_structure, individual_estimates=res.individual_estimates
        )

    wf = td_exhaustive_block_structure(model)
    fit_tasks = [task.name for task in wf.tasks if task.name.startswith('run')]

    assert len(fit_tasks) == no_of_models


def test_rv_block_structures_4_etas(load_model_for_test, pheno_path):
    model = load_model_for_test(pheno_path)
    model = add_iiv(model, ['TAD', 'S1'], 'exp')

    block_structures = list(_rv_block_structures(model.random_variables.iiv.names))

    assert len(block_structures) == 15

    block_structures_integer_partitions = [
        tuple(map(len, block_structure)) for block_structure in block_structures
    ]
    assert block_structures_integer_partitions.count((4,)) == 1
    assert block_structures_integer_partitions.count((1, 3)) == 4
    assert block_structures_integer_partitions.count((2, 2)) == 3
    assert block_structures_integer_partitions.count((1, 1, 2)) == 6
    assert block_structures_integer_partitions.count((1, 1, 1, 1)) == 1


def test_rv_block_structures_5_etas(load_model_for_test, pheno_path):
    model = load_model_for_test(pheno_path)
    model = add_iiv(model, ['TVCL', 'TAD', 'S1'], 'exp')

    block_structures = list(_rv_block_structures(model.random_variables.iiv.names))
    assert len(block_structures) == 52

    block_structures_integer_partitions = [
        tuple(map(len, block_structure)) for block_structure in block_structures
    ]
    assert block_structures_integer_partitions.count((5,)) == 1
    assert block_structures_integer_partitions.count((1, 4)) == 5
    assert block_structures_integer_partitions.count((2, 3)) == 10
    assert block_structures_integer_partitions.count((1, 1, 3)) == 10
    assert block_structures_integer_partitions.count((1, 2, 2)) == 15
    assert block_structures_integer_partitions.count((1, 1, 1, 2)) == 10
    assert block_structures_integer_partitions.count((1, 1, 1, 1, 1)) == 1


def test_is_rv_block_structure(pheno, pheno_path):
    res = parse_modelfit_results(pheno, pheno_path)
    model = add_iiv(pheno, ['TAD', 'S1'], 'exp')

    etas_block_structure = (('ETA_1', 'ETA_2'), ('ETA_TAD',), ('ETA_S1',))
    model = create_joint_distribution(
        model,
        list(etas_block_structure[0]),
        individual_estimates=res.individual_estimates,
    )
    etas = model.random_variables.iiv
    assert _is_rv_block_structure(etas, etas_block_structure, [])

    etas_block_structure = (('ETA_1',), ('ETA_2',), ('ETA_TAD', 'ETA_S1'))
    assert not _is_rv_block_structure(etas, etas_block_structure, [])

    etas_block_structure = (('ETA_1',), ('ETA_2', 'ETA_TAD'), ('ETA_S1',))
    assert not _is_rv_block_structure(etas, etas_block_structure, [])

    model = create_joint_distribution(model, individual_estimates=res.individual_estimates)
    etas_block_structure = (('ETA_1', 'ETA_2', 'ETA_TAD', 'ETA_S1'),)
    etas = model.random_variables.iiv
    assert _is_rv_block_structure(etas, etas_block_structure, [])


def test_create_joint_dist(load_model_for_test, testdata):
    path = testdata / 'nonmem' / 'models' / 'mox2.mod'
    model = load_model_for_test(path)
    res = parse_modelfit_results(model, path)

    model = add_peripheral_compartment(model)
    model = add_pk_iiv(model)
    etas_block_structure = (('ETA_1', 'ETA_2'), ('ETA_QP1',), ('ETA_VP1',))
    model = create_eta_blocks(etas_block_structure, model, res)
    assert len(model.random_variables.iiv) == 4

    model = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox2.mod')
    model = add_peripheral_compartment(model)
    model = add_pk_iiv(model)
    model = create_joint_distribution(
        model,
        ['ETA_1', 'ETA_2'],
        individual_estimates=res.individual_estimates,
    )
    etas_block_structure = (('ETA_1',), ('ETA_2',), ('ETA_3', 'ETA_VP1', 'ETA_QP1'))
    model = create_eta_blocks(etas_block_structure, model, res)
    assert len(model.random_variables.iiv) == 3


def test_get_param_names(create_model_for_test, load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox2.mod')

    param_dict = _create_param_dict(model, model.random_variables.iiv)
    param_dict_ref = {'ETA_1': 'CL', 'ETA_2': 'VC', 'ETA_3': 'MAT'}

    assert param_dict == param_dict_ref

    model_code = model.code.replace(
        'CL = THETA(1) * EXP(ETA(1))', 'ETA_1 = ETA(1)\nCL = THETA(1) * EXP(ETA_1)'
    )
    model = create_model_for_test(model_code)

    param_dict = _create_param_dict(model, model.random_variables.iiv)

    assert param_dict == param_dict_ref


def test_get_ref_model(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox2.mod')

    max_iiv_diag = set_name(model, 'max_iiv_diag')
    cand = set_name(model, 'cand')
    cand = remove_iiv(cand, ['VC'])
    min_iiv = set_name(model, 'min_iiv')
    min_iiv = remove_iiv(min_iiv, ['VC', 'MAT'])
    models = [max_iiv_diag, cand, min_iiv]
    assert get_ref_model(models, 'td_exhaustive_no_of_etas').name == 'max_iiv_diag'
    assert get_ref_model(models, 'bu_stepwise_no_of_etas').name == 'min_iiv'

    max_iiv_block = set_name(model, 'max_iiv_block')
    max_iiv_block = create_joint_distribution(max_iiv_block, None)
    models = [max_iiv_block, max_iiv_diag]
    assert get_ref_model(models, 'td_exhaustive_block_structure').name == 'max_iiv_block'

    with pytest.raises(ValueError):
        get_ref_model([models], 'x')


@pytest.mark.parametrize(
    'funcs, kwargs, search_space',
    [
        ([], {'keep': [], 'E_p': 1, 'E_q': None}, 'IIV?([CL,MAT,VC],EXP)'),
        (
            [],
            {'keep': [], 'E_p': 1, 'E_q': 1},
            'IIV?([CL,MAT,VC],EXP);COVARIANCE?(IIV,[CL,MAT,VC])',
        ),
        (
            [],
            {'keep': [], 'E_p': None, 'E_q': 1},
            'IIV([CL,MAT,VC],EXP);COVARIANCE?(IIV,[CL,MAT,VC])',
        ),
        ([], {'keep': ['CL'], 'E_p': 1, 'E_q': None}, 'IIV(CL,EXP);IIV?([MAT,VC],EXP)'),
        (
            [],
            {'keep': ['CL'], 'E_p': 1, 'E_q': 1},
            'IIV(CL,EXP);IIV?([MAT,VC],EXP);COVARIANCE?(IIV,[CL,MAT,VC])',
        ),
        (
            [add_peripheral_compartment, add_pk_iiv, create_joint_distribution],
            {'keep': [], 'E_p': 1, 'E_q': 1},
            'IIV?([CL,MAT,QP1,VC,VP1],EXP);COVARIANCE?(IIV,[CL,MAT,QP1,VC,VP1])',
        ),
        (
            [add_peripheral_compartment, add_pk_iiv, create_joint_distribution],
            {'keep': ['CL'], 'E_p': 1, 'E_q': 1},
            'IIV(CL,EXP);IIV?([MAT,QP1,VC,VP1],EXP);COVARIANCE?(IIV,[CL,MAT,QP1,VC,VP1])',
        ),
    ],
)
def test_get_mbic_search_space(load_model_for_test, testdata, funcs, kwargs, search_space):
    model = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox2.mod')
    for func in funcs:
        model = func(model)
    assert get_mbic_search_space(model, **kwargs) == search_space


@pytest.mark.parametrize(
    'funcs, keep, param_mapping, iiv_names',
    [
        (tuple(), None, None, {'ETA_1', 'ETA_2'}),
        (tuple(), ['CL'], None, {'ETA_2'}),
        ((partial(fix_parameters, parameter_names=['IVCL']),), None, None, {'ETA_2'}),
        ((set_iiv_on_ruv,), None, None, {'ETA_1', 'ETA_2'}),
        (tuple(), ['X'], {'ETA_1': 'X', 'ETA_2': 'VC'}, {'ETA_2'}),
    ],
)
def test_get_eta_names(load_model_for_test, testdata, funcs, keep, param_mapping, iiv_names):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    for func in funcs:
        model = func(model=model)
    assert set(get_eta_names(model, keep, param_mapping)) == iiv_names


def test_create_workflow_with_model(load_model_for_test, testdata):
    path = testdata / 'nonmem' / 'pheno.mod'
    model = load_model_for_test(path)
    results = parse_modelfit_results(model, path)
    assert isinstance(
        create_workflow(model=model, results=results, algorithm='top_down_exhaustive'), Workflow
    )


def test_validate_input_with_model(load_model_for_test, testdata):
    path = testdata / 'nonmem' / 'pheno.mod'
    model = load_model_for_test(path)
    results = parse_modelfit_results(model, path)
    validate_input(model=model, results=results, algorithm='top_down_exhaustive')


@pytest.mark.parametrize(
    ('model_path', 'arguments', 'exception', 'match'),
    [
        (None, dict(algorithm=1), ValueError, 'Invalid `algorithm`'),
        (None, dict(algorithm='brute_force_no_of_eta'), ValueError, 'Invalid `algorithm`'),
        (None, dict(rank_type=1), ValueError, 'Invalid `rank_type`'),
        (None, dict(rank_type='bi'), ValueError, 'Invalid `rank_type`'),
        (None, dict(iiv_strategy=['no_add']), ValueError, 'Invalid `iiv_strategy`'),
        (None, dict(iiv_strategy='diagonal'), ValueError, 'Invalid `iiv_strategy`'),
        (None, dict(cutoff='1'), TypeError, 'Invalid `cutoff`'),
        (
            None,
            dict(model=1),
            TypeError,
            'Invalid `model`',
        ),
        (
            ('nonmem/pheno.mod',),
            dict(strictness='rse'),
            ValueError,
            '`parameter_uncertainty_method` not set',
        ),
        (
            None,
            dict(algorithm='skip', correlation_algorithm='skip'),
            ValueError,
            'Both algorithm and correlation_algorithm',
        ),
        (
            None,
            dict(algorithm='skip', correlation_algorithm=None),
            ValueError,
            'correlation_algorithm need to be specified',
        ),
        (None, {'rank_type': 'ofv', 'E_p': 1.0}, ValueError, 'E_p and E_q can only be provided'),
        (None, {'rank_type': 'ofv', 'E_q': 1.0}, ValueError, 'E_p and E_q can only be provided'),
        (
            None,
            {'rank_type': 'mbic', 'algorithm': 'top_down_exhaustive'},
            ValueError,
            'Value `E_p` must be provided for `algorithm`',
        ),
        (
            None,
            {
                'rank_type': 'mbic',
                'algorithm': 'skip',
                'correlation_algorithm': 'top_down_exhaustive',
            },
            ValueError,
            'Value `E_q` must be provided for `correlation_algorithm`',
        ),
        (None, {'rank_type': 'mbic', 'E_p': 0.0}, ValueError, 'Value `E_p` must be more than 0'),
        (
            None,
            {
                'rank_type': 'mbic',
                'algorithm': 'skip',
                'correlation_algorithm': 'top_down_exhaustive',
                'E_q': 0.0,
            },
            ValueError,
            'Value `E_q` must be more than 0',
        ),
        (
            None,
            {'rank_type': 'mbic', 'E_p': '10'},
            ValueError,
            'Value `E_p` must be denoted with `%`',
        ),
        (
            None,
            {
                'rank_type': 'mbic',
                'algorithm': 'skip',
                'correlation_algorithm': 'top_down_exhaustive',
                'E_q': '10',
            },
            ValueError,
            'Value `E_q` must be denoted with `%`',
        ),
        (
            ('nonmem', 'pheno.mod'),
            {'keep': ('X',)},
            ValueError,
            'Symbol `X` does not exist in input model',
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
    if not model_path:
        model_path = ('nonmem/pheno.mod',)
    path = testdata.joinpath(*model_path)
    model = load_model_for_test(path)
    results = parse_modelfit_results(model, path)

    harmless_arguments = dict(
        algorithm='top_down_exhaustive',
    )

    kwargs = {'model': model, 'results': results, **harmless_arguments, **arguments}

    with pytest.raises(exception, match=match):
        validate_input(**kwargs)
