from functools import partial

import pytest

from pharmpy.modeling import (
    add_iiv,
    add_iov,
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
from pharmpy.tools import read_modelfit_results
from pharmpy.tools.iivsearch.algorithms import (
    _create_param_dict,
    _extract_clearance_parameter,
    _is_rv_block_structure,
    _rv_block_structures,
    create_block_structure_candidate_entry,
    create_description,
    create_eta_blocks,
    create_no_of_etas_candidate_entry,
    get_eta_names,
    td_exhaustive_block_structure,
    td_exhaustive_no_of_etas,
)
from pharmpy.tools.iivsearch.tool import add_iiv as iivsearch_add_iiv
from pharmpy.tools.iivsearch.tool import (
    create_param_mapping,
    create_workflow,
    get_mbic_penalties,
    get_ref_model,
    prepare_algorithms,
    prepare_base_model,
    prepare_input_model,
    update_linearized_base_model,
    validate_input,
)
from pharmpy.workflows import ModelEntry, Workflow


def test_prepare_input_model(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox2.mod')
    res = read_modelfit_results(testdata / 'nonmem' / 'models' / 'mox2.mod')
    model_input, me_input = prepare_input_model(model, res)

    assert model_input.description == '[CL]+[VC]+[MAT]'
    assert me_input.modelfit_results is not None


@pytest.mark.parametrize(
    'iiv_strategy, linearize, no_of_params_added, description, has_mfr',
    [
        ('no_add', False, 0, '[CL]+[VC]+[MAT]', True),
        ('add_diagonal', False, 2, '[CL]+[VC]+[MAT]+[QP1]+[VP1]', False),
        ('fullblock', False, 12, '[CL,VC,MAT,QP1,VP1]', False),
        ('add_diagonal', True, 0, '[CL]+[VC]+[MAT]+[QP1]+[VP1]', False),
    ],
)
def test_prepare_base_model(
    load_model_for_test, testdata, iiv_strategy, linearize, no_of_params_added, description, has_mfr
):
    model_start = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox2.mod')
    res_input = read_modelfit_results(testdata / 'nonmem' / 'models' / 'mox2.mod')
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
    model_start = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox2.mod')
    model_start = remove_iiv(model_start, ['ETA_3'])
    res_start = read_modelfit_results(testdata / 'nonmem' / 'models' / 'mox2.mod')

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
    model = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox2.mod')
    res = read_modelfit_results(testdata / 'nonmem' / 'models' / 'mox2.mod')
    me_input = ModelEntry.create(model, modelfit_results=res)

    param_mapping = create_param_mapping(me_input, linearize=False)
    assert param_mapping is None

    param_mapping = create_param_mapping(me_input, linearize=True)
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
        model_start = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox2.mod')
        res_input = read_modelfit_results(testdata / 'nonmem' / 'models' / 'mox2.mod')
        model_input = add_peripheral_compartment(model_start)
    else:
        model_start = load_model_for_test(testdata / 'nonmem' / 'pheno_pd.mod')
        res_input = read_modelfit_results(testdata / 'nonmem' / 'pheno_pd.mod')
        model_start = fix_parameters(model_start, model_start.parameters.names)
        model_input = set_direct_effect(model_start, expr='linear')

    model_iiv = iivsearch_add_iiv(iiv_strategy, model_input, res_input, linearize=linearize)
    assert (
        len(model_iiv.parameters.nonfixed) - len(model_input.parameters.nonfixed)
        == no_of_added_params
    )


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
    model_start = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox2.mod')
    res_start = read_modelfit_results(testdata / 'nonmem' / 'models' / 'mox2.mod')
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
    model_start = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox2.mod')
    res_start = read_modelfit_results(testdata / 'nonmem' / 'models' / 'mox2.mod')
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
    model = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox2.mod')
    res = read_modelfit_results(testdata / 'nonmem' / 'models' / 'mox2.mod')
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


def test_is_rv_block_structure(load_model_for_test, pheno_path):
    model = load_model_for_test(pheno_path)
    res = read_modelfit_results(pheno_path)
    model = add_iiv(model, ['TAD', 'S1'], 'exp')

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
    model = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox2.mod')
    res = read_modelfit_results(testdata / 'nonmem' / 'models' / 'mox2.mod')

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
    'as_fullblock, penalties_ref',
    [
        (
            False,
            [2.77, 1.39, 0],
        ),
        (True, [9.36, 3.58, 0]),
    ],
)
def test_get_mbic_penalties(load_model_for_test, testdata, as_fullblock, penalties_ref):
    model_base = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox2.mod')
    if as_fullblock:
        model_base = create_joint_distribution(model_base)
    cand1 = remove_iiv(model_base, 'VC')
    cand2 = remove_iiv(model_base, ['VC', 'MAT'])

    penalties = get_mbic_penalties(model_base, [model_base, cand1, cand2], ['CL'], E_p=1, E_q=1)
    assert [round(p, 2) for p in penalties] == penalties_ref


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
    model = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    results = read_modelfit_results(testdata / 'nonmem' / 'pheno.mod')
    assert isinstance(
        create_workflow(model=model, results=results, algorithm='top_down_exhaustive'), Workflow
    )


def test_validate_input_with_model(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    results = read_modelfit_results(testdata / 'nonmem' / 'pheno.mod')
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
    results = read_modelfit_results(path)

    harmless_arguments = dict(
        algorithm='top_down_exhaustive',
    )

    kwargs = {'model': model, 'results': results, **harmless_arguments, **arguments}

    with pytest.raises(exception, match=match):
        validate_input(**kwargs)


@pytest.mark.parametrize(
    ('model_path', 'arguments', 'warning', 'match'),
    [(["nonmem", "pheno.mod"], dict(keep=["CL"]), UserWarning, 'Parameter')],
)
def test_validate_input_warn(
    load_model_for_test,
    testdata,
    model_path,
    arguments,
    warning,
    match,
):
    if not model_path:
        model_path = ('nonmem/pheno.mod',)
    path = testdata.joinpath(*model_path)
    model = load_model_for_test(path)
    results = read_modelfit_results(path)
    model = remove_iiv(model, 'CL')

    harmless_arguments = dict(
        algorithm='top_down_exhaustive',
    )

    kwargs = {'model': model, 'results': results, **harmless_arguments, **arguments}

    with pytest.warns(warning, match=match):
        validate_input(**kwargs)
