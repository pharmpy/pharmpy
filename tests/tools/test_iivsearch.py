from io import StringIO

import pytest

from pharmpy import Model
from pharmpy.modeling import (
    add_iiv,
    add_peripheral_compartment,
    add_pk_iiv,
    create_joint_distribution,
)
from pharmpy.tools.iivsearch.algorithms import (
    _get_eta_combinations,
    _get_param_names,
    _is_current_block_structure,
    brute_force_block_structure,
    create_eta_blocks,
)


@pytest.mark.parametrize(
    'list_of_parameters, block_structure, no_of_models',
    [([], [], 4), (['QP1'], [], 14), ([], ['ETA(1)', 'ETA(2)'], 4)],
)
def test_brute_force_block_structure(testdata, list_of_parameters, block_structure, no_of_models):
    model = Model.create_model(testdata / 'nonmem' / 'models' / 'mox2.mod')
    add_peripheral_compartment(model)
    add_iiv(model, list_of_parameters, 'add')
    if block_structure:
        create_joint_distribution(model, block_structure)

    wf = brute_force_block_structure(model)
    fit_tasks = [task.name for task in wf.tasks if task.name.startswith('run')]

    assert len(fit_tasks) == no_of_models


def test_get_eta_combinations_4_etas(pheno_path):
    model = Model.create_model(pheno_path)
    add_iiv(model, ['TVCL', 'TVV'], 'exp')

    eta_combos = _get_eta_combinations(model.random_variables.iiv)
    assert len(eta_combos) == 15

    block_combos = _get_eta_combinations(model.random_variables.iiv, as_blocks=True)

    assert len(block_combos) == 15

    combos_unique = [(tuple(i) for i in combo) for combo in block_combos]
    assert len(combos_unique) == 15

    len_of_combos = [list(map(lambda i: len(i), combo)) for combo in block_combos]
    assert len_of_combos.count([4]) == 1
    assert len_of_combos.count([1, 3]) == 4
    assert len_of_combos.count([2, 2]) == 3
    assert len_of_combos.count([1, 1, 2]) == 6
    assert len_of_combos.count([1, 1, 1, 1]) == 1


def test_get_eta_combinations_5_etas(pheno_path):
    model = Model.create_model(pheno_path)
    add_iiv(model, ['TVCL', 'TVV', 'TAD'], 'exp')

    eta_combos = _get_eta_combinations(model.random_variables.iiv)
    assert len(eta_combos) == 31

    block_combos = _get_eta_combinations(model.random_variables.iiv, as_blocks=True)
    assert len(block_combos) == 52

    combos_unique = [(tuple(i) for i in combo) for combo in block_combos]
    assert len(combos_unique) == 52

    len_of_combos = [list(map(lambda i: len(i), combo)) for combo in block_combos]
    assert len_of_combos.count([5]) == 1
    assert len_of_combos.count([1, 4]) == 5
    assert len_of_combos.count([2, 3]) == 10
    assert len_of_combos.count([1, 1, 3]) == 10
    assert len_of_combos.count([1, 2, 2]) == 15
    assert len_of_combos.count([1, 1, 1, 2]) == 10
    assert len_of_combos.count([1, 1, 1, 1, 1]) == 1


def test_is_current_block_structure(pheno_path):
    model = Model.create_model(pheno_path)
    add_iiv(model, ['TVCL', 'TVV'], 'exp')
    etas = model.random_variables.iiv

    eta_combos = [['ETA(1)', 'ETA(2)'], ['ETA_TVCL'], ['ETA_TVV']]
    create_joint_distribution(model, eta_combos[0])
    assert _is_current_block_structure(etas, eta_combos)

    eta_combos = [['ETA(1)'], ['ETA(2)'], ['ETA_TVCL', 'ETA_TVV']]
    assert not _is_current_block_structure(etas, eta_combos)

    eta_combos = [['ETA(1)'], ['ETA(2)', 'ETA_TVCL'], ['ETA_TVV']]
    assert not _is_current_block_structure(etas, eta_combos)

    create_joint_distribution(model)
    eta_combos = [['ETA(1)', 'ETA(2)', 'ETA_TVCL', 'ETA_TVV']]
    assert _is_current_block_structure(etas, eta_combos)


def test_create_joint_dist(testdata):
    model = Model.create_model(testdata / 'nonmem' / 'models' / 'mox2.mod')
    add_peripheral_compartment(model)
    add_pk_iiv(model)
    eta_combos = [['ETA(1)', 'ETA(2)'], ['ETA_QP1'], ['ETA_VP1']]
    create_eta_blocks(eta_combos, model)
    assert len(model.random_variables.iiv.distributions()) == 4

    model = Model.create_model(testdata / 'nonmem' / 'models' / 'mox2.mod')
    add_peripheral_compartment(model)
    add_pk_iiv(model)
    create_joint_distribution(model, ['ETA(1)', 'ETA(2)'])
    eta_combos = [['ETA(1)'], ['ETA(2)'], ['ETA(3)', 'ETA_VP1', 'ETA_QP1']]
    create_eta_blocks(eta_combos, model)
    assert len(model.random_variables.iiv.distributions()) == 3


def test_get_param_names(testdata):
    model = Model.create_model(testdata / 'nonmem' / 'models' / 'mox2.mod')

    param_dict = _get_param_names(model)
    param_dict_ref = {'ETA(1)': 'CL', 'ETA(2)': 'VC', 'ETA(3)': 'MAT'}

    assert param_dict == param_dict_ref

    model_code = model.model_code.replace(
        'CL = THETA(1) * EXP(ETA(1))', 'ETA_1 = ETA(1)\nCL = THETA(1) * EXP(ETA_1)'
    )
    model = Model.create_model(StringIO(model_code))

    param_dict = _get_param_names(model)

    assert param_dict == param_dict_ref
