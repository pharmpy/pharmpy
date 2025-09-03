from functools import partial

import pytest

from pharmpy.deps import numpy as np
from pharmpy.deps import pandas as pd
from pharmpy.modeling import (
    add_peripheral_compartment,
    create_joint_distribution,
    remove_iiv,
    remove_parameter_uncertainty_step,
    set_initial_estimates,
    set_name,
)
from pharmpy.tools.modelrank.ranking import (
    get_aic,
    get_bic,
    get_ofv,
    get_rank_values,
    perform_lrt,
    rank_model_entries,
)
from pharmpy.tools.modelrank.strictness import (
    evaluate_strictness,
    get_strictness_expr,
    get_strictness_predicates,
    get_strictness_predicates_me,
)
from pharmpy.tools.modelrank.tool import (
    create_candidate_with_uncertainty,
    get_model_entries_to_rank,
    prepare_model_entries,
    rank_models,
    validate_input,
)
from pharmpy.tools.run import read_modelfit_results
from pharmpy.workflows import ModelEntry


def test_prepare_model_entries(load_model_for_test, pheno_path):
    model_ref = load_model_for_test(pheno_path)
    res_ref = read_modelfit_results(pheno_path)
    models_cand = []
    for i in range(5):
        model = set_initial_estimates(model_ref, {'PTVCL': float(i)})
        model = set_name(model, f'model{i}')
        models_cand.append(model)
    res_cand = [res_ref] * 5
    me_ref, me_cands = prepare_model_entries(
        [model_ref] + models_cand, [res_ref] + res_cand, model_ref
    )
    assert len(me_cands) == len(models_cand) == len(res_cand)
    assert me_ref.model == model_ref

    me_ref, me_cands = prepare_model_entries(
        [model_ref] + models_cand, [res_ref] + res_cand, model_ref
    )
    assert len(me_cands) == len(models_cand) == len(res_cand)
    assert me_ref.model == model_ref


def test_create_candidate_with_uncertainty(load_model_for_test, testdata):
    base_model = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox2.mod')
    assert base_model.execution_steps[-1].parameter_uncertainty_method is None
    model_entry = create_candidate_with_uncertainty(base_model, 'run1', 'SANDWICH')
    model = model_entry.model
    assert model.execution_steps[-1].parameter_uncertainty_method == 'SANDWICH'
    assert model.name == 'run1'
    assert model.description != base_model.description
    assert model_entry.modelfit_results is None
    assert model_entry.parent == base_model


@pytest.mark.parametrize(
    'strictness, rank_type, alpha',
    [
        ('minimization_successful', 'ofv', None),
    ],
)
def test_rank_models(
    load_model_for_test, pheno_path, model_entry_factory, strictness, rank_type, alpha
):
    model_ref = load_model_for_test(pheno_path)
    res_ref = read_modelfit_results(pheno_path)
    me_ref = ModelEntry.create(model_ref, modelfit_results=res_ref)
    models_cand = [set_name(model, f'model{i}') for i, model in enumerate([model_ref] * 5)]
    mes_cand = model_entry_factory(models_cand, parent=model_ref)

    res = rank_models(me_ref, mes_cand, strictness, rank_type, alpha, None, None)

    assert len(res.summary_tool) == 6
    assert res.final_model.name == 'model1'

    summary_tool_sorted_by_d_rank_type = res.summary_tool.sort_values(
        by=[f'd{rank_type}'], ascending=False
    )
    summary_tool_sorted_by_rank_type = res.summary_tool.sort_values(by=[rank_type])
    summary_tool_sorted_by_rank = res.summary_tool.sort_values(by=['rank'])
    pd.testing.assert_frame_equal(summary_tool_sorted_by_d_rank_type, summary_tool_sorted_by_rank)
    pd.testing.assert_frame_equal(
        summary_tool_sorted_by_d_rank_type, summary_tool_sorted_by_rank_type
    )


@pytest.mark.parametrize(
    'strictness, model_path, ref_predicates',
    [
        (
            'minimization_successful',
            'models/mox2.mod',
            {'minimization_successful': True, 'strictness_fulfilled': True},
        ),
        (
            '',
            'models/mox2.mod',
            {'strictness_fulfilled': True},
        ),
        (
            'minimization_successful or rounding_errors',
            'models/mox2.mod',
            {
                'minimization_successful': True,
                'rounding_errors': False,
                'strictness_fulfilled': True,
            },
        ),
        (
            'minimization_successful or (rounding_errors and sigdigs >= 0.1)',
            'models/mox2.mod',
            {
                'minimization_successful': True,
                'rounding_errors': False,
                'sigdigs': 3.8,
                'sigdigs >= 0.1': True,
                'strictness_fulfilled': True,
            },
        ),
        (
            'minimization_successful or (rse <= 0.1)',
            'models/mox2.mod',
            {
                'minimization_successful': True,
                'rse': None,
                'rse <= 0.1': None,
                'strictness_fulfilled': True,
            },
        ),
        (
            'not minimization_successful',
            'models/mox2.mod',
            {
                'minimization_successful': True,
                '~minimization_successful': False,
                'strictness_fulfilled': False,
            },
        ),
        (
            'not minimization_successful or not rounding_errors',
            'models/mox2.mod',
            {
                'minimization_successful': True,
                '~minimization_successful': False,
                'rounding_errors': False,
                '~rounding_errors': True,
                'strictness_fulfilled': True,
            },
        ),
        (
            'not estimate_near_boundary',
            'models/mox2.mod',
            {
                'estimate_near_boundary': True,
                '~estimate_near_boundary': False,
                'strictness_fulfilled': False,
            },
        ),
        (
            'not estimate_near_boundary_theta and not estimate_near_boundary_omega',
            'models/mox2.mod',
            {
                'estimate_near_boundary_omega': True,
                '~estimate_near_boundary_omega': False,
                'estimate_near_boundary_theta': False,
                '~estimate_near_boundary_theta': True,
                'strictness_fulfilled': False,
            },
        ),
        (
            'condition_number <= 100',
            'models/mox2.mod',
            {
                'condition_number': None,
                'condition_number <= 100': None,
                'strictness_fulfilled': None,
            },
        ),
        (
            'minimization_successful and (rse <= 1.0)',
            'models/pheno5.mod',
            {
                'minimization_successful': True,
                'rse': 0.5814089116181571,
                'rse <= 1.0': True,
                'strictness_fulfilled': True,
            },
        ),
        (
            'minimization_successful and (rse <= 0.5)',
            'models/pheno5.mod',
            {
                'minimization_successful': True,
                'rse': 0.5814089116181571,
                'rse <= 0.5': False,
                'strictness_fulfilled': False,
            },
        ),
        (
            'condition_number <= 100',
            'models/pheno5.mod',
            {
                'condition_number': 6595781.0,
                'condition_number <= 100': False,
                'strictness_fulfilled': False,
            },
        ),
        (
            '(minimization_successful or (rounding_errors and sigdigs >= 0.1)) and rse <= 0.5',
            'models/mox2.mod',
            {
                'rse': None,
                'rse <= 0.5': None,
                'minimization_successful': True,
                'rounding_errors': False,
                'sigdigs': 3.8,
                'sigdigs >= 0.1': True,
                'strictness_fulfilled': None,
            },
        ),
    ],
)
def test_get_strictness_predicates(
    testdata, load_model_for_test, strictness, model_path, ref_predicates
):
    model_start = load_model_for_test(testdata / 'nonmem' / model_path)
    res_start = read_modelfit_results(testdata / 'nonmem' / model_path)
    me_start = ModelEntry.create(model_start, modelfit_results=res_start)

    expr = get_strictness_expr(strictness)
    predicates = get_strictness_predicates_me(me_start, expr)

    assert predicates.keys() == ref_predicates.keys()
    assert all(_is_equal(v1, v2) for v1, v2 in zip(predicates.values(), ref_predicates.values()))

    me_predicates = get_strictness_predicates([me_start], expr)
    assert predicates.keys() == me_predicates[me_start].keys()
    assert all(
        _is_equal(v1, v2) for v1, v2 in zip(predicates.values(), me_predicates[me_start].values())
    )


def _is_equal(val1, val2):
    if val1 and np.isnan(val1):
        return np.isnan(val2)
    elif isinstance(val1, float):
        return pytest.approx(val1) == pytest.approx(val2)
    else:
        return val1 == val2


@pytest.mark.parametrize(
    'strictness, model_path, strictness_fulfilled',
    [
        ('minimization_successful', 'models/mox2.mod', True),
        ('minimization_successful or rounding_errors', 'models/mox2.mod', True),
        (
            'minimization_successful or (rounding_errors and sigdigs >= 0.1)',
            'models/mox2.mod',
            True,
        ),
        ('minimization_successful or (rse <= 0.1)', 'models/mox2.mod', True),
        ('not minimization_successful', 'models/mox2.mod', False),
        ('not minimization_successful or not rounding_errors', 'models/mox2.mod', True),
        ('minimization_successful and (rse <= 0.1)', 'models/mox2.mod', None),
        ('minimization_successful and not estimate_near_boundary', 'models/mox2.mod', False),
        ('minimization_successful and not estimate_near_boundary_theta', 'models/mox2.mod', True),
        ('minimization_successful and not estimate_near_boundary_omega', 'models/mox2.mod', False),
        ('maxevals_exceeded', 'models/mox2.mod', False),
        ('condition_number <= 100', 'models/mox2.mod', None),
        ('final_zero_gradient', 'models/mox2.mod', None),
        ('final_zero_gradient', 'pheno.mod', False),
        ('minimization_successful and (rse <= 1.0)', 'models/pheno5.mod', True),
        ('minimization_successful and (rse <= 0.5)', 'models/pheno5.mod', False),
        ('condition_number <= 100', 'models/pheno5.mod', False),
    ],
)
def test_evaluate_strictness(
    testdata, load_model_for_test, strictness, model_path, strictness_fulfilled
):
    model_start = load_model_for_test(testdata / 'nonmem' / model_path)
    res_start = read_modelfit_results(testdata / 'nonmem' / model_path)
    me_start = ModelEntry.create(model_start, modelfit_results=res_start)

    expr = get_strictness_expr(strictness)
    predicates = get_strictness_predicates_me(me_start, expr)
    assert evaluate_strictness(expr, predicates) == strictness_fulfilled


@pytest.mark.parametrize(
    'rank_func, kwargs, ref_dict',
    [
        (get_ofv, {'ref_value': None}, {'dofv': 0, 'ofv': -1292.18676}),
        (
            get_ofv,
            {'ref_value': -1200.0},
            {'dofv': 92.18676, 'ofv': -1292.18676},
        ),
        (
            get_aic,
            {'ref_value': None},
            {
                'ofv': -1292.18676,
                'aic_penalty': 16,
                'daic': 0,
                'aic': -1276.18676,
            },
        ),
        (
            get_aic,
            {'ref_value': -1200.0},
            {
                'ofv': -1292.18676,
                'aic_penalty': 16,
                'daic': 76.18676,
                'aic': -1276.18676,
            },
        ),
        (
            get_bic,
            {'ref_value': None, 'rank_type': 'bic_fixed', 'search_space': None},
            {
                'ofv': -1292.18676,
                'bic_penalty': 55.30990,
                'dbic': 0,
                'bic': -1236.87686,
            },
        ),
        (
            get_bic,
            {'ref_value': -1200, 'rank_type': 'bic_fixed', 'search_space': None},
            {
                'ofv': -1292.18676,
                'bic_penalty': 55.30990,
                'dbic': 36.87686,
                'bic': -1236.87686,
            },
        ),
        (
            get_bic,
            {'ref_value': None, 'rank_type': 'bic_mixed', 'search_space': None},
            {
                'ofv': -1292.18676,
                'bic_penalty': 36.94695,
                'dbic': 0,
                'bic': -1255.23981,
            },
        ),
        (
            get_bic,
            {'ref_value': None, 'rank_type': 'bic_iiv', 'search_space': None},
            {
                'ofv': -1292.18676,
                'bic_penalty': 17.16184,
                'dbic': 0,
                'bic': -1275.02492,
            },
        ),
        (
            get_bic,
            {
                'ref_value': None,
                'rank_type': 'mbic_mixed',
                'search_space': 'IIV?([CL,VC,MAT],exp)',
                'E': 1.0,
            },
            {
                'ofv': -1292.18676,
                'bic_penalty': 36.94695,
                'mbic_penalty': 6.59167,
                'dbic': 0,
                'bic': -1248.64813,
            },
        ),
        (
            get_bic,
            {
                'ref_value': None,
                'rank_type': 'mbic_mixed',
                'search_space': 'IIV?([CL,VC,MAT],exp);COV?([CL,VC,MAT])',
                'E': (1.0, 1.0),
            },
            {
                'ofv': -1292.18676,
                'bic_penalty': 36.94695,
                'mbic_penalty': 8.78890,
                'dbic': 0,
                'bic': -1246.45091,
            },
        ),
    ],
)
def test_get_rank_values(testdata, load_model_for_test, rank_func, kwargs, ref_dict):
    model_start = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox2.mod')
    model_start = create_joint_distribution(model_start, rvs=['ETA_1', 'ETA_2'])
    res_start = read_modelfit_results(testdata / 'nonmem' / 'models' / 'mox2.mod')
    me_start = ModelEntry.create(model_start, modelfit_results=res_start)

    rank_dict = rank_func(me_start, **kwargs)
    assert rank_dict.keys() == ref_dict.keys()
    for key1, key2 in zip(rank_dict.keys(), ref_dict.keys()):
        val1 = rank_dict[key1]
        val2 = ref_dict[key2]
        assert pytest.approx(val1) == pytest.approx(val2)


class DummyResults:
    def __init__(
        self,
        name,
        ofv,
    ):
        self.name = name
        self.ofv = ofv


@pytest.mark.parametrize(
    'funcs, ofv, alpha, ref_dict',
    [
        (
            [add_peripheral_compartment],
            -1300,
            0.05,
            {
                'df': 2,
                'alpha': 0.05,
                'cutoff': 5.99146,
                'p_value': 0.02010836,
                'dofv': 7.81324,
                'ofv': -1300,
                'significant': True,
            },
        ),
        (
            [add_peripheral_compartment],
            -1300,
            0.01,
            {
                'df': 2,
                'alpha': 0.01,
                'cutoff': 9.21034,
                'p_value': 0.02010836,
                'dofv': 7.81324,
                'ofv': -1300,
                'significant': False,
            },
        ),
        (
            [add_peripheral_compartment],
            -1295,
            0.05,
            {
                'df': 2,
                'alpha': 0.05,
                'cutoff': 5.99146,
                'p_value': 0.244970,
                'dofv': 2.81324,
                'ofv': -1295,
                'significant': False,
            },
        ),
        (
            [partial(create_joint_distribution, rvs=['ETA_1', 'ETA_2'])],
            -1300,
            0.05,
            {
                'df': 1,
                'alpha': 0.05,
                'cutoff': 3.84146,
                'p_value': 0.00518648,
                'dofv': 7.81324,
                'ofv': -1300,
                'significant': True,
            },
        ),
        (
            [partial(remove_iiv, to_remove=['ETA_3'])],
            -1290,
            (0.01, 0.05),
            {
                'df': -1,
                'alpha': 0.05,
                'cutoff': -3.84146,
                'p_value': 0.1392018,
                'dofv': -2.18676,
                'ofv': -1290,
                'significant': True,
            },
        ),
    ],
)
def test_perform_lrt(testdata, load_model_for_test, funcs, ofv, alpha, ref_dict):
    model_parent = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox2.mod')
    res_parent = read_modelfit_results(testdata / 'nonmem' / 'models' / 'mox2.mod')
    me_parent = ModelEntry.create(model_parent, modelfit_results=res_parent)

    model_child = set_name(model_parent, 'cand')
    for func in funcs:
        model_child = func(model_child)

    res_child = DummyResults(name='cand', ofv=ofv)
    me_child = ModelEntry.create(model_child, modelfit_results=res_child, parent=model_parent)

    rank_dict = perform_lrt(me_child, me_parent, alpha)
    assert rank_dict.keys() == ref_dict.keys()
    for key1, key2 in zip(rank_dict.keys(), ref_dict.keys()):
        val1 = rank_dict[key1]
        val2 = ref_dict[key2]
        if isinstance(val1, bool):
            assert val1 == val2
        elif np.isnan(val1):
            assert np.isnan(val2)
        else:
            assert pytest.approx(val1) == pytest.approx(val2)


@pytest.mark.parametrize(
    'rank_type, rank_kwargs, final_model',
    [
        ('ofv', {'alpha': None, 'search_space': None, 'E': None}, 'model1'),
        ('aic', {'alpha': None, 'search_space': None, 'E': None}, 'model1'),
        ('bic_mixed', {'alpha': None, 'search_space': None, 'E': None}, 'model1'),
        ('bic_iiv', {'alpha': None, 'search_space': None, 'E': None}, 'model1'),
        ('bic_fixed', {'alpha': None, 'search_space': None, 'E': None}, 'model1'),
        (
            'mbic_mixed',
            {'alpha': None, 'search_space': 'IIV?([CL,VC,MAT],exp)', 'E': 1.0},
            'model1',
        ),
        (
            'mbic_iiv',
            {
                'alpha': None,
                'search_space': 'IIV?([CL,VC,MAT],exp);COV?([CL,VC,MAT])',
                'E': (1.0, 1.0),
            },
            'model1',
        ),
        ('lrt', {'alpha': 0.05, 'search_space': None, 'E': None}, 'model1'),
        (
            'lrt',
            {'alpha': 0.0000000000000000000000000001, 'search_space': None, 'E': None},
            'mox2',
        ),
    ],
)
def test_rank_model_entries(
    load_model_for_test, testdata, model_entry_factory, rank_type, rank_kwargs, final_model
):
    model_ref = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox2.mod')
    res_ref = read_modelfit_results(testdata / 'nonmem' / 'models' / 'mox2.mod')
    me_ref = ModelEntry.create(model_ref, modelfit_results=res_ref)

    models_cand = _create_candidates(model_ref)
    mes_cand = model_entry_factory(models_cand, ref_val=res_ref.ofv, parent=model_ref)

    rank_values = get_rank_values(
        me_ref, mes_cand, rank_type, **rank_kwargs, mes_to_rank=[me_ref] + mes_cand
    )

    assert len(rank_values) == len(models_cand) + 1
    assert round(rank_values[me_ref]['ofv'], 5) == -1292.18676
    if rank_type not in ('ofv', 'lrt'):
        assert rank_values[me_ref]['rank_val'] > rank_values[me_ref]['ofv']

    ranking = rank_model_entries(rank_values, rank_type)

    me_best = list(ranking.keys())[0]
    assert me_best.model.name == final_model

    if rank_type == 'lrt' and final_model == model_ref.name:
        assert all(val['significant'] is not True for val in ranking.values())
    else:
        assert ranking[me_best]['rank_val'] == min(d['rank_val'] for d in ranking.values())


def _create_candidates(model_ref):
    funcs = [
        add_peripheral_compartment,
        partial(create_joint_distribution, rvs=['ETA_1', 'ETA_2']),
        create_joint_distribution,
    ]

    models_cand = []
    for i, func in enumerate(funcs):
        model = set_name(model_ref, f'model{i}')
        model = func(model)
        models_cand.append(model)

    return models_cand


@pytest.mark.parametrize(
    'strictness, no_of_mes_not_strict, no_of_mes_strict',
    [
        ('minimization_successful', 4, 4),
        ('minimization_successful and (rse <= 0.1)', 4, 0),
        ('not minimization_successful', 0, 0),
        ('not minimization_successful and (rse <= 0.1)', 0, 0),
    ],
)
def test_get_model_entries_to_rank(
    testdata,
    load_model_for_test,
    model_entry_factory,
    strictness,
    no_of_mes_not_strict,
    no_of_mes_strict,
):
    model_ref = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox2.mod')
    res_ref = read_modelfit_results(testdata / 'nonmem' / 'models' / 'mox2.mod')
    me_ref = ModelEntry.create(model_ref, modelfit_results=res_ref)

    models_cand = _create_candidates(model_ref)
    mes_cand = model_entry_factory(models_cand, ref_val=res_ref.ofv, parent=model_ref)

    expr = get_strictness_expr(strictness)
    me_predicates = get_strictness_predicates([me_ref] + mes_cand, expr)

    assert len(get_model_entries_to_rank(me_predicates, strict=False)) == no_of_mes_not_strict
    assert len(get_model_entries_to_rank(me_predicates, strict=True)) == no_of_mes_strict


@pytest.mark.parametrize(
    (
        'no_of_res',
        'kwargs',
        'exception',
        'match',
    ),
    [
        (
            4,
            dict(),
            ValueError,
            'Length mismatch',
        ),
        (
            5,
            {'strictness': 'rse'},
            ValueError,
            '`parameter_uncertainty_method`',
        ),
        (
            5,
            dict(search_space=1),
            TypeError,
            'Invalid `search_space`',
        ),
        (
            5,
            dict(search_space='x'),
            ValueError,
            'Invalid `search_space`',
        ),
        (5, {'rank_type': 'ofv', 'E': 1.0}, ValueError, 'E can only be provided'),
        (5, {'rank_type': 'mbic_mixed'}, ValueError, 'Argument `search_space` must be provided'),
        (
            5,
            {'search_space': 'ABSORPTION(FO)', 'rank_type': 'mbic_mixed'},
            ValueError,
            'Value `E` must be provided when using mbic',
        ),
        (
            5,
            {'search_space': 'ABSORPTION(FO)', 'rank_type': 'mbic_mixed', 'E': 0.0},
            ValueError,
            'Value `E` must be more than 0',
        ),
        (
            5,
            {'search_space': 'ABSORPTION(FO)', 'rank_type': 'mbic_mixed', 'E': '10'},
            ValueError,
            'Value `E` must be denoted with `%`',
        ),
    ],
)
def test_validate_input_raises(
    load_model_for_test, pheno_path, no_of_res, kwargs, exception, match
):
    model_ref = load_model_for_test(pheno_path)
    model_ref = remove_parameter_uncertainty_step(model_ref)
    res_ref = read_modelfit_results(pheno_path)
    models_cand = [model_ref] * 5
    res_cand = [res_ref] * no_of_res

    with pytest.raises(exception, match=match):
        validate_input(
            models=[model_ref] + models_cand,
            results=[res_ref] + res_cand,
            ref_model=model_ref,
            **kwargs,
        )


def test_validate_input_raises_ref_model(load_model_for_test, pheno_path):
    model_ref = load_model_for_test(pheno_path)
    res_ref = read_modelfit_results(pheno_path)

    models_cand = []
    for i in range(5):
        model = set_initial_estimates(model_ref, {'PTVCL': float(i)})
        model = set_name(model, f'model{i}')
        models_cand.append(model)

    res_cand = [res_ref] * 5

    with pytest.raises(ValueError, match='Incorrect `ref_model`'):
        validate_input(models=models_cand, results=res_cand, ref_model=model_ref)

    with pytest.raises(ValueError, match='Cannot perform LRT'):
        validate_input(
            models=[model_ref] + models_cand,
            results=[res_ref] + res_cand,
            ref_model=model_ref,
            rank_type='lrt',
        )
