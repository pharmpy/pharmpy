import pytest

from pharmpy.deps import pandas as pd
from pharmpy.modeling import remove_parameter_uncertainty_step, set_initial_estimates, set_name
from pharmpy.tools.modelrank.strictness import (
    evaluate_strictness,
    get_strictness_expr,
    get_strictness_predicates,
)
from pharmpy.tools.modelrank.tool import (
    get_rank_type_kwargs,
    prepare_model_entries,
    rank_models,
    validate_input,
)
from pharmpy.tools.run import read_modelfit_results
from pharmpy.workflows import ModelEntry


def test_prepare_model_entries(load_model_for_test, pheno_path):
    model_ref = load_model_for_test(pheno_path)
    res_ref = read_modelfit_results(pheno_path)
    models_cand = [set_initial_estimates(model_ref, {'PTVCL': float(i)}) for i in range(5)]
    res_cand = [res_ref] * 5
    me_ref, me_cands = prepare_model_entries(
        [model_ref] + models_cand, [res_ref] + res_cand, model_ref
    )
    assert len(me_cands) == len(models_cand) == len(res_cand)
    assert me_ref.model == model_ref


@pytest.mark.parametrize(
    'strictness, rank_type, cutoff',
    [
        ('minimization_successful', 'ofv', None),
    ],
)
def test_rank_models(
    load_model_for_test, pheno_path, model_entry_factory, strictness, rank_type, cutoff
):
    model_ref = load_model_for_test(pheno_path)
    res_ref = read_modelfit_results(pheno_path)
    me_ref = ModelEntry.create(model_ref, modelfit_results=res_ref)
    models_cand = [set_name(model, f'model{i}') for i, model in enumerate([model_ref] * 5)]
    mes_cand = model_entry_factory(models_cand)

    parent_dict = {me.model.name: me_ref.model.name for me in [me_ref] + mes_cand}
    res = rank_models(me_ref, mes_cand, strictness, rank_type, cutoff, None, parent_dict)

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
    'rank_type, kwargs',
    [
        ('ofv', {'rank_type': 'ofv'}),
        ('bic_mixed', {'rank_type': 'bic', 'bic_type': 'mixed'}),
        ('bic_iiv', {'rank_type': 'bic', 'bic_type': 'iiv'}),
        ('mbic_mixed', {'rank_type': 'bic', 'bic_type': 'mixed'}),
    ],
)
def test_get_rank_type_kwargs(rank_type, kwargs):
    assert get_rank_type_kwargs(rank_type) == kwargs


@pytest.mark.parametrize(
    'strictness, model_path, ref_predicates',
    [
        ('minimization_successful', 'models/mox2.mod', {'minimization_successful': True}),
        (
            'minimization_successful or rounding_errors',
            'models/mox2.mod',
            {'minimization_successful': True, 'rounding_errors': False},
        ),
        (
            'minimization_successful or (rounding_errors and sigdigs >= 0.1)',
            'models/mox2.mod',
            {
                'minimization_successful': True,
                'rounding_errors': False,
                'sigdigs': 3.8,
                'sigdigs >= 0.1': True,
            },
        ),
        (
            'minimization_successful or (rse <= 0.1)',
            'models/mox2.mod',
            {'minimization_successful': True, 'rse': None, 'rse <= 0.1': None},
        ),
        (
            'not minimization_successful',
            'models/mox2.mod',
            {'~minimization_successful': False, 'minimization_successful': True},
        ),
        (
            'not minimization_successful or not rounding_errors',
            'models/mox2.mod',
            {
                '~minimization_successful': False,
                'minimization_successful': True,
                '~rounding_errors': True,
                'rounding_errors': False,
            },
        ),
        (
            'not estimate_near_boundary',
            'models/mox2.mod',
            {'~estimate_near_boundary': False, 'estimate_near_boundary': True},
        ),
        (
            'not estimate_near_boundary_theta and not estimate_near_boundary_omega',
            'models/mox2.mod',
            {
                '~estimate_near_boundary_theta': True,
                'estimate_near_boundary_theta': False,
                '~estimate_near_boundary_omega': False,
                'estimate_near_boundary_omega': True,
            },
        ),
        (
            'condition_number <= 100',
            'models/mox2.mod',
            {'condition_number': None, 'condition_number <= 100': None},
        ),
        (
            'minimization_successful and (rse <= 1.0)',
            'models/pheno5.mod',
            {'minimization_successful': True, 'rse': 0.5814089116181571, 'rse <= 1.0': True},
        ),
        (
            'minimization_successful and (rse <= 0.5)',
            'models/pheno5.mod',
            {'minimization_successful': True, 'rse': 0.5814089116181571, 'rse <= 0.5': False},
        ),
        (
            'condition_number <= 100',
            'models/pheno5.mod',
            {'condition_number': 6828363899484635.0, 'condition_number <= 100': False},
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
    predicates = get_strictness_predicates(me_start, expr)
    assert predicates.keys() == ref_predicates.keys()
    for key, value in predicates.items():
        assert ref_predicates[key] == value


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
    predicates = get_strictness_predicates(me_start, expr)
    assert evaluate_strictness(expr, predicates) == strictness_fulfilled


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
