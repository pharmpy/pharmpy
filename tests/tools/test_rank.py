import pytest

from pharmpy.deps import pandas as pd
from pharmpy.modeling import remove_parameter_uncertainty_step, set_name
from pharmpy.tools.rank.tool import (
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
    models_cand = [model_ref] * 5
    res_cand = [res_ref] * 5
    me_ref, me_cands = prepare_model_entries(model_ref, res_ref, models_cand, res_cand)
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

    res = rank_models(me_ref, mes_cand, strictness, rank_type, cutoff)

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
    ],
)
def test_get_rank_type_kwargs(rank_type, kwargs):
    assert get_rank_type_kwargs(rank_type) == kwargs


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
        validate_input(model_ref, res_ref, models_cand, res_cand, **kwargs)
