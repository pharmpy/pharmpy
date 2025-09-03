import itertools

import pytest

from pharmpy.deps import numpy as np
from pharmpy.deps import pandas as pd
from pharmpy.internals.fs.cwd import chdir
from pharmpy.modeling import add_covariate_effect, set_name
from pharmpy.tools import fit, run_modelrank


@pytest.mark.parametrize(
    'kwargs, no_of_ranked_models, best_model_name',
    [
        ({'rank_type': 'ofv'}, 5, 'model1'),
        ({'rank_type': 'bic_mixed'}, 5, 'model1'),
        ({'rank_type': 'lrt', 'alpha': 0.05}, 3, 'model1'),
        ({'rank_type': 'mbic_mixed', 'E': 1.0, 'search_space': 'ABSORPTION(FO)'}, 5, 'model1'),
    ],
)
def test_modelrank_dummy(
    tmp_path, load_model_for_test, testdata, kwargs, no_of_ranked_models, best_model_name
):
    with chdir(tmp_path):
        model_base = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox2.mod')
        cov_effects = itertools.product(['WT', 'AGE'], ['CL', 'VC'])
        models = []
        for i, (cov, param) in enumerate(cov_effects):
            model = set_name(model_base, f'model{i}')
            model = add_covariate_effect(model, param, cov, effect='exp')
            models.append(model)
        results = fit([model_base] + models, esttool='dummy')

        res = run_modelrank(
            models=[model_base] + models,
            results=results,
            ref_model=model_base,
            **kwargs,
        )

        assert len(res.summary_tool) == len([model_base] + models)
        assert len(res.summary_tool.dropna(subset=['rank'])) == no_of_ranked_models

        assert res.final_model.name == best_model_name
        assert res.final_results

        assert len(res.summary_strictness) == len([model_base] + models)
        assert len(res.summary_selection_criteria) == len([model_base] + models)
        assert not np.isnan(res.summary_selection_criteria.loc[res.final_model.name, 'rank_val'])
        assert (
            bool(res.summary_strictness.loc[res.final_model.name, 'strictness_fulfilled']) is True
        )


@pytest.mark.parametrize(
    'kwargs, rse, no_of_models, no_of_steps, best_model_name',
    [
        ({'rank_type': 'ofv'}, 0.7, 6, 2, 'model1'),
        ({'rank_type': 'ofv'}, 0.5, 10, 6, None),
    ],
)
def test_modelrank_uncertainty_dummy(
    tmp_path, load_model_for_test, testdata, kwargs, rse, no_of_models, no_of_steps, best_model_name
):
    with chdir(tmp_path):
        model_base = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox2.mod')
        cov_effects = itertools.product(['WT', 'AGE'], ['CL', 'VC'])
        models = []
        for i, (cov, param) in enumerate(cov_effects):
            model = set_name(model_base, f'model{i}')
            model = add_covariate_effect(model, param, cov, effect='exp')
            models.append(model)
        results = fit([model_base] + models, esttool='dummy')

        strictness = f'minimization_successful and rse <= {rse}'
        res = run_modelrank(
            models=[model_base] + models,
            results=results,
            ref_model=model_base,
            strictness=strictness,
            parameter_uncertainty_method='SANDWICH',
            **kwargs,
        )

        assert len(res.summary_tool) == no_of_models - no_of_steps + 1
        assert len(res.summary_strictness) == no_of_models
        assert len(res.summary_selection_criteria) == no_of_models
        assert len(res.summary_models) == no_of_models

        idx1 = res.summary_strictness.index
        idx2 = res.summary_selection_criteria.index
        idx3 = res.summary_models.index

        assert idx1.equals(idx2) and idx1.equals(idx3)

        assert set(res.summary_strictness.index.get_level_values('step')) == set(
            range(0, no_of_steps)
        )

        if best_model_name is not None:
            assert res.final_model.name == best_model_name
            assert res.final_results
            assert not np.isnan(
                res.summary_selection_criteria.loc[0, res.final_model.name]['rank_val']
            )
            assert (
                res.summary_strictness.loc[0, res.final_model.name]['strictness_fulfilled'] is None
            )
            assert bool(res.summary_strictness.iloc[-1]['strictness_fulfilled']) is True
        else:
            assert res.final_model is None
            assert res.final_results is None

            assert pd.isna(res.summary_tool.iloc[-1]['rank'])
