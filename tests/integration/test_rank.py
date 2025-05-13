import itertools

import pytest

from pharmpy.internals.fs.cwd import chdir
from pharmpy.modeling import add_covariate_effect, set_name
from pharmpy.tools import fit, run_rank


@pytest.mark.parametrize(
    'kwargs, no_of_ranked_models, best_model_name',
    [
        ({'rank_type': 'ofv'}, 5, 'model1'),
        ({'rank_type': 'bic_mixed'}, 5, 'model1'),
        ({'rank_type': 'lrt', 'cutoff': 0.05}, 3, 'model1'),
    ],
)
def test_rank_dummy(
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

        res = run_rank(
            model_ref=model_base,
            results_ref=results[0],
            models_cand=models,
            results_cand=results[1:],
            **kwargs,
        )

        assert len(res.summary_tool) == len([model_base] + models)
        assert len(res.summary_tool.dropna(subset=['rank'])) == no_of_ranked_models
        assert res.final_model.name == best_model_name
        assert res.final_results
