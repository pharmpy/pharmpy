import pytest

from pharmpy.internals.fs.cwd import chdir
from pharmpy.tools import run_tool


def test_default(tmp_path, model_count, start_model):
    with chdir(tmp_path):
        effects = [
            ('CL', 'AGE', 'exp', '*'),
            ('MAT', 'AGE', 'exp', '*'),
            ('KA', 'AGE', 'exp', '*'),
            ('V', 'AGE', 'exp', '*'),
            ('CL', 'SEX', 'cat', '*'),
            ('MAT', 'SEX', 'cat', '*'),
            ('KA', 'SEX', 'cat', '*'),
            ('V', 'SEX', 'cat', '*'),
            ('CL', 'WT', 'exp', '*'),
            ('MAT', 'WT', 'exp', '*'),
            ('KA', 'WT', 'exp', '*'),
            ('V', 'WT', 'exp', '*'),
        ]
        res = run_tool(
            'covsearch',
            effects,
            results=start_model.modelfit_results,
            model=start_model,
        )

        rundir = tmp_path / 'covsearch_dir1'
        assert model_count(rundir) >= len(effects)

        # Checks that description have been added correctly, can maybe be removed
        model_dict = {model.name: model for model in res.models}
        for model in res.models:
            try:
                parent_model = model_dict[model.parent_model]
            except KeyError:
                continue
            if not parent_model.description:
                continue
            cov_effects_child = model.description.split(';')
            cov_effects_parent = parent_model.description.split(';')
            min_cov_effects = min([cov_effects_child, cov_effects_parent], key=len)
            max_cov_effects = max([cov_effects_child, cov_effects_parent], key=len)
            assert set(min_cov_effects).issubset(max_cov_effects)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_default_str(tmp_path, model_count, start_model):
    with chdir(tmp_path):
        run_tool(
            'covsearch',
            'LET(CONTINUOUS, [AGE, WT]); LET(CATEGORICAL, SEX)\n'
            'COVARIATE([CL, MAT, VC], @CONTINUOUS, exp, *)\n'
            'COVARIATE([CL, MAT, VC], @CATEGORICAL, cat, *)',
            results=start_model.modelfit_results,
            model=start_model,
        )

        rundir = tmp_path / 'covsearch_dir1'
        assert model_count(rundir) >= 9
