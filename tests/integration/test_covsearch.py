import pytest

from pharmpy.internals.fs.cwd import chdir
from pharmpy.tools import run_tool


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
