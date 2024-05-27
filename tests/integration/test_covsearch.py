import pytest

from pharmpy.internals.fs.cwd import chdir
from pharmpy.tools import run_tool


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_default_str(tmp_path, model_count, start_modelres):
    with chdir(tmp_path):
        run_tool(
            'covsearch',
            'LET(CONTINUOUS, [AGE, WT]); LET(CATEGORICAL, SEX)\n'
            'COVARIATE?([CL, MAT, VC], @CONTINUOUS, exp, *)\n'
            'COVARIATE?([CL, MAT, VC], @CATEGORICAL, cat, *)',
            results=start_modelres[1],
            model=start_modelres[0],
        )

        rundir = tmp_path / 'covsearch1'
        assert model_count(rundir) == 39 + 2


def test_adaptive_scope_reduction(tmp_path, model_count, start_modelres):
    with chdir(tmp_path):
        run_tool(
            'covsearch',
            'LET(CONTINUOUS, [AGE, WT]); LET(CATEGORICAL, SEX)\n'
            'COVARIATE?([CL, MAT, VC], @CONTINUOUS, exp, *)\n'
            'COVARIATE?([CL, MAT, VC], @CATEGORICAL, cat, *)',
            results=start_modelres[1],
            model=start_modelres[0],
            adaptive_scope_reduction=True,
        )

        rundir = tmp_path / 'covsearch1'
        assert model_count(rundir) == 33 + 2
