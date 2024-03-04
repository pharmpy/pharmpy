import sys

import pytest

from pharmpy.internals.fs.cwd import chdir
from pharmpy.tools import read_modelfit_results, run_tool

tflite_condition = (
    sys.version_info >= (3, 12)
    and sys.platform == 'win32'
    or sys.version_info >= (3, 12)
    and sys.platform == 'darwin'
)


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

        rundir = tmp_path / 'covsearch_dir1'
        assert model_count(rundir) >= 9


@pytest.mark.skipif(tflite_condition, reason="Skipping tests requiring tflite for Python 3.12")
def test_default_str_dummy(tmp_path, load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox2.mod')
    results = read_modelfit_results(testdata / 'nonmem' / 'models' / 'mox2.mod')
    with chdir(tmp_path):
        run_tool(
            'covsearch',
            'LET(CONTINUOUS, [AGE, WT]); LET(CATEGORICAL, SEX)\n'
            'COVARIATE?([CL, MAT, VC], @CONTINUOUS, exp, *)\n'
            'COVARIATE?([CL, MAT, VC], @CATEGORICAL, cat, *)',
            results=results,
            model=model,
            estimation_tool='dummy',
        )
