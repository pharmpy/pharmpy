from pharmpy.tools import run_tool
from pharmpy.utils import TemporaryDirectoryChanger


def test_default(tmp_path, model_count, start_model):
    with TemporaryDirectoryChanger(tmp_path):
        res = run_tool(
            'covsearch',
            [
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
            ],
            model=start_model,
        )

        rundir = tmp_path / 'covsearch_dir1'
        assert model_count(rundir) == 54

        assert res.final_model_name == 'mox2+2+7+10+5'


def test_default_str(tmp_path, model_count, start_model):
    with TemporaryDirectoryChanger(tmp_path):
        res = run_tool(
            'covsearch',
            'LET(CONTINUOUS, [AGE, WT]); LET(CATEGORICAL, SEX)\n'
            'COVARIATE([CL, MAT, VC], @CONTINUOUS, exp, *)\n'
            'COVARIATE([CL, MAT, VC], @CATEGORICAL, cat, *)',
            model=start_model,
        )

        rundir = tmp_path / 'covsearch_dir1'
        assert model_count(rundir) == 39

        assert res.final_model_name == 'mox2+2+7+7+6'
