from pathlib import Path

from pharmpy.tools import run_tool
from pharmpy.utils import TemporaryDirectoryChanger


def _model_count(rundir: Path):
    return sum(
        map(
            lambda path: 0 if path.name in ['.lock', '.datasets'] else 1,
            ((rundir / 'models').iterdir()),
        )
    )


def test_default(tmp_path, start_model):
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
        import pandas as pd
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_colwidth', None)

        print(res.summary_tool)
        print(res.summary_errors)

        rundir = tmp_path / 'covsearch_dir1'

        for model in res.models:
            lst_path = rundir / 'models' / model.name / f'{model.name}.lst'
            with open(lst_path, 'r') as f:
                print(f.read())

        assert _model_count(rundir) == 54

        assert res.best_model.name == 'mox2+2+7+10+5'


def test_default_str(tmp_path, start_model):
    with TemporaryDirectoryChanger(tmp_path):
        res = run_tool(
            'covsearch',
            'LET(CONTINUOUS, [AGE, WT]); LET(CATEGORICAL, SEX)\n'
            'COVARIATE([CL, MAT, VC], @CONTINUOUS, exp, *)\n'
            'COVARIATE([CL, MAT, VC], @CATEGORICAL, cat, *)',
            model=start_model,
        )

        rundir = tmp_path / 'covsearch_dir1'
        assert _model_count(rundir) == 39

        assert res.best_model.name == 'mox2+2+7+7+6'
