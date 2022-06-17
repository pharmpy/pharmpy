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

        rundir = tmp_path / 'covsearch_dir1'
        assert _model_count(rundir) == 50

        assert res.best_model.name == 'mox2-2-7-10-5'


def test_default_str(tmp_path, start_model):
    with TemporaryDirectoryChanger(tmp_path):
        res = run_tool(
            'covsearch',
            'CONTINUOUS([AGE, WT]); CATEGORICAL(SEX)\n'
            'COVARIATE([CL, MAT, VC], @CONTINUOUS, exp, *)\n'
            'COVARIATE([CL, MAT, VC], @CATEGORICAL, cat, *)',
            model=start_model,
        )

        rundir = tmp_path / 'covsearch_dir1'
        assert _model_count(rundir) == 35

        assert res.best_model.name == 'mox2-2-7-7-6'
