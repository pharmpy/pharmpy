import shutil
from pathlib import Path

import pytest

from pharmpy.internals.fs.cwd import chdir
from pharmpy.tools import run_amd


def _model_count(rundir: Path):
    return sum(
        map(
            lambda path: 0 if path.name in ['.lock', '.datasets'] else 1,
            ((rundir / 'models').iterdir()),
        )
    )


@pytest.mark.filterwarnings(
    'ignore:.*Adjusting initial estimates to create positive semidefinite omega/sigma matrices.',
    'ignore::UserWarning',
)
def test_amd(tmp_path, testdata):
    with chdir(tmp_path):
        shutil.copy2(testdata / 'nonmem' / 'models' / 'moxo_simulated_amd.csv', '.')
        shutil.copy2(testdata / 'nonmem' / 'models' / 'moxo_simulated_amd.datainfo', '.')
        input = 'moxo_simulated_amd.csv'
        res = run_amd(
            input,
            modeltype='pk_oral',
            search_space='PERIPHERALS(1)',
            occasion='VISI',
        )

        rundir = tmp_path / 'amd_dir1'
        assert rundir.is_dir()
        assert (rundir / 'results.json').exists()
        assert (rundir / 'results.csv').exists()
        subrundir = [
            'modelfit',
            'modelsearch',
            'iivsearch',
            'ruvsearch',
            'iovsearch',
            'allometry',
            'covsearch',
        ]
        for dir in subrundir:
            dir = rundir / dir
            assert _model_count(dir) >= 1

        assert len(res.summary_tool) == len(subrundir)
        assert len(res.summary_models) >= 1
        assert len(res.summary_individuals_count) >= 1
