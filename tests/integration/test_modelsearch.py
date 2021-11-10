import shutil

import pytest

from pharmpy import Model
from pharmpy.modeling import fit, run_tool
from pharmpy.utils import TemporaryDirectoryChanger


@pytest.mark.parametrize(
    'mfl, no_of_models',
    [
        ('ABSORPTION(ZO)\nPERIPHERALS(1)', 4),
        ('ABSORPTION(ZO)\nTRANSITS(1)', 3),
        ('ABSORPTION([ZO,SEQ-ZO-FO])\nPERIPHERALS(1)', 7),
        ('LAGTIME()\nTRANSITS(1)', 2),
    ],
)
def test_exhaustive_stepwise(tmp_path, testdata, mfl, no_of_models):
    with TemporaryDirectoryChanger(tmp_path):
        shutil.copy2(testdata / 'nonmem' / 'models' / 'mox2.mod', tmp_path)
        shutil.copy2(testdata / 'nonmem' / 'models' / 'mx19B.csv', tmp_path)
        model = Model('mox2.mod')
        model.dataset_path = tmp_path / 'mx19B.csv'
        res = run_tool('modelsearch', 'exhaustive_stepwise', mfl, model=model)

        assert len(res.summary) == no_of_models
        assert len(res.models) == no_of_models
        assert all(int(model.modelfit_results.ofv) in range(-1500, -1400) for model in res.models)
        rundir = tmp_path / 'modelsearch_dir1'
        assert rundir.is_dir()
        assert len(list((rundir / 'models').iterdir())) == no_of_models + 1


def test_exhaustive_stepwise_already_fit(tmp_path, testdata):
    with TemporaryDirectoryChanger(tmp_path):
        shutil.copy2(testdata / 'nonmem' / 'models' / 'mox2.mod', tmp_path)
        shutil.copy2(testdata / 'nonmem' / 'models' / 'mx19B.csv', tmp_path)
        model = Model('mox2.mod')
        model.dataset_path = tmp_path / 'mx19B.csv'

        fit(model)

        mfl = 'ABSORPTION(ZO)\nPERIPHERALS(1)'
        res = run_tool('modelsearch', 'exhaustive_stepwise', mfl, model=model)

        assert len(res.summary) == 4
        assert len(res.models) == 4
        assert all(int(model.modelfit_results.ofv) in range(-1500, -1400) for model in res.models)
        rundir = tmp_path / 'modelsearch_dir1'
        assert rundir.is_dir()
        assert len(list((rundir / 'models').iterdir())) == 4
