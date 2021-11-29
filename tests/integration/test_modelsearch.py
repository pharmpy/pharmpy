import shutil

import pytest

from pharmpy import Model
from pharmpy.modeling import add_iiv, fit, run_tool
from pharmpy.utils import TemporaryDirectoryChanger


@pytest.mark.parametrize(
    'mfl, no_of_models, best_model_name',
    [
        ('ABSORPTION(ZO)\nPERIPHERALS(1)', 4, 'modelsearch_candidate2'),
        ('ABSORPTION(ZO)\nTRANSITS(1)', 3, 'mox2'),
        ('ABSORPTION([ZO,SEQ-ZO-FO])\nPERIPHERALS(1)', 7, 'modelsearch_candidate3'),
        ('LAGTIME()\nTRANSITS(1)', 2, 'mox2'),
    ],
)
def test_exhaustive_stepwise(tmp_path, testdata, mfl, no_of_models, best_model_name):
    with TemporaryDirectoryChanger(tmp_path):
        shutil.copy2(testdata / 'nonmem' / 'models' / 'mox2.mod', tmp_path)
        shutil.copy2(testdata / 'nonmem' / 'models' / 'mx19B.csv', tmp_path)
        model_start = Model('mox2.mod')
        model_start.dataset_path = tmp_path / 'mx19B.csv'
        res = run_tool('modelsearch', 'exhaustive_stepwise', mfl, model=model_start)

        assert len(res.summary) == no_of_models
        assert len(res.models) == no_of_models
        assert all(int(model.modelfit_results.ofv) in range(-1500, -1400) for model in res.models)
        assert res.best_model.name == best_model_name
        rundir = tmp_path / 'modelsearch_dir1'
        assert rundir.is_dir()
        assert len(list((rundir / 'models').iterdir())) == no_of_models + 1


@pytest.mark.parametrize(
    'mfl, no_of_models, best_model_name, no_of_added_etas',
    [
        ('ABSORPTION(ZO)\nPERIPHERALS(1)', 4, 'modelsearch_candidate2', 3),
        ('ABSORPTION(ZO)\nPERIPHERALS([1, 2])', 7, 'modelsearch_candidate2', 5),
        ('LAGTIME()\nTRANSITS(1)', 2, 'modelsearch_candidate2', 1),
    ],
)
def test_exhaustive_stepwise_add_etas(
    tmp_path, testdata, mfl, no_of_models, best_model_name, no_of_added_etas
):
    with TemporaryDirectoryChanger(tmp_path):
        shutil.copy2(testdata / 'nonmem' / 'models' / 'mox2.mod', tmp_path)
        shutil.copy2(testdata / 'nonmem' / 'models' / 'mx19B.csv', tmp_path)
        model_start = Model('mox2.mod')
        model_start.dataset_path = tmp_path / 'mx19B.csv'
        res = run_tool('modelsearch', 'exhaustive_stepwise', mfl, add_etas=True, model=model_start)

        assert len(res.summary) == no_of_models
        assert len(res.models) == no_of_models
        assert res.best_model.name == best_model_name
        model_last = res.models[no_of_models - 1]
        assert (
            len(model_last.random_variables.etas) - len(model_start.random_variables.etas)
            == no_of_added_etas
        )
        rundir = tmp_path / 'modelsearch_dir1'
        assert rundir.is_dir()
        assert len(list((rundir / 'models').iterdir())) == no_of_models + 1


def test_exhaustive_stepwise_already_fit(tmp_path, testdata):
    with TemporaryDirectoryChanger(tmp_path):
        shutil.copy2(testdata / 'nonmem' / 'models' / 'mox2.mod', tmp_path)
        shutil.copy2(testdata / 'nonmem' / 'models' / 'mx19B.csv', tmp_path)
        model_start = Model('mox2.mod')
        model_start.dataset_path = tmp_path / 'mx19B.csv'

        fit(model_start)

        mfl = 'ABSORPTION(ZO)\nPERIPHERALS(1)'
        res = run_tool('modelsearch', 'exhaustive_stepwise', mfl, model=model_start)

        assert len(res.summary) == 4
        assert len(res.models) == 4
        assert all(int(model.modelfit_results.ofv) in range(-1500, -1400) for model in res.models)
        rundir = tmp_path / 'modelsearch_dir1'
        assert rundir.is_dir()
        assert len(list((rundir / 'models').iterdir())) == 4


def test_exhaustive_stepwise_start_model_fail(tmp_path, testdata):
    with TemporaryDirectoryChanger(tmp_path):
        shutil.copy2(testdata / 'nonmem' / 'models' / 'mox2.mod', tmp_path)
        shutil.copy2(testdata / 'nonmem' / 'models' / 'mx19B.csv', tmp_path)
        model_start = Model('mox2.mod')
        model_start.dataset_path = tmp_path / 'mx19B.csv'

        add_iiv(model_start, 'CL', 'incorrect_syntax')

        mfl = 'ABSORPTION(ZO)\nPERIPHERALS(1)'
        res = run_tool('modelsearch', 'exhaustive_stepwise', mfl, model=model_start)

        assert len(res.summary) == 4
        assert res.summary['dofv'].isnull().values.all()
        assert len(res.models) == 4
        assert all(model.modelfit_results is None for model in res.models)
        rundir = tmp_path / 'modelsearch_dir1'
        assert rundir.is_dir()
        assert len(list((rundir / 'models').iterdir())) == 5
