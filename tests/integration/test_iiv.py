from pathlib import Path

import numpy as np
import pytest

from pharmpy.modeling import create_joint_distribution, fit, run_tool, set_seq_zo_fo_absorption
from pharmpy.utils import TemporaryDirectoryChanger


def _model_count(rundir: Path):
    return sum(
        map(
            lambda path: 0 if path.name in ['.lock', '.datasets'] else 1,
            ((rundir / 'models').iterdir()),
        )
    )


def test_iiv_block_structure(tmp_path, start_model):
    with TemporaryDirectoryChanger(tmp_path):
        res = run_tool('iiv', 'brute_force_block_structure', model=start_model)

        no_of_candidate_models = 4
        assert len(res.summary_tool) == no_of_candidate_models + 1
        assert len(res.summary_models) == no_of_candidate_models + 1
        assert len(res.models) == no_of_candidate_models
        assert all(
            model.modelfit_results and not np.isnan(model.modelfit_results.ofv)
            for model in res.models
        )
        assert all(model.random_variables != start_model.random_variables for model in res.models)
        rundir = tmp_path / 'iiv_dir1'
        assert rundir.is_dir()
        assert _model_count(rundir) == no_of_candidate_models
        assert (rundir / 'metadata.json').exists()


def test_iiv_no_of_etas(tmp_path, start_model):
    with TemporaryDirectoryChanger(tmp_path):
        res = run_tool('iiv', 'brute_force_no_of_etas', model=start_model)

        no_of_candidate_models = 7
        assert len(res.summary_tool) == no_of_candidate_models + 1
        assert len(res.summary_models) == no_of_candidate_models + 1
        assert len(res.models) == no_of_candidate_models
        assert res.best_model.name == 'iiv_no_of_etas_candidate3'
        rundir = tmp_path / 'iiv_dir1'
        assert rundir.is_dir()
        assert _model_count(rundir) == no_of_candidate_models
        assert (rundir / 'metadata.json').exists()


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize(
    'iiv_strategy, best_model_name',
    [(1, 'iiv_no_of_etas_candidate4'), (2, 'iiv_no_of_etas_candidate4')],
)
def test_iiv_no_of_etas_added_iiv(tmp_path, start_model, iiv_strategy, best_model_name):
    with TemporaryDirectoryChanger(tmp_path):
        start_model = start_model.copy()
        start_model.name = 'start_model_copy'
        start_model.modelfit_results = None

        set_seq_zo_fo_absorption(start_model)
        fit(start_model)

        res = run_tool(
            'iiv',
            'brute_force_no_of_etas',
            iiv_strategy=iiv_strategy,
            rankfunc='bic',
            model=start_model,
        )

        if iiv_strategy == 2:
            assert len(res.start_model.random_variables['ETA(1)'].joint_names) > 0

        no_of_candidate_models = 15
        assert len(res.summary_tool) == no_of_candidate_models + 1
        assert len(res.summary_models) == no_of_candidate_models + 1
        assert len(res.models) == no_of_candidate_models
        assert res.best_model.name == best_model_name
        rundir = tmp_path / 'iiv_dir1'
        assert rundir.is_dir()
        assert _model_count(rundir) == no_of_candidate_models + 1
        assert (rundir / 'metadata.json').exists()


def test_iiv_no_of_etas_fullblock(tmp_path, start_model):
    with TemporaryDirectoryChanger(tmp_path):
        start_model = start_model.copy()
        start_model.name = 'start_model_copy'
        start_model.modelfit_results = None

        create_joint_distribution(start_model)
        fit(start_model)

        res = run_tool('iiv', 'brute_force_no_of_etas', model=start_model)

        no_of_candidate_models = 7
        assert len(res.summary_tool) == no_of_candidate_models + 1
        assert len(res.summary_models) == no_of_candidate_models + 1
        assert len(res.models) == no_of_candidate_models
        assert res.best_model.name == 'iiv_no_of_etas_candidate3'
        rundir = tmp_path / 'iiv_dir1'
        assert rundir.is_dir()
        assert _model_count(rundir) == no_of_candidate_models
        assert (rundir / 'metadata.json').exists()
