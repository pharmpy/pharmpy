from pathlib import Path

import numpy as np
import pytest

from pharmpy.modeling import fit, run_iivsearch, set_seq_zo_fo_absorption
from pharmpy.utils import TemporaryDirectoryChanger


def _model_count(rundir: Path):
    return sum(
        map(
            lambda path: 0 if path.name in ['.lock', '.datasets'] else 1,
            ((rundir / 'models').iterdir()),
        )
    )


def test_block_structure(tmp_path, start_model):
    with TemporaryDirectoryChanger(tmp_path):
        res = run_iivsearch('brute_force_block_structure', model=start_model)

        no_of_candidate_models = 4
        assert len(res.summary_tool) == no_of_candidate_models + 1
        assert len(res.summary_models) == no_of_candidate_models + 1
        assert len(res.models) == no_of_candidate_models

        assert all(
            model.modelfit_results and not np.isnan(model.modelfit_results.ofv)
            for model in res.models
        )
        assert all(model.random_variables != start_model.random_variables for model in res.models)

        assert res.summary_tool.loc['mox2']['description'] == '[CL]+[VC]+[MAT]'
        assert not res.input_model.random_variables['ETA(1)'].joint_names
        assert (
            res.summary_tool.loc['iivsearch_block_structure_candidate1']['description']
            == '[CL,VC,MAT]'
        )
        assert len(res.models[0].random_variables['ETA(1)'].joint_names) == 3

        rundir = tmp_path / 'iivsearch_dir1'
        assert rundir.is_dir()
        assert _model_count(rundir) == no_of_candidate_models
        assert (rundir / 'metadata.json').exists()


def test_no_of_etas(tmp_path, start_model):
    with TemporaryDirectoryChanger(tmp_path):
        res = run_iivsearch('brute_force_no_of_etas', model=start_model)

        no_of_candidate_models = 7
        assert len(res.summary_tool) == no_of_candidate_models + 1
        assert len(res.summary_models) == no_of_candidate_models + 1
        assert len(res.models) == no_of_candidate_models

        assert res.models[-1].modelfit_results

        assert res.summary_tool.loc['mox2']['description'] == '[CL,VC,MAT]'
        assert res.input_model.random_variables.iiv.names == ['ETA(1)', 'ETA(2)', 'ETA(3)']
        assert res.summary_tool.iloc[-1]['description'] == '[]'
        assert res.models[0].random_variables.iiv.names == ['ETA(2)', 'ETA(3)']

        rundir = tmp_path / 'iivsearch_dir1'
        assert rundir.is_dir()
        assert _model_count(rundir) == no_of_candidate_models
        assert (rundir / 'metadata.json').exists()


def test_brute_force(tmp_path, start_model):
    with TemporaryDirectoryChanger(tmp_path):
        res = run_iivsearch('brute_force', model=start_model)

        no_of_candidate_models = 8
        assert len(res.summary_tool) == no_of_candidate_models + 2
        assert len(res.summary_models) == no_of_candidate_models + 2
        assert len(res.models) == no_of_candidate_models

        assert 'iivsearch_no_of_etas_candidate3' in res.summary_errors.index.get_level_values(
            'model'
        )

        assert all(
            model.modelfit_results and not np.isnan(model.modelfit_results.ofv)
            for model in res.models
        )
        assert all(model.random_variables != start_model.random_variables for model in res.models)

        rundir = tmp_path / 'iivsearch_dir1'
        assert rundir.is_dir()
        assert _model_count(rundir) == no_of_candidate_models
        assert (rundir / 'metadata.json').exists()


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize(
    'iiv_strategy',
    ['diagonal', 'fullblock'],
)
def test_no_of_etas_iiv_strategies(tmp_path, start_model, iiv_strategy):
    with TemporaryDirectoryChanger(tmp_path):
        start_model = start_model.copy()
        start_model.name = 'moxo2_copy'
        start_model.modelfit_results = None

        set_seq_zo_fo_absorption(start_model)
        fit(start_model)

        res = run_iivsearch(
            'brute_force_no_of_etas',
            iiv_strategy=iiv_strategy,
            rankfunc='bic',
            model=start_model,
        )

        if iiv_strategy == 'fullblock':
            base_model = [model for model in res.models if model.name == 'base_model'].pop()
            base_rvs = base_model.random_variables.iiv
            assert len(base_rvs['ETA(1)'].joint_names) == len(base_rvs)

        no_of_candidate_models = 15
        assert len(res.summary_tool) == no_of_candidate_models + 1
        assert len(res.summary_models) == no_of_candidate_models + 1
        assert len(res.models) == no_of_candidate_models + 1
        assert res.models[-1].modelfit_results
        rundir = tmp_path / 'iivsearch_dir1'
        assert rundir.is_dir()
        assert _model_count(rundir) == no_of_candidate_models + 1
        assert (rundir / 'metadata.json').exists()
