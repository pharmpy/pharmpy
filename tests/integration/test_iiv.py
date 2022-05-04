from pathlib import Path

import numpy as np
import pytest

from pharmpy.modeling import fit, run_tool, set_seq_zo_fo_absorption
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

        assert res.summary_tool.loc['mox2']['features'] == '[CL]+[VC]+[MAT]'
        assert not res.start_model.random_variables['ETA(1)'].joint_names
        assert res.summary_tool.loc['iiv_block_structure_candidate1']['features'] == '[CL,VC,MAT]'
        assert len(res.models[0].random_variables['ETA(1)'].joint_names) == 3

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

        assert res.models[-1].modelfit_results

        assert res.summary_tool.loc['mox2']['features'] == '[CL,VC,MAT]'
        assert res.start_model.random_variables.iiv.names == ['ETA(1)', 'ETA(2)', 'ETA(3)']
        assert res.summary_tool.iloc[-1]['features'] == '[]'
        assert res.models[0].random_variables.iiv.names == ['ETA(2)', 'ETA(3)']

        rundir = tmp_path / 'iiv_dir1'
        assert rundir.is_dir()
        assert _model_count(rundir) == no_of_candidate_models
        assert (rundir / 'metadata.json').exists()


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize(
    'iiv_strategy',
    [1, 2],
)
def test_iiv_no_of_etas_added_iiv(tmp_path, start_model, iiv_strategy):
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
        assert res.models[-1].modelfit_results
        rundir = tmp_path / 'iiv_dir1'
        assert rundir.is_dir()
        assert _model_count(rundir) == no_of_candidate_models + 1
        assert (rundir / 'metadata.json').exists()
