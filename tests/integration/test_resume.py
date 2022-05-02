import shutil

import pytest

from pharmpy import Model
from pharmpy.modeling import run_tool, set_seq_zo_fo_absorption
from pharmpy.utils import TemporaryDirectoryChanger


def test_run_tool_resmod_resume_flag(tmp_path, testdata):
    with TemporaryDirectoryChanger(tmp_path):
        for path in (testdata / 'nonmem').glob('pheno_real.*'):
            shutil.copy2(path, tmp_path)
        shutil.copy2(testdata / 'nonmem' / 'pheno.dta', tmp_path)
        shutil.copy2(testdata / 'nonmem' / 'sdtab1', tmp_path)

        model = Model.create_model('pheno_real.mod')
        model.datainfo.path = tmp_path / 'pheno.dta'
        path = 'x'
        for i, resume in enumerate([False, False, True]):
            try:
                res = run_tool(
                    'resmod', model, groups=4, p_value=0.05, skip=[], path=path, resume=resume
                )
                if i != 0 and not resume:
                    assert False
            except FileExistsError as e:
                if i == 0 or resume:
                    raise e

            if i == 0 or resume:
                assert res


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize(
    'iiv_strategy, best_model_name',
    [(1, 'iiv_no_of_etas_candidate4'), (2, 'iiv_no_of_etas_candidate4')],
)
def test_run_tool_iiv_resume_flag(tmp_path, testdata, iiv_strategy, best_model_name):
    with TemporaryDirectoryChanger(tmp_path):
        shutil.copy2(testdata / 'nonmem' / 'models' / 'mox2.mod', tmp_path)
        shutil.copy2(testdata / 'nonmem' / 'models' / 'mox_simulated_normal.csv', tmp_path)
        model_start = Model.create_model('mox2.mod')
        model_start.datainfo.path = tmp_path / 'mox_simulated_normal.csv'

        path = 'x'

        set_seq_zo_fo_absorption(model_start)

        for i, resume in enumerate([False, False, True]):
            try:
                res = run_tool(
                    'iiv',
                    'brute_force_no_of_etas',
                    iiv_strategy=iiv_strategy,
                    rankfunc='bic',
                    model=model_start,
                    path=path,
                    resume=resume,
                )
                if i != 0 and not resume:
                    assert False
            except FileExistsError as e:
                if i == 0 or resume:
                    raise e

            if i == 0 or resume:

                if iiv_strategy == 2:
                    assert len(res.start_model.random_variables['ETA(1)'].joint_names) > 0
                assert len(res.summary_tool) == 16
                assert len(res.summary_models) == 16
                assert len(res.models) == 15
                assert res.best_model.name == best_model_name
                rundir = tmp_path / path
                assert rundir.is_dir()
                assert len(list((rundir / 'models').iterdir())) == 17
                assert (rundir / 'metadata.json').exists()


@pytest.mark.parametrize(
    'mfl, no_of_models, best_model_name, last_model_parent_name',
    [
        ('ABSORPTION(ZO);PERIPHERALS(1)', 4, 'modelsearch_candidate2', 'modelsearch_candidate2'),
        ('ABSORPTION(ZO);TRANSITS(1)', 2, 'modelsearch_candidate2', 'mox2'),
        (
            'ABSORPTION([ZO,SEQ-ZO-FO]);PERIPHERALS(1)',
            7,
            'modelsearch_candidate6',
            'modelsearch_candidate3',
        ),
        ('LAGTIME();TRANSITS(1)', 2, 'modelsearch_candidate2', 'mox2'),
        ('ABSORPTION(ZO);TRANSITS(3, *)', 3, 'modelsearch_candidate2', 'mox2'),
    ],
)
def test_run_tool_modelsearch_resume_flag(
    tmp_path, testdata, mfl, no_of_models, best_model_name, last_model_parent_name
):
    with TemporaryDirectoryChanger(tmp_path):
        shutil.copy2(testdata / 'nonmem' / 'models' / 'mox2.mod', tmp_path)
        shutil.copy2(testdata / 'nonmem' / 'models' / 'mox_simulated_normal.csv', tmp_path)
        model_start = Model.create_model('mox2.mod')
        model_start.datainfo.path = tmp_path / 'mox_simulated_normal.csv'
        path = 'x'
        for i, resume in enumerate([False, False, True]):
            try:
                res = run_tool(
                    'modelsearch',
                    mfl,
                    'exhaustive_stepwise',
                    model=model_start,
                    path=path,
                    resume=resume,
                )
                if i != 0 and not resume:
                    assert False
            except FileExistsError as e:
                if i == 0 or resume:
                    raise e

            if i == 0 or resume:
                assert len(res.summary_tool) == no_of_models + 1
                assert len(res.summary_models) == no_of_models + 1
                assert len(res.models) == no_of_models
                assert res.best_model.name == best_model_name

                assert res.models[0].parent_model == 'mox2'
                assert res.models[-1].parent_model == last_model_parent_name
                if last_model_parent_name != 'mox2':
                    last_model_features = res.summary_tool.loc[res.models[-1].name]['features']
                    parent_model_features = res.summary_tool.loc[last_model_parent_name]['features']
                    assert (
                        last_model_features[: len(parent_model_features)] == parent_model_features
                    )

                rundir = tmp_path / path
                assert rundir.is_dir()
                assert len(list((rundir / 'models').iterdir())) == no_of_models + 2
                assert (rundir / 'results.json').exists()
                assert (rundir / 'results.csv').exists()
                assert (rundir / 'metadata.json').exists()
