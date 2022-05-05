import shutil

import pytest

from pharmpy import Model
from pharmpy.config import ConfigurationContext
from pharmpy.modeling import fit, run_tool
from pharmpy.plugins.nonmem import conf
from pharmpy.utils import TemporaryDirectoryChanger


def test_run_tool_resmod_resume_flag(tmp_path, testdata):
    with TemporaryDirectoryChanger(tmp_path):
        for path in (testdata / 'nonmem').glob('pheno_real.*'):
            shutil.copy2(path, tmp_path)
        shutil.copy2(testdata / 'nonmem' / 'pheno.dta', tmp_path)
        shutil.copy2(testdata / 'nonmem' / 'sdtab1', tmp_path)
        # FIXME: temporary workaround so that read in parameter estimates use the Pharmpy name
        with ConfigurationContext(conf, parameter_names=['comment', 'basic']):
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


def test_run_tool_iivsearch_resume_flag(tmp_path, testdata):
    with TemporaryDirectoryChanger(tmp_path):
        shutil.copy2(testdata / 'nonmem' / 'models' / 'mox2.mod', tmp_path)
        shutil.copy2(testdata / 'nonmem' / 'models' / 'mox_simulated_normal.csv', tmp_path)
        # FIXME: temporary workaround so that read in parameter estimates use the Pharmpy name
        with ConfigurationContext(conf, parameter_names=['comment', 'basic']):
            model_start = Model.create_model('mox2.mod')
            model_start.datainfo.path = tmp_path / 'mox_simulated_normal.csv'
            fit(model_start)

            path = 'x'
            for i, resume in enumerate([False, False, True]):
                try:
                    res = run_tool(
                        'iivsearch',
                        'brute_force_no_of_etas',
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
                    no_of_candidate_models = 7
                    assert len(res.summary_tool) == no_of_candidate_models + 1
                    assert len(res.summary_models) == no_of_candidate_models + 1
                    assert len(res.models) == no_of_candidate_models

                    assert res.models[-1].modelfit_results

                    rundir = tmp_path / path
                    assert rundir.is_dir()
                    assert len(list((rundir / 'models').iterdir())) == 9
                    assert (rundir / 'metadata.json').exists()


@pytest.mark.parametrize(
    'search_space, no_of_models, last_model_parent_name',
    [
        ('ABSORPTION(ZO);PERIPHERALS(1)', 4, 'modelsearch_candidate2'),
    ],
)
def test_run_tool_modelsearch_resume_flag(
    tmp_path, testdata, search_space, no_of_models, last_model_parent_name
):
    shutil.copy2(testdata / 'nonmem' / 'models' / 'mox2.mod', tmp_path)
    shutil.copy2(testdata / 'nonmem' / 'models' / 'mox_simulated_normal.csv', tmp_path)
    # FIXME: temporary workaround so that read in parameter estimates use the Pharmpy name
    with TemporaryDirectoryChanger(tmp_path):
        with ConfigurationContext(conf, parameter_names=['comment', 'basic']):
            model_start = Model.create_model('mox2.mod')
            model_start.datainfo.path = tmp_path / 'mox_simulated_normal.csv'

            fit(model_start)

            path = 'x'
            for i, resume in enumerate([False, False, True]):
                try:
                    res = run_tool(
                        'modelsearch',
                        search_space,
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
                    assert res.models[-1].modelfit_results

                    assert res.models[0].parent_model == 'mox2'
                    assert res.models[-1].parent_model == last_model_parent_name
                    if last_model_parent_name != 'mox2':
                        last_model_features = res.summary_tool.loc[res.models[-1].name][
                            'description'
                        ]
                        parent_model_features = res.summary_tool.loc[last_model_parent_name][
                            'description'
                        ]
                        assert (
                            last_model_features[: len(parent_model_features)]
                            == parent_model_features
                        )

                    rundir = tmp_path / path
                    assert rundir.is_dir()
                    assert len(list((rundir / 'models').iterdir())) == no_of_models + 2
                    assert (rundir / 'results.json').exists()
                    assert (rundir / 'results.csv').exists()
                    assert (rundir / 'metadata.json').exists()
