import shutil

import pytest
from pandas.testing import assert_frame_equal

from pharmpy.config import ConfigurationContext
from pharmpy.internals.fs.cwd import chdir
from pharmpy.model import Model
from pharmpy.tools import fit, read_modelfit_results, resume_tool, run_tool
from pharmpy.tools.external.nonmem import conf


def test_run_tool_ruvsearch_resume_flag(tmp_path, testdata):
    with chdir(tmp_path):
        for path in (testdata / 'nonmem').glob('pheno_real.*'):
            shutil.copy2(path, tmp_path)
        shutil.copy2(testdata / 'nonmem' / 'pheno.dta', tmp_path)
        shutil.copy2(testdata / 'nonmem' / 'sdtab1', tmp_path)
        # FIXME: temporary workaround so that read in parameter estimates use the Pharmpy name
        with ConfigurationContext(conf, parameter_names=['comment', 'basic']):
            model = Model.parse_model('pheno_real.mod')
            results = read_modelfit_results('pheno_real.mod')
            model = model.replace(datainfo=model.datainfo.replace(path=tmp_path / 'pheno.dta'))
            path = 'x'
            for i, resume in enumerate([False, False, True]):
                try:
                    res = run_tool(
                        'ruvsearch',
                        model=model,
                        results=results,
                        groups=4,
                        p_value=0.05,
                        skip=[],
                        path=path,
                        resume=resume,
                    )
                    if i != 0 and not resume:
                        assert False
                except FileExistsError as e:
                    if i == 0 or resume:
                        raise e

                if i == 0 or resume:
                    assert res


def test_run_tool_iivsearch_resume_flag(tmp_path, testdata, model_count):
    with chdir(tmp_path):
        shutil.copy2(testdata / 'nonmem' / 'models' / 'mox2.mod', tmp_path)
        shutil.copy2(testdata / 'nonmem' / 'models' / 'mox_simulated_normal.csv', tmp_path)
        # FIXME: temporary workaround so that read in parameter estimates use the Pharmpy name
        with ConfigurationContext(conf, parameter_names=['comment', 'basic']):
            model_start = Model.parse_model('mox2.mod')
            model_start = model_start.replace(
                datainfo=model_start.datainfo.replace(path=tmp_path / 'mox_simulated_normal.csv')
            )
            start_res = fit(model_start)
            # FIXME: Remove
            model_start = model_start.replace(modelfit_results=start_res)

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
                    # Candidate model directories + input_model directory + results.json + results.csv
                    assert len(list((rundir / 'models').iterdir())) == no_of_candidate_models + 3
                    assert model_count(rundir) == no_of_candidate_models
                    assert (rundir / 'metadata.json').exists()


@pytest.mark.parametrize(
    'search_space, no_of_models, last_model_parent_name',
    [
        ('ABSORPTION(ZO);PERIPHERALS(1)', 4, 'modelsearch_run2'),
    ],
)
def test_run_tool_modelsearch_resume_flag(
    tmp_path, testdata, model_count, search_space, no_of_models, last_model_parent_name
):
    shutil.copy2(testdata / 'nonmem' / 'models' / 'mox2.mod', tmp_path)
    shutil.copy2(testdata / 'nonmem' / 'models' / 'mox_simulated_normal.csv', tmp_path)
    # FIXME: temporary workaround so that read in parameter estimates use the Pharmpy name
    with chdir(tmp_path):
        with ConfigurationContext(conf, parameter_names=['comment', 'basic']):
            model_start = Model.parse_model('mox2.mod')
            model_start = model_start.replace(
                datainfo=model_start.datainfo.replace(path=tmp_path / 'mox_simulated_normal.csv')
            )

            start_res = fit(model_start)
            model_start = model_start.replace(modelfit_results=start_res)

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
                    # Candidate model directories + input_model directory + results.json + results.csv
                    assert len(list((rundir / 'models').iterdir())) == no_of_models + 3
                    assert model_count(rundir) == no_of_models
                    assert (rundir / 'results.json').exists()
                    assert (rundir / 'results.csv').exists()
                    assert (rundir / 'metadata.json').exists()


def test_resume_tool_ruvsearch(tmp_path, testdata):
    with chdir(tmp_path):
        for path in (testdata / 'nonmem').glob('pheno_real.*'):
            shutil.copy2(path, tmp_path)
        shutil.copy2(testdata / 'nonmem' / 'pheno.dta', tmp_path)
        shutil.copy2(testdata / 'nonmem' / 'sdtab1', tmp_path)

        model = Model.parse_model('pheno_real.mod')
        results = read_modelfit_results('pheno_real.mod')
        model = model.replace(datainfo=model.datainfo.replace(path=tmp_path / 'pheno.dta'))
        path = 'x'
        run_tool_res = run_tool(
            'ruvsearch',
            model=model,
            results=results,
            groups=4,
            p_value=0.05,
            skip=[],
            path=path,
        )
        assert run_tool_res

        resume_tool_res = resume_tool(path)
        assert resume_tool_res

        assert type(resume_tool_res) == type(run_tool_res)  # noqa: E721

        assert_frame_equal(resume_tool_res.cwres_models, run_tool_res.cwres_models)
        assert_frame_equal(resume_tool_res.summary_individuals, run_tool_res.summary_individuals)
        assert_frame_equal(
            resume_tool_res.summary_individuals_count, run_tool_res.summary_individuals_count
        )
        assert resume_tool_res.final_model_name == run_tool_res.final_model_name
        assert_frame_equal(resume_tool_res.summary_models, run_tool_res.summary_models)
        assert_frame_equal(resume_tool_res.summary_tool, run_tool_res.summary_tool)
        assert_frame_equal(resume_tool_res.summary_errors, run_tool_res.summary_errors)
        assert type(resume_tool_res.tool_database) == type(run_tool_res.tool_database)  # noqa: E721
        assert resume_tool_res.tool_database.to_dict() == run_tool_res.tool_database.to_dict()
