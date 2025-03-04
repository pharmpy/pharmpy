import shutil

import pytest

from pharmpy.internals.fs.cwd import chdir
from pharmpy.model import Model
from pharmpy.tools import fit, read_modelfit_results, run_tool
from pharmpy.workflows import LocalDirectoryContext


def test_run_tool_ruvsearch_resume_flag(tmp_path, testdata):
    with chdir(tmp_path):
        for path in (testdata / 'nonmem' / 'ruvsearch').glob('mox3.*'):
            shutil.copy2(path, tmp_path)
        shutil.copy2(testdata / 'nonmem' / 'ruvsearch' / 'moxo_simulated_resmod.csv', tmp_path)
        shutil.copy2(testdata / 'nonmem' / 'ruvsearch' / 'mytab', tmp_path)

        model = Model.parse_model('mox3.mod')
        results = read_modelfit_results('mox3.mod')
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
                # if i != 0 and not resume:
                #    assert False
            except FileExistsError as e:
                if i == 0 or resume:
                    raise e

            if i == 0 or resume:
                assert res


def test_run_tool_iivsearch_resume_flag(tmp_path, testdata, model_count):
    with chdir(tmp_path):
        shutil.copy2(testdata / 'nonmem' / 'models' / 'mox2.mod', tmp_path)
        shutil.copy2(testdata / 'nonmem' / 'models' / 'mox_simulated_normal.csv', tmp_path)
        model_start = Model.parse_model('mox2.mod')
        model_start = model_start.replace(
            datainfo=model_start.datainfo.replace(path=tmp_path / 'mox_simulated_normal.csv')
        )
        start_res = fit(model_start)

        path = 'x'
        for i, resume in enumerate([False, False, True]):
            try:
                res = run_tool(
                    'iivsearch',
                    model=model_start,
                    results=start_res,
                    algorithm='top_down_exhaustive',
                    correlation_algorithm='skip',
                    rank_type='mbic',
                    E_p=1.0,
                    keep=[],
                    path=path,
                    resume=resume,
                )
                # if i != 0 and not resume:
                #    assert False
            except FileExistsError as e:
                if i == 0 or resume:
                    raise e

            if i == 0 or resume:
                no_of_candidate_models = 7
                assert len(res.summary_tool) == no_of_candidate_models + 3
                assert len(res.summary_models) == no_of_candidate_models + 1

                ctx = LocalDirectoryContext(path)
                names = ctx.list_all_names()
                res_models = [
                    ctx.retrieve_model_entry(name).model
                    for name in names
                    if name not in ['input', 'final']
                ]
                assert len(res_models) == no_of_candidate_models

                rundir = tmp_path / path
                assert rundir.is_dir()
                assert len(list((rundir / 'models').iterdir())) == no_of_candidate_models + 2
                assert model_count(rundir) == no_of_candidate_models + 2
                assert (rundir / 'metadata.json').exists()


@pytest.mark.parametrize(
    'search_space, no_of_models, last_model_parent_name',
    [
        ('ABSORPTION([FO,ZO]);PERIPHERALS([0,1])', 4, 'modelsearch_run2'),
    ],
)
def test_run_tool_modelsearch_resume_flag(
    tmp_path, testdata, model_count, search_space, no_of_models, last_model_parent_name
):
    shutil.copy2(testdata / 'nonmem' / 'models' / 'mox2.mod', tmp_path)
    shutil.copy2(testdata / 'nonmem' / 'models' / 'mox_simulated_normal.csv', tmp_path)
    # FIXME: Temporary workaround so that read in parameter estimates use the Pharmpy name
    with chdir(tmp_path):
        model_start = Model.parse_model('mox2.mod')
        model_start = model_start.replace(
            datainfo=model_start.datainfo.replace(path=tmp_path / 'mox_simulated_normal.csv')
        )

        start_res = fit(model_start)

        path = 'x'
        for i, resume in enumerate([False, False, True]):
            try:
                res = run_tool(
                    'modelsearch',
                    model=model_start,
                    results=start_res,
                    search_space=search_space,
                    algorithm='exhaustive_stepwise',
                    path=path,
                    resume=resume,
                )
                # if i != 0 and not resume:
                #    assert False
            except FileExistsError as e:
                if i == 0 or resume:
                    raise e

            if i == 0 or resume:
                assert len(res.summary_tool) == no_of_models + 1
                assert len(res.summary_models) == no_of_models + 1
                assert len(res.models) == no_of_models + 1

                rundir = tmp_path / path
                assert rundir.is_dir()
                assert len(list((rundir / 'models').iterdir())) == no_of_models + 2
                assert model_count(rundir) == no_of_models + 2
                assert (rundir / 'results.json').exists()
                assert (rundir / 'results.csv').exists()
                assert (rundir / 'metadata.json').exists()
