import os
import shutil
from pathlib import Path

import pytest

from pharmpy.internals.fs.cwd import chdir
from pharmpy.tools import run_amd
from pharmpy.tools.context import open_context


def _model_count(rundir: Path):
    return sum(
        map(
            lambda path: 0 if path.name in ['.lock', '.datasets'] else 1,
            ((rundir / 'subcontexts' / 'models').iterdir()),
        )
    )


@pytest.mark.parametrize(
    'model_kwargs, run_kwargs, search_space, subtools',
    [
        (
            {
                'modeltype': 'basic_pk',
                'administration': 'iv',
                'cl_init': 1,
                'vc_init': 10,
                'occasion': 'VISI',
            },
            {'strategy': 'default', 'retries_strategy': 'skip'},
            'ABSORPTION([FO,ZO]);PERIPHERALS(0..2)',
            {
                'modelfit',
                'modelsearch',
                'iivsearch',
                'ruvsearch',
                'iovsearch',
                'allometry',
                'covsearch_exploratory',
                'simulation',
            },
        ),
        (
            {
                'modeltype': 'basic_pk',
                'administration': 'iv',
                'cl_init': 1,
                'vc_init': 10,
                'occasion': 'VISI',
            },
            {'strategy': 'reevaluation', 'retries_strategy': 'skip'},
            'ABSORPTION([FO,ZO]);PERIPHERALS(0..2)',
            {
                'modelfit',
                'modelsearch',
                'iivsearch',
                'ruvsearch',
                'iovsearch',
                'allometry',
                'covsearch_exploratory',
                'rerun_iivsearch',
                'rerun_ruvsearch',
                'simulation',
            },
        ),
    ],
)
def test_amd_dummy(tmp_path, testdata, model_kwargs, run_kwargs, search_space, subtools):
    with chdir(tmp_path):
        shutil.copy2(testdata / 'nonmem' / 'models' / 'moxo_simulated_amd.csv', '.')
        shutil.copy2(testdata / 'nonmem' / 'models' / 'moxo_simulated_amd.datainfo', '.')
        input = 'moxo_simulated_amd.csv'
        res = run_amd(input, **model_kwargs, **run_kwargs, esttool='dummy')

        assert (
            len(res.summary_tool) == len(subtools) - 1
        )  # Simulation is not part of the result table
        assert len(res.summary_models) > len(subtools)

        rundir = tmp_path / 'amd1'
        assert rundir.is_dir()
        assert (rundir / 'results.json').exists()
        assert (rundir / 'results.csv').exists()

        ctx = open_context("amd1")
        subnames = ctx.list_all_subcontexts()
        assert set(subnames) == subtools


# def test_structure_mechanistic_exploratory(tmp_path, testdata):
#     with chdir(tmp_path):
#         shutil.copy2(testdata / 'nonmem' / 'models' / 'moxo_simulated_amd.csv', '.')
#         shutil.copy2(testdata / 'nonmem' / 'models' / 'moxo_simulated_amd.datainfo', '.')
#         input = 'moxo_simulated_amd.csv'
#         res = run_amd(
#             input,
#             modeltype='basic_pk',
#             administration='oral',
#             search_space='PERIPHERALS(1);COVARIATE(CL,WT,pow);COVARIATE?(VC,AGE,exp);COVARIATE?(Q,SEX,cat)',
#             mechanistic_covariates=["AGE"],
#             occasion='VISI',
#             strictness='minimization_successful or rounding_errors',
#             retries_strategy='skip',
#         )

#         rundir = tmp_path / 'amd_dir1'
#         assert rundir.is_dir()
#         assert (rundir / 'results.json').exists()
#         assert (rundir / 'results.csv').exists()
#         subrundir = [
#             'modelfit',
#             'modelsearch',
#             'iivsearch',
#             'ruvsearch',
#             'iovsearch',
#             'allometry',
#             'covsearch_structural',
#             'covsearch_mechanistic',
#             'covsearch_exploratory',
#         ]
#         for dir in subrundir:
#             dir = rundir / dir
#             assert _model_count(dir) >= 1

#         assert len(res.summary_tool) == len(subrundir) - 1  # Mechanistic/Exploratory grouped as one
#         assert len(res.summary_models) >= 1
#         assert len(res.summary_individuals_count) >= 1


@pytest.mark.filterwarnings('ignore::UserWarning')
def test_amd_dollar_design(tmp_path, testdata):
    if os.name == 'nt':
        pytest.skip("TODO Fails on GHA but not locally, temporarily skipping.")

    with chdir(tmp_path):
        shutil.copy2(testdata / 'nonmem' / 'models' / 'moxo_simulated_amd.csv', '.')
        shutil.copy2(testdata / 'nonmem' / 'models' / 'moxo_simulated_amd.datainfo', '.')
        input = 'moxo_simulated_amd.csv'
        res = run_amd(
            input,
            modeltype='basic_pk',
            administration='oral',
            search_space='ABSORPTION(FO);PERIPHERALS(1)',
            strategy='default',
            occasion='VISI',
            strictness='minimization_successful or rounding_errors',
            retries_strategy='skip',
            parameter_uncertainty_method='EFIM',
            cl_init=0.01,
            vc_init=1.0,
            mat_init=0.1,
        )

        rundir = tmp_path / 'amd1'
        assert rundir.is_dir()
        assert (rundir / 'results.json').exists()
        assert (rundir / 'results.csv').exists()

        assert len(res.summary_models) >= 1
