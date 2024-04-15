import os
import shutil
from pathlib import Path

import pytest

from pharmpy.internals.fs.cwd import chdir
from pharmpy.tools import run_amd
from pharmpy.workflows import LocalDirectoryContext


def _model_count(rundir: Path):
    return sum(
        map(
            lambda path: 0 if path.name in ['.lock', '.datasets'] else 1,
            ((rundir / 'subcontexts' / 'models').iterdir()),
        )
    )


@pytest.mark.parametrize(
    'strategy, subrundir',
    [
        (
            'default',
            [
                'modelsearch',
                'iivsearch',
                'ruvsearch',
                'iovsearch',
                'allometry',
                'covsearch_exploratory',
                'covsearch_mechanistic',  # FIXME: Theses two are currently created as empty
                'covsearch_structural',
                'simulation',
            ],
        ),
        (
            'reevaluation',
            [
                'modelsearch',
                'iivsearch',
                'ruvsearch',
                'iovsearch',
                'allometry',
                'covsearch_exploratory',
                'covsearch_mechanistic',  # FIXME: Theses two are currently created as empty
                'covsearch_structural',
                'rerun_iivsearch',
                'rerun_ruvsearch',
                'simulation',
            ],
        ),
    ],
)
@pytest.mark.filterwarnings(
    'ignore:.*Adjusting initial estimates to create positive semidefinite omega/sigma matrices.',
    'ignore::UserWarning',
)
def test_amd_basic(tmp_path, testdata, strategy, subrundir):
    with chdir(tmp_path):
        shutil.copy2(testdata / 'nonmem' / 'models' / 'moxo_simulated_amd.csv', '.')
        shutil.copy2(testdata / 'nonmem' / 'models' / 'moxo_simulated_amd.datainfo', '.')
        input = 'moxo_simulated_amd.csv'
        res = run_amd(
            input,
            modeltype='basic_pk',
            administration='oral',
            search_space='PERIPHERALS(1)',
            strategy=strategy,
            occasion='VISI',
            strictness='minimization_successful or rounding_errors',
            retries_strategy='skip',
            cl_init=0.01,
            vc_init=1.0,
            mat_init=0.1,
        )

        rundir = tmp_path / 'amd1'
        assert rundir.is_dir()
        assert (rundir / 'results.json').exists()
        assert (rundir / 'results.csv').exists()

        ctx = LocalDirectoryContext("amd1")
        subnames = ctx.list_all_subcontexts()
        assert set(subnames) == set(subrundir)

        assert len(res.summary_tool) >= 1
        assert len(res.summary_models) >= 1
        assert len(res.summary_individuals_count) >= 1


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
            search_space='PERIPHERALS(1)',
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
        assert len(res.summary_individuals_count) >= 1
