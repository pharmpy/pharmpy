import shutil

import pytest

from pharmpy import Model
from pharmpy.modeling import fit, run_tool, set_first_order_absorption, set_first_order_elimination
from pharmpy.utils import TemporaryDirectoryChanger


@pytest.mark.parametrize(
    'mfl, no_of_models',
    [
        ('ABSORPTION(ZO)\nPERIPHERALS(1)', 4),
        ('ABSORPTION(ZO)\nTRANSITS(1)', 3),
        ('ABSORPTION([ZO,SEQ-ZO-FO])\nPERIPHERALS(1)', 7),
    ],
)
def test_exhaustive_stepwise(tmp_path, testdata, mfl, no_of_models):
    with TemporaryDirectoryChanger(tmp_path):
        shutil.copy2(testdata / 'nonmem' / 'models' / 'mox2.mod', tmp_path)
        shutil.copy2(testdata / 'nonmem' / 'models' / 'mx19B_ZO.csv', tmp_path)
        model = Model('mox2.mod')
        model.dataset_path = tmp_path / 'mx19B_ZO.csv'

        set_first_order_absorption(model)
        set_first_order_elimination(model)
        model.update_source(nofiles=True)

        res = run_tool('modelsearch', model, 'exhaustive_stepwise', mfl, rankfunc='ofv', cutoff=0)

        assert len(res.summary) == no_of_models
        assert len(res.models) == no_of_models
        assert all(int(model.modelfit_results.ofv) in range(-1443, -1430) for model in res.models)


def test_exhaustive_stepwise_already_fit(tmp_path, testdata):
    with TemporaryDirectoryChanger(tmp_path):
        shutil.copy2(testdata / 'nonmem' / 'models' / 'mox2.mod', tmp_path)
        shutil.copy2(testdata / 'nonmem' / 'models' / 'mx19B_ZO.csv', tmp_path)
        model = Model('mox2.mod')
        model.dataset_path = tmp_path / 'mx19B_ZO.csv'

        set_first_order_absorption(model)
        set_first_order_elimination(model)
        model.update_source(nofiles=True)
        fit(model)

        mfl = 'ABSORPTION(ZO)\nPERIPHERALS(1)'
        res = run_tool('modelsearch', model, 'exhaustive_stepwise', mfl, rankfunc='ofv', cutoff=0)

        assert len(res.summary) == 4
        assert len(res.models) == 4
        assert all(int(model.modelfit_results.ofv) in range(-1443, -1430) for model in res.models)
