import shutil

import pytest

from pharmpy import Model
from pharmpy.modeling import set_first_order_absorption, set_first_order_elimination
from pharmpy.tools.modelsearch import ModelSearch
from pharmpy.utils import TemporaryDirectoryChanger


@pytest.mark.parametrize(
    'mfl, no_of_models, ofv_range',
    [
        ('ABSORPTION(ZO)\nPERIPHERALS(1)', 4, range(-1443, -1430)),
        ('ABSORPTION(ZO)\nTRANSITS(1)', 3, range(-1443, -1430)),
    ],
)
def test_exhaustive_stepwise(tmp_path, testdata, mfl, no_of_models, ofv_range):
    with TemporaryDirectoryChanger(tmp_path):
        shutil.copy2(testdata / 'nonmem' / 'models' / 'mox2.mod', tmp_path)
        shutil.copy2(testdata / 'nonmem' / 'models' / 'mx19B_ZO.csv', tmp_path)
        model = Model('mox2.mod')
        model.dataset_path = tmp_path / 'mx19B_ZO.csv'

        set_first_order_absorption(model)
        set_first_order_elimination(model)
        model.update_source(nofiles=True)

        res = ModelSearch(model, 'exhaustive_stepwise', mfl, rankfunc='ofv', cutoff=0).run()

        assert len(res.summary) == no_of_models
        assert len(res.models) == no_of_models
        assert all(int(model.modelfit_results.ofv) in ofv_range for model in res.models)
