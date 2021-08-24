import shutil

from pharmpy import Model
from pharmpy.modeling import set_first_order_absorption, set_first_order_elimination
from pharmpy.tools.modelsearch import ModelSearch
from pharmpy.utils import TemporaryDirectoryChanger


def test_exhaustive_stepwise(tmp_path, testdata):
    with TemporaryDirectoryChanger(tmp_path):
        shutil.copy2(testdata / 'nonmem' / 'models' / 'mox2.mod', tmp_path)
        shutil.copy2(testdata / 'nonmem' / 'models' / 'mx19B_ZO.csv', tmp_path)
        model = Model('mox2.mod')
        model.dataset_path = tmp_path / 'mx19B_ZO.csv'

        set_first_order_absorption(model)
        set_first_order_elimination(model)
        model.update_source(force=True)

        res = ModelSearch(model, 'exhaustive_stepwise', 'ABSORPTION(ZO)\nPERIPHERALS(1)').run()
        assert res
