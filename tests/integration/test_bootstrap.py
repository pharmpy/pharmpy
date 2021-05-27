import shutil

from pharmpy import Model
from pharmpy.tools.bootstrap import Bootstrap
from pharmpy.utils import TemporaryDirectoryChanger


def test_bootstrap(tmp_path, testdata):
    with TemporaryDirectoryChanger(tmp_path):
        shutil.copy2(testdata / 'nonmem' / 'pheno.mod', tmp_path)
        shutil.copy2(testdata / 'nonmem' / 'pheno.dta', tmp_path)
        model = Model('pheno.mod')
        model.dataset_path = tmp_path / 'pheno.dta'
        models = Bootstrap(model, 3).run()
        assert len(models) == 3
        assert not all(mod.dataset.equals(model.dataset) for mod in models)
        ofvs = [mod.modelfit_results.ofv for mod in models]
        assert len(set(ofvs)) > 1
