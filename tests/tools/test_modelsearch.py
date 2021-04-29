from pyfakefs.fake_filesystem_unittest import Patcher

import pharmpy.tools.modelsearch as ms
from pharmpy import Model
from pharmpy.plugins.nonmem import conf


def test_modelsearch(datadir):
    assert conf  # Make linter happy. Don't know why conf needs to be imported
    with Patcher(additional_skip_names=['pkgutil']) as patcher:
        fs = patcher.fs
        fs.add_real_file(datadir / 'pheno_real.mod', target_path='run1.mod')
        fs.add_real_file(datadir / 'pheno_real.phi', target_path='run1.phi')
        fs.add_real_file(datadir / 'pheno_real.lst', target_path='run1.lst')
        fs.add_real_file(datadir / 'pheno_real.ext', target_path='run1.ext')
        fs.add_real_file(datadir / 'pheno.dta', target_path='pheno.dta')
        model = Model('run1.mod')

        tool = ms.ModelSearch(model, 'stepwise', 'ABSORPTION(FO)')
        assert tool
