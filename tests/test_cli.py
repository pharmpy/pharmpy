import re

import pytest

from pharmpy import cli, source
from pharmpy.plugins.nonmem.records import etas_record


# Skip pkgutil, reload source
@pytest.mark.parametrize('fs', [[['pkgutil'], [source, etas_record]]], indirect=True)
def test_add_covariate_effect(datadir, fs):
    fs.add_real_file(datadir / 'pheno_real.mod', target_path='run1.mod')
    fs.add_real_file(datadir / 'pheno.dta', target_path='pheno.dta')

    args = ['model', 'add_cov_effect', 'run1.mod', 'CL', 'WGT', 'exp']
    cli.main(args)

    with open('run1.mod', 'r') as f_ori, open('run2.mod', 'r') as f_cov:
        mod_ori = f_ori.read()
        mod_cov = f_cov.read()

    assert mod_ori != mod_cov

    assert not re.search('CLWGT', mod_ori)
    assert re.search('CLWGT', mod_cov)
