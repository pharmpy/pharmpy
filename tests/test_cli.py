import re

import pytest

from pharmpy import cli, source
from pharmpy.plugins.nonmem.records import etas_record


# Skip pkgutil, reload source
@pytest.mark.parametrize('fs', [[['pkgutil'], [source, etas_record]]], indirect=True)
@pytest.mark.parametrize('operation', ['*', '+'])
def test_add_covariate_effect(datadir, fs, operation):
    fs.add_real_file(datadir / 'pheno_real.mod', target_path='run1.mod')
    fs.add_real_file(datadir / 'pheno.dta', target_path='pheno.dta')

    args = ['model', 'add_cov_effect', 'run1.mod', 'CL', 'WGT', 'exp', '--operation', operation]
    cli.main(args)

    with open('run1.mod', 'r') as f_ori, open('run2.mod', 'r') as f_cov:
        mod_ori = f_ori.read()
        mod_cov = f_cov.read()

    assert mod_ori != mod_cov

    assert not re.search('CLWGT', mod_ori)
    assert re.search('CLWGT', mod_cov)


@pytest.mark.parametrize('fs', [[['pkgutil'], [source, etas_record]]], indirect=True)
@pytest.mark.parametrize(
    'transformation, eta', [('boxcox', 'ETAB1'), ('tdist', 'ETAT1'), ('john_draper', 'ETAD1')]
)
def test_eta_transformation(datadir, fs, transformation, eta):
    fs.add_real_file(datadir / 'pheno_real.mod', target_path='run1.mod')
    fs.add_real_file(datadir / 'pheno.dta', target_path='pheno.dta')

    args = ['model', transformation, 'run1.mod', '--etas', 'ETA(1)']
    cli.main(args)

    with open('run1.mod', 'r') as f_ori, open('run2.mod', 'r') as f_box:
        mod_ori = f_ori.read()
        mod_box = f_box.read()

    assert mod_ori != mod_box

    assert not re.search(eta, mod_ori)
    assert re.search(eta, mod_box)


@pytest.mark.parametrize('fs', [[['pkgutil'], [source, etas_record]]], indirect=True)
@pytest.mark.parametrize('operation', ['*', '+'])
def test_add_etas(datadir, fs, operation):
    fs.add_real_file(datadir / 'pheno_real.mod', target_path='run1.mod')
    fs.add_real_file(datadir / 'pheno.dta', target_path='pheno.dta')

    args = ['model', 'add_etas', 'run1.mod', 'S1', 'exp', '--operation', operation]
    cli.main(args)

    with open('run1.mod', 'r') as f_ori, open('run2.mod', 'r') as f_cov:
        mod_ori = f_ori.read()
        mod_cov = f_cov.read()

    assert mod_ori != mod_cov

    assert not re.search(r'EXP\(ETA\(3\)\)', mod_ori)
    assert re.search(r'EXP\(ETA\(3\)\)', mod_cov)
