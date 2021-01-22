import os
import re

import pytest

from pharmpy import cli, source
from pharmpy.plugins.nonmem.records import etas_record


def test_model_print(datadir, capsys):
    args = ['model', 'print', str(datadir / 'pheno.mod')]
    cli.main(args)
    captured = capsys.readouterr()
    assert 'ETA(1)' in captured.out


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
@pytest.mark.parametrize(
    'options', [['--operation', '+'], ['--operation', '*'], ['--eta_name', 'ETA(3)']]
)
def test_add_iiv(datadir, fs, options):
    fs.add_real_file(datadir / 'pheno_real.mod', target_path='run1.mod')
    fs.add_real_file(datadir / 'pheno.dta', target_path='pheno.dta')

    args = ['model', 'add_iiv', 'run1.mod', 'S1', 'exp'] + options
    cli.main(args)

    with open('run1.mod', 'r') as f_ori, open('run2.mod', 'r') as f_cov:
        mod_ori = f_ori.read()
        mod_cov = f_cov.read()

    assert mod_ori != mod_cov

    assert not re.search(r'EXP\(ETA\(3\)\)', mod_ori)
    assert re.search(r'EXP\(ETA\(3\)\)', mod_cov)


@pytest.mark.parametrize('fs', [[['pkgutil'], [source, etas_record]]], indirect=True)
@pytest.mark.parametrize('options', [['--eta_names', 'ETA(3) ETA(4)']])
def test_add_iov(datadir, fs, options):
    fs.add_real_file(datadir / 'pheno_real.mod', target_path='run1.mod')
    fs.add_real_file(datadir / 'pheno.dta', target_path='pheno.dta')

    args = ['model', 'add_iov', 'run1.mod', 'FA1', '--etas', 'ETA(1)'] + options
    cli.main(args)

    with open('run1.mod', 'r') as f_ori, open('run2.mod', 'r') as f_cov:
        mod_ori = f_ori.read()
        mod_cov = f_cov.read()

    assert mod_ori != mod_cov

    assert not re.search(r'ETAI1', mod_ori)
    assert re.search(r'ETAI1', mod_cov)


@pytest.mark.parametrize('fs', [[['pkgutil'], [source, cli]]], indirect=True)
def test_results_linearize(datadir, fs):
    path = datadir / 'linearize' / 'linearize_dir1'
    fs.create_dir('linearize_dir1')
    fs.add_real_file(path / 'pheno_linbase.mod', target_path='linearize_dir1/pheno_linbase.mod')
    fs.add_real_file(path / 'pheno_linbase.ext', target_path='linearize_dir1/pheno_linbase.ext')
    fs.add_real_file(path / 'pheno_linbase.lst', target_path='linearize_dir1/pheno_linbase.lst')
    fs.add_real_file(path / 'pheno_linbase.phi', target_path='linearize_dir1/pheno_linbase.phi')
    fs.create_dir('linearize_dir1/scm_dir1')
    fs.add_real_file(
        path / 'scm_dir1' / 'derivatives.mod', target_path='linearize_dir1/scm_dir1/derivatives.mod'
    )
    fs.add_real_file(
        path / 'scm_dir1' / 'derivatives.ext', target_path='linearize_dir1/scm_dir1/derivatives.ext'
    )
    fs.add_real_file(
        path / 'scm_dir1' / 'derivatives.lst', target_path='linearize_dir1/scm_dir1/derivatives.lst'
    )
    fs.add_real_file(
        path / 'scm_dir1' / 'derivatives.phi', target_path='linearize_dir1/scm_dir1/derivatives.phi'
    )

    args = ['results', 'linearize', 'linearize_dir1']
    cli.main(args)

    assert os.path.exists('linearize_dir1/results.json')


@pytest.mark.parametrize('fs', [[['pkgutil'], [source, etas_record]]], indirect=True)
@pytest.mark.parametrize('eta_args', [['--etas', 'ETA(1) ETA(2)'], []])
def test_create_rv_block(datadir, fs, eta_args):
    fs.add_real_file(datadir / 'pheno_real.mod', target_path='run1.mod')
    fs.add_real_file(datadir / 'pheno.dta', target_path='pheno.dta')

    args = ['model', 'create_rv_block', 'run1.mod'] + eta_args
    cli.main(args)

    with open('run1.mod', 'r') as f_ori, open('run2.mod', 'r') as f_cov:
        mod_ori = f_ori.read()
        mod_cov = f_cov.read()

    assert mod_ori != mod_cov

    assert not re.search(r'BLOCK\(2\)', mod_ori)
    assert re.search(r'BLOCK\(2\)', mod_cov)


@pytest.mark.parametrize('fs', [[['pkgutil'], [source, etas_record]]], indirect=True)
@pytest.mark.parametrize(
    'epsilons_args', [['--eps', 'EPS(1)'], [], ['--same_eta', 'False'], ['--eta_names', 'ETA(3)']]
)
def test_iiv_on_ruv(datadir, fs, epsilons_args):
    fs.add_real_file(datadir / 'pheno_real.mod', target_path='run1.mod')
    fs.add_real_file(datadir / 'pheno.dta', target_path='pheno.dta')

    args = ['model', 'iiv_on_ruv', 'run1.mod'] + epsilons_args
    cli.main(args)

    with open('run1.mod', 'r') as f_ori, open('run2.mod', 'r') as f_cov:
        mod_ori = f_ori.read()
        mod_cov = f_cov.read()

    assert mod_ori != mod_cov

    assert not re.search(r'EXP\(ETA\(3\)\)', mod_ori)
    assert re.search(r'EXP\(ETA\(3\)\)', mod_cov)


@pytest.mark.parametrize('fs', [[['pkgutil'], [source, etas_record]]], indirect=True)
@pytest.mark.parametrize('to_remove', [['--to_remove', 'ETA(2)'], []])
def test_remove_iiv(datadir, fs, to_remove):
    fs.add_real_file(datadir / 'pheno.mod', target_path='run1.mod')
    fs.add_real_file(datadir / 'pheno.dta', target_path='pheno.dta')

    args = ['model', 'remove_iiv', 'run1.mod'] + to_remove
    cli.main(args)

    with open('run1.mod', 'r') as f_ori, open('run2.mod', 'r') as f_cov:
        mod_ori = f_ori.read()
        mod_cov = f_cov.read()

    assert mod_ori != mod_cov

    assert re.search(r'EXP\(ETA\(2\)\)', mod_ori)
    assert not re.search(r'EXP\(ETA\(2\)\)', mod_cov)


@pytest.mark.parametrize('fs', [[['pkgutil'], [source, etas_record]]], indirect=True)
def test_remove_iov(datadir, fs):
    fs.add_real_file(datadir / 'qa/iov.mod', target_path='run1.mod')
    fs.add_real_file(datadir / 'pheno.dta', target_path='pheno.dta')

    args = ['model', 'remove_iov', 'run1.mod']
    cli.main(args)

    with open('run1.mod', 'r') as f_ori, open('run2.mod', 'r') as f_cov:
        mod_ori = f_ori.read()
        mod_cov = f_cov.read()

    assert mod_ori != mod_cov

    assert re.search('SAME', mod_ori)
    assert not re.search('SAME', mod_cov)

    assert re.search(r'ETA\(3\)', mod_ori)
    assert not re.search(r'ETA\(3\)', mod_cov)


@pytest.mark.parametrize('fs', [[['pkgutil'], [source, etas_record]]], indirect=True)
@pytest.mark.parametrize('epsilons_args', [['--eps', 'EPS(1)'], []])
def test_power_on_ruv(datadir, fs, epsilons_args):
    fs.add_real_file(datadir / 'pheno_real.mod', target_path='run1.mod')
    fs.add_real_file(datadir / 'pheno.dta', target_path='pheno.dta')

    args = ['model', 'power_on_ruv', 'run1.mod'] + epsilons_args
    cli.main(args)

    with open('run1.mod', 'r') as f_ori, open('run2.mod', 'r') as f_cov:
        mod_ori = f_ori.read()
        mod_cov = f_cov.read()

    assert mod_ori != mod_cov

    assert not re.search(r'CIPREDI', mod_ori)
    assert re.search(r'CIPREDI', mod_cov)


@pytest.mark.parametrize('fs', [[['pkgutil'], [source, etas_record]]], indirect=True)
@pytest.mark.parametrize(
    'force_args, file_exists', [(['--force_update', 'True'], True), ([], False)]
)
def test_update_inits(datadir, fs, force_args, file_exists):
    fs.add_real_file(datadir / 'pheno_real.mod', target_path='run1.mod')
    fs.add_real_file(datadir / 'pheno_real.ext', target_path='run1.ext')
    fs.add_real_file(datadir / 'pheno_real.phi', target_path='run1.phi')
    fs.add_real_file(datadir / 'pheno.dta', target_path='pheno.dta')

    args = ['model', 'update_inits', 'run1.mod'] + force_args
    cli.main(args)

    with open('run1.mod', 'r') as f_ori, open('run2.mod', 'r') as f_cov:
        mod_ori = f_ori.read()
        mod_cov = f_cov.read()

    assert mod_ori != mod_cov

    assert not re.search(r'\$ETAS FILE=run2_input.phi', mod_ori)
    assert bool(re.search(r'\$ETAS FILE=run2_input.phi', mod_cov)) is file_exists
    assert (os.path.isfile('run2_input.phi')) is file_exists


def test_main():
    import pharmpy.__main__ as main

    main
