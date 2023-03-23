import io
import os
import re
import shutil
from contextlib import redirect_stdout

import pytest

import pharmpy.cli as cli
from pharmpy.internals.fs.cwd import chdir


def test_model_print(datadir, capsys):
    args = ['model', 'print', str(datadir / 'pheno.mod')]
    cli.main(args)
    captured = capsys.readouterr()
    assert 'ETA₁' in captured.out


@pytest.mark.parametrize('operation', ['*', '+'])
def test_add_covariate_effect(datadir, operation, tmp_path):
    shutil.copy(datadir / 'pheno.mod', tmp_path / 'run1.mod')
    shutil.copy(datadir / 'pheno.dta', tmp_path / 'pheno.dta')
    shutil.copy(datadir / 'pheno.datainfo', tmp_path / 'pheno.datainfo')

    with chdir(tmp_path):
        args = ['model', 'add_cov_effect', 'run1.mod', 'CL', 'WGT', 'exp', '--operation', operation]
        cli.main(args)

        with open('run1.mod', 'r') as f_ori, open('run2.mod', 'r') as f_cov:
            mod_ori = f_ori.read()
            mod_cov = f_cov.read()

        assert mod_ori != mod_cov

        assert not re.search('CLWGT', mod_ori)
        assert re.search('CLWGT', mod_cov)


@pytest.mark.parametrize(
    'transformation, eta', [('boxcox', 'ETAB1'), ('tdist', 'ETAT1'), ('john_draper', 'ETAD1')]
)
def test_eta_transformation(datadir, transformation, eta, tmp_path):
    shutil.copy(datadir / 'pheno_real.mod', tmp_path / 'run1.mod')
    shutil.copy(datadir / 'pheno.dta', tmp_path / 'pheno.dta')

    with chdir(tmp_path):
        args = ['model', transformation, 'run1.mod', '--etas', 'ETA_1']
        cli.main(args)

        with open('run1.mod', 'r') as f_ori, open('run2.mod', 'r') as f_box:
            mod_ori = f_ori.read()
            mod_box = f_box.read()

        assert mod_ori != mod_box

        assert not re.search(eta, mod_ori)
        assert re.search(eta, mod_box)


@pytest.mark.parametrize(
    'options, eta_name',
    [
        (['--operation', '+'], 'ETA_S1'),
        (['--operation', '*'], 'ETA_S1'),
        (['--eta_name', 'ETA(3)'], 'ETA(3)'),
    ],
)
def test_add_iiv(datadir, options, eta_name, tmp_path):
    shutil.copy(datadir / 'pheno_real.mod', tmp_path / 'run1.mod')
    shutil.copy(datadir / 'pheno.dta', tmp_path / 'pheno.dta')

    with chdir(tmp_path):
        args = ['model', 'add_iiv', 'run1.mod', 'S1', 'exp'] + options
        cli.main(args)

        with open('run1.mod', 'r') as f_ori, open('run2.mod', 'r') as f_cov:
            mod_ori = f_ori.read()
            mod_cov = f_cov.read()

        assert mod_ori != mod_cov
        assert f'EXP({eta_name})' not in mod_ori
        assert f'EXP({eta_name})' in mod_cov


@pytest.mark.parametrize('options', [['--eta_names', 'ETA(3) ETA(4)']])
def test_add_iov(datadir, options, tmp_path):
    shutil.copy(datadir / 'pheno_real.mod', tmp_path / 'run1.mod')
    shutil.copy(datadir / 'pheno.dta', tmp_path / 'pheno.dta')

    with chdir(tmp_path):
        args = ['model', 'add_iov', 'run1.mod', 'FA1', '--etas', 'ETA_1'] + options
        cli.main(args)

        with open('run1.mod', 'r') as f_ori, open('run2.mod', 'r') as f_cov:
            mod_ori = f_ori.read()
            mod_cov = f_cov.read()

        assert mod_ori != mod_cov

        assert not re.search(r'ETAI1', mod_ori)
        assert re.search(r'ETAI1', mod_cov)


def test_results_linearize(datadir, tmp_path):
    path = datadir / 'linearize' / 'linearize_dir1'
    shutil.copy(path / 'pheno_linbase.mod', tmp_path)
    shutil.copy(path / 'pheno_linbase.ext', tmp_path)
    shutil.copy(path / 'pheno_linbase.lst', tmp_path)
    shutil.copy(path / 'pheno_linbase.phi', tmp_path)
    scmdir = tmp_path / 'scm_dir1'
    scmdir.mkdir()
    shutil.copy(path / 'scm_dir1' / 'derivatives.mod', scmdir)
    shutil.copy(path / 'scm_dir1' / 'derivatives.ext', scmdir)
    shutil.copy(path / 'scm_dir1' / 'derivatives.lst', scmdir)
    shutil.copy(path / 'scm_dir1' / 'derivatives.phi', scmdir)

    args = ['results', 'linearize', str(tmp_path)]
    cli.main(args)

    assert os.path.exists(tmp_path / 'results.json')


@pytest.mark.parametrize('eta_args', [['--etas', 'ETA_1 ETA_2'], []])
def test_create_joint_distribution(datadir, eta_args, tmp_path):
    shutil.copy(datadir / 'pheno_real.mod', tmp_path / 'run1.mod')
    shutil.copy(datadir / 'pheno.dta', tmp_path / 'pheno.dta')

    with chdir(tmp_path):
        args = ['model', 'create_joint_distribution', 'run1.mod'] + eta_args
        cli.main(args)

        with open('run1.mod', 'r') as f_ori, open('run2.mod', 'r') as f_cov:
            mod_ori = f_ori.read()
            mod_cov = f_cov.read()

        assert mod_ori != mod_cov

        assert not re.search(r'BLOCK\(2\)', mod_ori)
        assert re.search(r'BLOCK\(2\)', mod_cov)


@pytest.mark.parametrize(
    'epsilons_args, eta_name',
    [
        (['--eps', 'EPS_1'], 'ETA_RV1'),
        ([], 'ETA_RV1'),
        (['--same_eta', 'False'], 'ETA_RV1'),
        (['--eta_names', 'ETA_3'], 'ETA(3)'),
    ],
)
def test_iiv_on_ruv(datadir, epsilons_args, eta_name, tmp_path):
    shutil.copy(datadir / 'pheno_real.mod', tmp_path / 'run1.mod')
    shutil.copy(datadir / 'pheno.dta', tmp_path / 'pheno.dta')

    with chdir(tmp_path):
        args = ['model', 'iiv_on_ruv', 'run1.mod'] + epsilons_args
        cli.main(args)

        with open('run1.mod', 'r') as f_ori, open('run2.mod', 'r') as f_cov:
            mod_ori = f_ori.read()
            mod_cov = f_cov.read()

        assert mod_ori != mod_cov
        assert f'EXP({eta_name})' not in mod_ori
        assert f'EXP({eta_name})' in mod_cov


@pytest.mark.parametrize('to_remove', [['--to_remove', 'ETA_2'], []])
def test_remove_iiv(datadir, to_remove, tmp_path):
    shutil.copy(datadir / 'pheno_real.mod', tmp_path / 'run1.mod')
    shutil.copy(datadir / 'pheno.dta', tmp_path / 'pheno.dta')

    with chdir(tmp_path):
        args = ['model', 'remove_iiv', 'run1.mod'] + to_remove
        cli.main(args)

        with open('run1.mod', 'r') as f_ori, open('run2.mod', 'r') as f_cov:
            mod_ori = f_ori.read()
            mod_cov = f_cov.read()

        assert mod_ori != mod_cov

        assert re.search(r'EXP\(ETA\(2\)\)', mod_ori)
        assert not re.search(r'EXP\(ETA\(2\)\)', mod_cov)


def test_remove_iov(datadir, tmp_path):
    shutil.copy(datadir / 'qa/iov.mod', tmp_path / 'run1.mod')
    shutil.copy(datadir / 'pheno.dta', tmp_path / 'pheno.dta')

    with chdir(tmp_path):
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


@pytest.mark.parametrize('epsilons_args', [['--eps', 'EPS_1'], []])
def test_power_on_ruv(datadir, epsilons_args, tmp_path):
    shutil.copy(datadir / 'pheno_real.mod', tmp_path / 'run1.mod')
    shutil.copy(datadir / 'pheno.dta', tmp_path / 'pheno.dta')

    with chdir(tmp_path):
        args = ['model', 'power_on_ruv', 'run1.mod'] + epsilons_args
        cli.main(args)

        with open('run1.mod', 'r') as f_ori, open('run2.mod', 'r') as f_cov:
            mod_ori = f_ori.read()
            mod_cov = f_cov.read()

        assert mod_ori != mod_cov

        assert not re.search(r'\*\*', mod_ori)
        assert re.search(r'\*\*', mod_cov)


def test_update_inits(datadir, tmp_path):
    shutil.copy(datadir / 'pheno_real.mod', tmp_path / 'run1.mod')
    shutil.copy(datadir / 'pheno_real.ext', tmp_path / 'run1.ext')
    shutil.copy(datadir / 'pheno_real.phi', tmp_path / 'run1.phi')
    shutil.copy(datadir / 'pheno.dta', tmp_path / 'pheno.dta')

    with chdir(tmp_path):
        args = ['model', 'update_inits', 'run1.mod']
        cli.main(args)


def test_model_sample(datadir, tmp_path):
    shutil.copy(datadir / 'pheno_real.mod', tmp_path / 'run1.mod')
    shutil.copy(datadir / 'pheno_real.ext', tmp_path / 'run1.ext')
    shutil.copy(datadir / 'pheno_real.lst', tmp_path / 'run1.lst')
    shutil.copy(datadir / 'pheno_real.phi', tmp_path / 'run1.phi')
    shutil.copy(datadir / 'pheno_real.cov', tmp_path / 'run1.cov')
    shutil.copy(datadir / 'pheno.dta', tmp_path / 'pheno.dta')

    with chdir(tmp_path):
        args = ['model', 'sample', 'run1.mod', '--seed=24']
        cli.main(args)

        with open('run1.mod', 'r') as f_ori, open('sample_1.mod', 'r') as f_cov:
            mod_ori = f_ori.read()
            mod_cov = f_cov.read()

    assert mod_ori != mod_cov


def test_usage():
    f = io.StringIO()
    with redirect_stdout(f):
        cli.main([])
        out = f.getvalue()
    assert 'usage:' in out


def test_main():
    import pharmpy.__main__ as main

    main
