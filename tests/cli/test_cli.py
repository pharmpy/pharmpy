import io
import os
import re
import shutil
from contextlib import redirect_stdout

import pharmpy.cli as cli
from pharmpy.internals.fs.cwd import chdir


def test_model_print(datadir, capsys):
    args = ['model', 'print', str(datadir / 'pheno.mod')]
    cli.main(args)
    captured = capsys.readouterr()
    assert 'ETA₁' in captured.out


def test_results_linearize(datadir, tmp_path):
    path = datadir / 'linearize' / 'linearize_dir1'
    shutil.copy(path / 'pheno_linbase.mod', tmp_path)
    shutil.copy(path / 'pheno_linbase.ext', tmp_path)
    shutil.copy(path / 'pheno_linbase.lst', tmp_path)
    shutil.copy(path / 'pheno_linbase.phi', tmp_path)
    shutil.copy(path / 'scm_dir1' / 'pheno_linbase.dta', tmp_path)
    scmdir = tmp_path / 'scm_dir1'
    scmdir.mkdir()
    shutil.copy(path / 'scm_dir1' / 'derivatives.mod', scmdir)
    shutil.copy(path / 'scm_dir1' / 'derivatives.ext', scmdir)
    shutil.copy(path / 'scm_dir1' / 'derivatives.lst', scmdir)
    shutil.copy(path / 'scm_dir1' / 'derivatives.phi', scmdir)

    args = ['psn', 'linearize', str(tmp_path)]
    cli.main(args)

    assert os.path.exists(tmp_path / 'results.json')


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

        with open('run1.mod', 'r') as f_ori, open('sample_1.ctl', 'r') as f_cov:
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


def test_reference(datadir, tmp_path):
    shutil.copy(datadir / 'pheno_real.mod', tmp_path / 'run1.mod')
    shutil.copy(datadir / 'pheno.dta', tmp_path / 'pheno.dta')

    with chdir(tmp_path):
        args = ['data', 'reference', 'run1.mod', 'WGT=70', 'APGR=4', '--output', 'run2.mod']
        cli.main(args)

        with open('run2.mod', 'r') as f_new:
            mod_new = f_new.read()

        assert re.search(r'run2\.csv', mod_new)
