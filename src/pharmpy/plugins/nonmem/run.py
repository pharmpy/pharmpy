import os
import subprocess
import uuid
from pathlib import Path

from pharmpy.plugins.nonmem import conf, convert_model


def execute_model(model):
    database = model.database
    model = convert_model(model)
    path = Path.cwd() / f'NONMEM_run_{model.name}-{uuid.uuid1()}'
    path.mkdir(parents=True, exist_ok=True)
    model = model.copy()
    model._dataset_updated = True  # Hack to get update_source to update IGNORE
    model.update_source(nofiles=True)
    try:
        model.dataset.name
    except AttributeError:
        model.dataset.name = "dataset"
    datapath = model.dataset.pharmpy.write_csv(path=path)
    model.dataset_path = datapath.name  # Make path in $DATA local
    model.write(path=path, force=True)
    basepath = Path(model.name)
    args = [
        nmfe_path(),
        model.name + model.filename_extension,
        str(basepath.with_suffix('.lst')),
    ]
    # Create wrapper script that cd:s into rundirectory
    # This enables the execute_model function to be parallelized
    # using threads. chdir here does not work since all threads will
    # share cwd. Using processes on Windows from R currently hangs.
    # Also the -rundir option to nmfe does not work entirely
    if os.name == 'nt':
        with open(path / 'cdwrapper.bat', 'w') as fp:
            fp.write(f"@echo off\ncd {path}\n{' '.join(args)}\n")
        cmd = str(path / 'cdwrapper.bat')
    else:
        with open(path / 'cdwrapper', 'w') as fp:
            fp.write(f"#!/bin/sh\ncd {path}\n{' '.join(args)}\n")
        os.chmod(path / 'cdwrapper', 0o744)
        cmd = str(path / 'cdwrapper')
    subprocess.call(
        [cmd], stdin=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL
    )
    database.store_local_file(model, (path / basepath).with_suffix(model.filename_extension))
    database.store_local_file(model, (path / basepath).with_suffix('.lst'))
    database.store_local_file(model, (path / basepath).with_suffix('.ext'))
    database.store_local_file(model, (path / basepath).with_suffix('.phi'))
    database.store_local_file(model, (path / basepath).with_suffix('.cov'))
    database.store_local_file(model, (path / basepath).with_suffix('.cor'))
    database.store_local_file(model, (path / basepath).with_suffix('.coi'))
    if False:
        # FIXME: Have option for saving the dataset.
        database.store_local_file(model, (path / model.dataset.name).with_suffix('.csv'))
    for rec in model.control_stream.get_records('TABLE'):
        database.store_local_file(model, path / rec.path)
    # Read in results for the server side
    model.read_modelfit_results()
    # FIXME: the database path is changed in write
    model.database = database

    return model


def nmfe_path():
    if os.name == 'nt':
        nmfe_candidates = ['nmfe74.bat', 'nmfe75.bat', 'nmfe73.bat']
    else:
        nmfe_candidates = ['nmfe74', 'nmfe75', 'nmfe73']
    path = conf.default_nonmem_path
    if path != Path(''):
        path /= 'run'
    for nmfe in nmfe_candidates:
        candidate_path = path / nmfe
        if candidate_path.is_file():
            path = candidate_path
            break
    else:
        raise FileNotFoundError(f'Cannot find nmfe script for NONMEM ({path})')
    return str(path)
