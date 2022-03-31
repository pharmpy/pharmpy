import os
import subprocess
import uuid
from pathlib import Path

from pharmpy.modeling import write_csv, write_model
from pharmpy.plugins.nonmem import conf, convert_model


def execute_model(model):
    database = model.database
    parent_model = model.parent_model
    model = convert_model(model)
    path = Path.cwd() / f'NONMEM_run_{model.name}-{uuid.uuid1()}'
    path.mkdir(parents=True, exist_ok=True)
    model = model.copy()
    model.parent_model = parent_model
    write_csv(model, path=path, force=True)
    # Set local path
    model.datainfo.path = model.datainfo.path.relative_to(model.datainfo.path.parent)
    model._dataset_updated = True  # Hack to get update_source to update IGNORE
    write_model(model, path=path, force=True)
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
    database.store_model(model)
    database.store_local_file(model, (path / basepath).with_suffix('.lst'))
    database.store_local_file(model, (path / basepath).with_suffix('.ext'))
    database.store_local_file(model, (path / basepath).with_suffix('.phi'))
    database.store_local_file(model, (path / basepath).with_suffix('.cov'))
    database.store_local_file(model, (path / basepath).with_suffix('.cor'))
    database.store_local_file(model, (path / basepath).with_suffix('.coi'))
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


def evaluate_design(model):
    # Prepare and run model for design evaluation
    model = model.copy()
    model.name = '_design_model'

    estrecs = model.control_stream.get_records('ESTIMATION')
    model.control_stream.remove_records(estrecs)

    design_code = '$DESIGN APPROX=FOCEI MODE=1 NELDER FIMDIAG=0 DATASIM=1 GROUPSIZE=32 OFVTYPE=0'
    model.control_stream.insert_record(design_code)

    execute_model(model)

    from pharmpy.tools.evaldesign import EvalDesignResults

    res = EvalDesignResults(
        ofv=model.modelfit_results.ofv,
        individual_ofv=model.modelfit_results.individual_ofv,
        information_matrix=model.modelfit_results.information_matrix,
    )
    return res
