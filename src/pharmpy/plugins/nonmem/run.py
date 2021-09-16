import os
import subprocess
import uuid
from pathlib import Path

from pharmpy.plugins.nonmem import conf, convert_model
from pharmpy.utils import TemporaryDirectoryChanger


def execute_model(model):
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
        model.name + model.source.filename_extension,
        str(basepath.with_suffix('.lst')),
    ]
    with TemporaryDirectoryChanger(path):
        subprocess.call(
            args, stdin=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL
        )
        model.database.store_local_file(
            model, basepath.with_suffix(model.source.filename_extension)
        )
        model.database.store_local_file(model, basepath.with_suffix('.lst'))
        model.database.store_local_file(model, basepath.with_suffix('.ext'))
        model.database.store_local_file(model, basepath.with_suffix('.phi'))
        cov_path = basepath.with_suffix('.cov')
        if cov_path.is_file():
            model.database.store_local_file(model, cov_path)

    # Read in results
    model._modelfit_results = None
    # FIXME: On the fly reading doesn't work since files
    #  doesn't get copied up. Reading in now as a workaround.
    if any(step.cov for step in model.estimation_steps):
        model.modelfit_results.covariance_matrix
    model.modelfit_results.minimization_successful
    model.modelfit_results.individual_estimates

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
