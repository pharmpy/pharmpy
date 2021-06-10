import os
import subprocess
import uuid
from pathlib import Path

from pharmpy.plugins.nonmem import conf
from pharmpy.tools.workflows import Task, Workflow
from pharmpy.utils import TemporaryDirectoryChanger


def create_workflow(models=None):
    wf_run = Workflow()

    if models:
        for model in models:
            task_execute = Task('run', execute_model, model)
            wf_run.add_tasks(task_execute)
    else:
        task_execute = Task('run', execute_model)
        wf_run.add_tasks(task_execute)

    result_task = Task('fit_results', results)
    wf_run.add_tasks(result_task, connect=True, as_single_element=False)

    return wf_run


def execute_model(model):
    path = Path.cwd() / f'NONMEM_run_{model.name}-{uuid.uuid1()}'
    path.mkdir(parents=True, exist_ok=True)
    model = model.copy()
    model.update_source(nofiles=True)
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
        model.database.store_local_file(model, basepath.with_suffix('.mod'))
        model.database.store_local_file(model, basepath.with_suffix('.lst'))
        model.database.store_local_file(model, basepath.with_suffix('.ext'))
        model.database.store_local_file(model, basepath.with_suffix('.phi'))
        cov_path = basepath.with_suffix('.cov')
        if cov_path.is_file():
            model.database.store_local_file(model, cov_path)
    return model


def results(models):
    for model in models:
        model._modelfit_results = None
        # FIXME: On the fly reading doesn't work since files
        # doesn't get copied up. Reading in now as a workaround.
        model.modelfit_results.ofv
        model.modelfit_results.covariance_matrix
        model.modelfit_results.individual_estimates
    return models


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
