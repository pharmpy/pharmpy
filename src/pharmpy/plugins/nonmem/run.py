import os
import subprocess
from pathlib import Path

from pharmpy.plugins.nonmem import conf
from pharmpy.tools.workflows import Task, Workflow


def create_workflow(models):
    wf = Workflow(models=models)
    task_names, execute_tasks = [], []

    for i, model in enumerate(models):
        execute_tasks.append(Task(f'run-{i}', execute_model, (model, i)))
        task_names.append(f'run-{i}')

    wf.add_tasks(execute_tasks)
    wf.add_tasks(Task('results', results, task_names))

    for i, model in enumerate(models):
        wf.add_infiles(model.dataset_path, destination=f'NONMEM_run{i}')
        wf.add_outfiles(
            [
                f'NONMEM_run{i}/{model.name}.lst',
                f'NONMEM_run{i}/{model.name}.ext',
                f'NONMEM_run{i}/{model.name}.phi',
            ]
        )
    return wf


def execute_model(model, i):
    path = Path(f'NONMEM_run{i}').resolve()
    model = model.copy()
    model.update_source()
    model.dataset_path = model.dataset_path.name  # Make path in $DATA local
    model.write(path=path, force=True)
    args = [
        nmfe_path(),
        model.name + model.source.filename_extension,
        str(Path(model.name).with_suffix('.lst')),
        f'-rundir={str(path)}',
    ]
    subprocess.call(
        args, stdin=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL
    )
    return model


def results(models):
    for model in models:
        model._modelfit_results = None
        model.modelfit_results.ofv
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
