import os
import subprocess
from pathlib import Path

from pharmpy.methods.modelfit.job import ModelfitJob
from pharmpy.plugins.nonmem import conf


def create_job(models):
    # Return ModelfitJob object
    dsk = {f'run-{i}': (execute_model, model, i) for i, model in enumerate(models)}
    dsk['results'] = (results, ['run-%d' % i for i, _ in enumerate(models)])
    job = ModelfitJob(dsk)
    job.models = models
    for i, model in enumerate(models):
        job.add_infiles(model.dataset_path, destination=f'NONMEM_run{i}')
        job.add_outfiles(
            [
                f'NONMEM_run{i}/{model.name}.lst',
                f'NONMEM_run{i}/{model.name}.ext',
                f'NONMEM_run{i}/{model.name}.phi',
            ]
        )
    return job


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
    subprocess.call(args, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
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
