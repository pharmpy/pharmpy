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
    path = Path(f'NONMEM_run{i}')
    temp = model.source.path
    model.source.path = path / 'dummy'
    model.write(path=path, force=True)
    model.source.path = temp
    args = [
        nmfe_path(),
        model.name + model.source.filename_extension,
        str(Path(model.name).with_suffix('.lst')),
        f'-rundir={str(path)}',
    ]
    subprocess.call(args)
    return model


def results(models):
    for model in models:
        model.modelfit_results = None
    return models


def nmfe_path():
    if os.name == 'nt':
        nmfe = 'nmfe74.bat'
    else:
        nmfe = 'nmfe74'
    path = conf.default_nonmem_path
    if path != Path(''):
        path /= 'run'
    path /= nmfe
    return str(path)
