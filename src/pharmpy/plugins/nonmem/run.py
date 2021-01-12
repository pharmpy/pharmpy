import shutil
import subprocess
from pathlib import Path

from pharmpy.plugins.nonmem import conf


class NONMEMRunDirectory:
    def __init__(self, model, n, path=None):
        """Create a directory for running NONMEM.
        If path not specified use the current path
        n is the index of the run
        """
        if not path:
            path = Path.cwd()
        name = f'NONMEM_run{n}'
        self.path = path / name
        self.path.mkdir(exist_ok=True)
        self.write_model(model)

    def write_model(self, model):
        shutil.copy(model.dataset_path, self.path)  # FIXME!
        self.model_path = model.write(path=self.path, force=True)


def run(models, path):
    dsk = {f'run-{i}': (execute_model, model, i, path) for i, model in enumerate(models)}
    dsk['results'] = (results, ['run-%d' % i for i, _ in enumerate(models)])
    return dsk


def execute_model(model, i, path):
    original_model_path = model.source.path
    rundir = NONMEMRunDirectory(model, i, path=path)
    subprocess.call(
        [
            nmfe_path(),
            model.name + model.source.filename_extension,
            Path(model.name).with_suffix('.lst'),
            f'-rundir={rundir.path}',
        ]
    )
    model.modelfit_results = None
    for ext in ['ext', 'lst', 'phi']:
        source_path = rundir.model_path.with_suffix(f'.{ext}')
        dest_path = original_model_path.with_suffix(f'.{ext}')
        shutil.copy(source_path, dest_path)
    return model


def results(model):
    return model


def nmfe_path():
    path = conf.default_nonmem_path
    if path != Path(''):
        path /= 'run'
    path /= 'nmfe74'
    return str(path)
