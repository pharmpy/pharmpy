import shutil
import subprocess
from pathlib import Path

from dask.multiprocessing import get


class NONMEMRunDirectory:
    def __init__(self, model, n, path=None):
        """Create a directory for running NONMEN.
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
        model.write(path=self.path, force=True)
        shutil.copy(model.dataset_path, self.path)   # FIXME!


def run(models):
    dsk = {f'run-{i}': (execute_model, model, i) for i, model in enumerate(models)}
    dsk['results'] = (results, ['run-%d' % i for i, _ in enumerate(models)])
    res = get(dsk, 'results')  # executes in parallel
    print(res)


def execute_model(model, i):
    rundir = NONMEMRunDirectory(model, i)
    subprocess.call(['nmfe74', model.name + model.source.filename_extension,
                     Path(model.name).with_suffix('.lst'), f'-rundir={rundir.path}'])
    return model


def results(res):
    return res
