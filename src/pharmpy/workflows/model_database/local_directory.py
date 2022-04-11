import json
import shutil
from os import stat
from pathlib import Path

from pharmpy.datainfo import DataInfo

from .baseclass import ModelDatabase

DIRECTORY_PHARMPY_METADATA = '.pharmpy'
FILE_METADATA = 'metadata.json'
FILE_MODELFIT_RESULTS = 'results.json'


class LocalDirectoryDatabase(ModelDatabase):
    """ModelDatabase implementation for single local directory

    All files will be stored in the same directory. It is assumed that
    all files connected to a model are named modelname + extension. This means that
    care must be taken to keep filenames unique. Clashing filenames will
    be overwritten. It is recommended to use the LocalModelDirectoryDatabase
    instead.

    Parameters
    ----------
    path : str or Path
        Path to the database directory. Will be created if it does not exist.
    file_extension : str
        File extension to use for model files.
    """

    def __init__(self, path='.', file_extension='.mod'):
        path = Path(path)
        if not path.exists():
            path.mkdir(parents=True)
        self.path = path.resolve()
        self.file_extension = file_extension
        self.ignored_names = frozenset(('stdout', 'stderr', 'nonmem.json', 'nlmixr.json'))

    def store_model(self, model):
        pass

    def store_local_file(self, model, path):
        path_object = Path(path)
        if path_object.name not in self.ignored_names and path_object.is_file():
            shutil.copy2(path, self.path)

    def retrieve_local_files(self, name, destination_path):
        # Retrieve all files stored for one model
        files = self.path.glob(f'{name}.*')
        for f in files:
            shutil.copy2(f, destination_path)

    def retrieve_file(self, name, filename):
        # Return path to file
        path = self.path / filename
        if path.is_file() and stat(path).st_size > 0:
            return path
        else:
            raise FileNotFoundError(f"Cannot retrieve {filename} for {name}")

    def get_model(self, name):
        filename = name + self.file_extension
        path = self.path / filename
        from pharmpy.model import Model

        try:
            model = Model.create_model(path)
        except FileNotFoundError:
            raise KeyError('Model cannot be found in database')
        model.database = self
        model.read_modelfit_results()
        return model

    def store_metadata(self, model, metadata):
        pass

    def store_modelfit_results(self, model):
        pass

    def __repr__(self):
        return f"LocalDirectoryDatabase({self.path})"


class LocalModelDirectoryDatabase(LocalDirectoryDatabase):
    """ModelDatabase implementation for a local directory structure

    Files will be stored in separate subdirectories named after each model.
    There are no restrictions on names of the files so models can have the same
    name of some connected file without creating a name clash.

    Parameters
    ----------
    path : str or Path
        Path to the base database directory. Will be created if it does not exist.
    file_extension : str
        File extension to use for model files.
    """

    def store_model(self, model):
        from pharmpy.modeling import read_dataset_from_datainfo, write_csv, write_model

        model = model.copy()
        model.update_datainfo()
        path = self.path / '.datasets'
        path.mkdir(parents=True, exist_ok=True)
        for dipath in path.glob('*.datainfo'):
            curdi = DataInfo.read_json(dipath)
            if curdi == model.datainfo:
                df = read_dataset_from_datainfo(curdi)
                if df.equals(model.dataset):
                    model.datainfo.path = curdi.path
                    break
        else:
            data_path = path / (model.name + '.csv')
            model.datainfo.path = data_path.resolve()
            model.datainfo.to_json(path / (model.name + '.datainfo'))
            write_csv(model, path=data_path)
        model_path = self.path / model.name
        model_path.mkdir(exist_ok=True)
        write_model(model, model_path / (model.name + model.filename_extension))

    def store_local_file(self, model, path):
        if Path(path).is_file():
            destination = self.path / model.name
            if not destination.is_dir():
                destination.mkdir(parents=True)
            shutil.copy2(path, destination)

    def retrieve_local_files(self, name, destination_path):
        path = self.path / name
        files = path.glob('*')
        for f in files:
            if f.is_file():
                shutil.copy2(f, destination_path)
            else:
                shutil.copytree(f, Path(destination_path) / f.stem)

    def retrieve_file(self, name, filename):
        # Return path to file
        path = self.path / name / filename
        if path.is_file() and stat(path).st_size > 0:
            return path
        else:
            raise FileNotFoundError(f"Cannot retrieve {filename} for {name}")

    def get_model(self, name):
        filename = name + self.file_extension
        path = self.path / name / filename
        from pharmpy.model import Model

        model = Model.create_model(path)
        model.database = self
        model.read_modelfit_results()
        return model

    def store_metadata(self, model, metadata):
        destination = self.path / model.name / DIRECTORY_PHARMPY_METADATA
        if not destination.is_dir():
            destination.mkdir(parents=True)
        with open(destination / FILE_METADATA, 'w') as f:
            json.dump(metadata, f, indent=2)

    def store_modelfit_results(self, model):
        destination = self.path / model.name / DIRECTORY_PHARMPY_METADATA
        if not destination.is_dir():
            destination.mkdir(parents=True)

        if model.modelfit_results:
            model.modelfit_results.to_json(destination / FILE_MODELFIT_RESULTS)

    def __repr__(self):
        return f"LocalModelDirectoryDatabase({self.path})"
