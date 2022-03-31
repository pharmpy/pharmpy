import os.path
import shutil
from os import stat
from pathlib import Path

from pharmpy.datainfo import DataInfo

from .baseclass import ModelDatabase


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

    def store_model(self, model):
        pass

    def store_local_file(self, model, path):
        if Path(path).is_file():
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
        from pharmpy.modeling import write_csv, write_model

        model = model.copy()
        model.update_datainfo()
        path = self.path / '.datasets'
        if not path.is_dir():
            path.mkdir(parents=True)
        di = model.datainfo.copy()
        for dipath in path.glob('*.datainfo'):
            curdi = DataInfo.read_json(dipath)
            di.path = curdi.path  # This could be different in comparison
            if curdi == di:  # TODO: Should compare dataset here as well
                model.datainfo.path = curdi.path
                break
        else:
            data_path = path / (model.name + '.csv')
            di.path = os.path.relpath(data_path, self.path / model.name)
            di.to_json(path / (model.name + '.datainfo'))
            write_csv(model, path=data_path)
            model.datainfo = di
        model_path = self.path / model.name
        if not model_path.is_dir():
            model_path.mkdir()
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
            shutil.copy2(f, destination_path)

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

    def __repr__(self):
        return f"LocalModelDirectoryDatabase({self.path})"
