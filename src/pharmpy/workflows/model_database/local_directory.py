import json
import shutil
from contextlib import contextmanager
from os import stat
from pathlib import Path
from typing import Union

from pharmpy.internals.df import hash_df_fs
from pharmpy.internals.fs.lock import path_lock
from pharmpy.internals.fs.path import path_absolute
from pharmpy.model import DataInfo, Model
from pharmpy.results import read_results

from .baseclass import (
    ModelSnapshot,
    ModelTransaction,
    NonTransactionalModelDatabase,
    PendingTransactionError,
    TransactionalModelDatabase,
)

DIRECTORY_PHARMPY_METADATA = '.pharmpy'
DIRECTORY_DATASETS = '.datasets'
DIRECTORY_INDEX = '.hash'
FILE_METADATA = 'metadata.json'
FILE_MODELFIT_RESULTS = 'results.json'
FILE_PENDING = 'PENDING'
FILE_LOCK = '.lock'


def get_modelfit_results(model, path):
    # FIXME: This is a workaround. The proper solution is to only read the results.json from
    # the database. For this to work roundtrip of DataFrames in json is needed.
    # This is currently broken because of rounding issue in pandas
    # Also the modelfit_results attribute will soon be removed from model objects.
    import pharmpy.model.external.nonmem as nonmem_model
    import pharmpy.tools.external.nonmem as nonmem

    if isinstance(model, nonmem_model.Model):
        res = nonmem.parse_modelfit_results(model, path)
    else:
        import pharmpy.model.external.nlmixr as nlmixr_model
        import pharmpy.tools.external.nlmixr as nlmixr

        assert isinstance(model, nlmixr_model.Model)
        res = nlmixr.parse_modelfit_results(model, path)

    model = model.replace(modelfit_results=res)
    return model


class LocalDirectoryDatabase(NonTransactionalModelDatabase):
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
        path.mkdir(parents=True, exist_ok=True)
        self.path = path_absolute(path)
        self.file_extension = file_extension
        self.ignored_names = frozenset(('stdout', 'stderr', 'nonmem.json', 'nlmixr.json'))

    def store_model(self, model):
        pass

    def store_local_file(self, model, path, new_filename=None):
        path_object = Path(path)
        if path_object.name not in self.ignored_names and path_object.is_file():
            dest_path = self.path
            if new_filename:
                dest_path = self.path / new_filename
            shutil.copy2(path, dest_path)

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

    def retrieve_model(self, name):
        filename = name + self.file_extension
        path = self.path / filename
        from pharmpy.model import Model

        try:
            model = Model.parse_model(path)
        except FileNotFoundError:
            raise KeyError('Model cannot be found in database')
        model = get_modelfit_results(model, self.path)
        return model

    def retrieve_modelfit_results(self, name):
        return self.retrieve_model(name).modelfit_results

    def store_metadata(self, model, metadata):
        pass

    def store_modelfit_results(self, model):
        pass

    def __repr__(self):
        return f"LocalDirectoryDatabase({self.path})"


class LocalModelDirectoryDatabase(TransactionalModelDatabase):
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

    def __init__(self, path: Union[str, Path] = '.', file_extension='.mod'):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.path = path_absolute(path)
        self.file_extension = file_extension

    def _read_lock(self):
        # NOTE Obtain shared (blocking) lock on the entire database
        path = self.path / FILE_LOCK
        path.touch(exist_ok=True)
        return path_lock(str(path), shared=True)

    def _write_lock(self):
        # NOTE Obtain exclusive (blocking) lock on the entire database
        path = self.path / FILE_LOCK
        path.touch(exist_ok=True)
        return path_lock(str(path), shared=False)

    @contextmanager
    def snapshot(self, model_name: str):
        model_path = self.path / model_name
        destination = model_path / DIRECTORY_PHARMPY_METADATA
        destination.mkdir(parents=True, exist_ok=True)
        with self._read_lock():
            # NOTE Check that no pending transaction exists
            path = destination / FILE_PENDING
            if path.exists():
                # TODO finish pending transaction from journal if possible
                raise PendingTransactionError()

            yield LocalModelDirectoryDatabaseSnapshot(self, model_name)

    @contextmanager
    def transaction(self, model: Model):
        model_path = self.path / model.name
        destination = model_path / DIRECTORY_PHARMPY_METADATA
        destination.mkdir(parents=True, exist_ok=True)
        with self._write_lock():
            # NOTE Mark state as pending
            path = destination / FILE_PENDING
            try:
                path.touch(exist_ok=False)
            except FileExistsError:
                # TODO finish pending transaction from journal if possible
                raise PendingTransactionError()

            yield LocalModelDirectoryDatabaseTransaction(self, model)

            # NOTE Commit transaction (only if no exception was raised)
            path.unlink()

    def list_models(self):
        model_dir_names = [p.name for p in self.path.glob('*') if not p.name.startswith('.')]
        return sorted(model_dir_names)

    def __repr__(self):
        return f"LocalModelDirectoryDatabase({self.path})"


class LocalModelDirectoryDatabaseTransaction(ModelTransaction):
    def __init__(self, database: LocalModelDirectoryDatabase, model: Model):
        self.db = database
        self.model = model

    def store_model(self):
        from pharmpy.modeling import read_dataset_from_datainfo, write_csv, write_model

        model = self.model
        datasets_path = self.db.path / DIRECTORY_DATASETS

        # NOTE Get the hash of the dataset and list filenames with contents
        # matching this hash only
        h = hash_df_fs(model.dataset)
        h_dir = datasets_path / DIRECTORY_INDEX / h
        h_dir.mkdir(parents=True, exist_ok=True)
        for hpath in h_dir.iterdir():
            # NOTE This variable holds a string similar to "run1.csv"
            matching_model_filename = hpath.name
            data_path = datasets_path / matching_model_filename
            dipath = data_path.with_suffix('.datainfo')
            # TODO Maybe catch FileNotFoundError and similar here (pass)
            curdi = DataInfo.read_json(dipath)
            # NOTE paths are not compared here
            if curdi == model.datainfo:
                df = read_dataset_from_datainfo(curdi)
                if df.equals(model.dataset):
                    # NOTE Update datainfo path
                    datainfo = model.datainfo.replace(path=curdi.path)
                    model = model.replace(datainfo=datainfo)
                    break
        else:
            model_filename = model.name + '.csv'

            # NOTE Create the index file at .datasets/.hash/<hash>/<model_filename>
            index_path = h_dir / model_filename
            index_path.touch()

            data_path = path_absolute(datasets_path / model_filename)
            datainfo = model.datainfo.replace(path=data_path)
            model = model.replace(datainfo=datainfo)
            model = write_csv(model, path=data_path, force=True)

            # NOTE Write datainfo last so that we are "sure" dataset is there
            # if datainfo is there
            model.datainfo.to_json(datasets_path / (model.name + '.datainfo'))

        # NOTE Write the model
        model_path = self.db.path / model.name
        model_path.mkdir(exist_ok=True)
        write_model(model, str(model_path / (model.name + model.filename_extension)), force=True)
        return model

    def store_local_file(self, path, new_filename=None):
        if Path(path).is_file():
            destination = self.db.path / self.model.name
            destination.mkdir(parents=True, exist_ok=True)
            if new_filename:
                destination = destination / new_filename
            shutil.copy2(path, destination)

    def store_metadata(self, metadata):
        destination = self.db.path / self.model.name / DIRECTORY_PHARMPY_METADATA
        destination.mkdir(parents=True, exist_ok=True)
        with open(destination / FILE_METADATA, 'w') as f:
            json.dump(metadata, f, indent=2)

    def store_modelfit_results(self):
        destination = self.db.path / self.model.name / DIRECTORY_PHARMPY_METADATA
        destination.mkdir(parents=True, exist_ok=True)

        if self.model.modelfit_results:
            self.model.modelfit_results.to_json(destination / FILE_MODELFIT_RESULTS)


class LocalModelDirectoryDatabaseSnapshot(ModelSnapshot):
    def __init__(self, database: LocalModelDirectoryDatabase, model_name: str):
        self.db = database
        self.name = model_name

    def retrieve_local_files(self, destination_path):
        path = self.db.path / self.name
        files = path.glob('*')
        for f in files:
            if f.is_file():
                shutil.copy2(f, destination_path)
            else:
                shutil.copytree(f, Path(destination_path) / f.stem)

    def retrieve_file(self, filename):
        # Return path to file
        path = self.db.path / self.name / filename
        if path.is_file() and stat(path).st_size > 0:
            return path
        else:
            raise FileNotFoundError(f"Cannot retrieve {filename} for {self.name}")

    def retrieve_model(self):
        extensions = ['.mod', '.ctl']
        from pharmpy.model import Model

        errors = []
        root = self.db.path / self.name
        for extension in extensions:
            filename = self.name + extension
            path = root / filename
            try:
                # NOTE this will guess the model type
                model = Model.parse_model(path)
                break
            except FileNotFoundError as e:
                errors.append(e)
                pass
        else:
            raise KeyError(
                f'Could not find {self.name} in {self.db}.'
                f' Looked up {", ".join(map(lambda e: f"`{e.filename}`", errors))}.'
            )

        model = get_modelfit_results(model, path)
        return model

    def retrieve_modelfit_results(self):
        res = self.retrieve_model().modelfit_results
        if res is not None:
            return res

        # FIXME The following does not work because deserialization of modelfit
        # results is not generic enough. We only use it to make the resume_tool
        # test pass.
        path = self.db.path / self.name / DIRECTORY_PHARMPY_METADATA / FILE_MODELFIT_RESULTS
        return read_results(path)
