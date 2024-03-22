import json
import shutil
from contextlib import contextmanager
from os import stat
from pathlib import Path
from typing import Union

from pharmpy.internals.fs.lock import path_lock
from pharmpy.internals.fs.path import path_absolute
from pharmpy.model import DataInfo, Model
from pharmpy.modeling import write_csv, write_model
from pharmpy.workflows.model_entry import ModelEntry
from pharmpy.workflows.results import ModelfitResults, read_results

from ..hashing import ModelHash
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

    return res


def create_model_entry(model, modelfit_results):
    # FIXME: This function is to avoid duplication of this logic, this can be removed once
    #  parent_model has been moved from Model and log has been moved from modelfit_results
    #  and each database implementation has methods for retrieving these
    if not isinstance(modelfit_results, ModelfitResults):
        modelfit_results = None
        log = None
    else:
        log = modelfit_results.log

    parent_model = model.parent_model

    return ModelEntry(model=model, modelfit_results=modelfit_results, parent=parent_model, log=log)


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
        return model

    def retrieve_modelfit_results(self, name):
        model = self.retrieve_model(name)
        return get_modelfit_results(model, self.path)

    def retrieve_model_entry(self, name):
        model = self.retrieve_model(name)
        modelfit_results = self.retrieve_modelfit_results(name)
        return create_model_entry(model, modelfit_results)

    def store_metadata(self, model, metadata):
        pass

    def store_modelfit_results(self, model):
        pass

    def store_model_entry(self, model_entry):
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
        # NOTE: Obtain shared (blocking) lock on the entire database
        path = self.path / FILE_LOCK
        path.touch(exist_ok=True)
        return path_lock(str(path), shared=True)

    def _write_lock(self):
        # NOTE: Obtain exclusive (blocking) lock on the entire database
        path = self.path / FILE_LOCK
        path.touch(exist_ok=True)
        return path_lock(str(path), shared=False)

    @contextmanager
    def snapshot(self, obj: Union[Model, ModelEntry, ModelHash]):
        key = ModelHash(obj)
        model_path = self.path / str(key)
        destination = model_path / DIRECTORY_PHARMPY_METADATA
        destination.mkdir(parents=True, exist_ok=True)
        with self._read_lock():
            # NOTE: Check that no pending transaction exists
            path = destination / FILE_PENDING
            if path.exists():
                # TODO: Finish pending transaction from journal if possible
                raise PendingTransactionError()

            yield LocalModelDirectoryDatabaseSnapshot(self, obj)

    @contextmanager
    def transaction(self, obj: Union[Model, ModelEntry, ModelHash]):
        key = ModelHash(obj)
        model_path = self.path / str(key)
        destination = model_path / DIRECTORY_PHARMPY_METADATA
        destination.mkdir(parents=True, exist_ok=True)
        with self._write_lock():
            # NOTE: Mark state as pending
            path = destination / FILE_PENDING
            try:
                path.touch(exist_ok=False)
            except FileExistsError:
                # TODO: Finish pending transaction from journal if possible
                raise PendingTransactionError()

            yield LocalModelDirectoryDatabaseTransaction(self, obj)

            # NOTE: Commit transaction (only if no exception was raised)
            path.unlink()

    def __repr__(self):
        return f"LocalModelDirectoryDatabase({self.path})"


class LocalModelDirectoryDatabaseTransaction(ModelTransaction):
    def store_model(self):
        if self.model_entry is None:
            raise ValueError("Cannot store model: No model attached to transaction")

        model = self.model_entry.model
        datasets_path = self.database.path / DIRECTORY_DATASETS

        # NOTE: Get the hash of the dataset and list filenames with contents
        # matching this hash only
        h = self.key.dataset_hash
        h_dir = datasets_path / DIRECTORY_INDEX / str(h)
        if h_dir.is_dir():
            hpath = next(h_dir.iterdir())
            # NOTE: This variable holds a string similar to "run1.csv"
            matching_model_filename = hpath.name
            data_path = datasets_path / matching_model_filename
            dipath = data_path.with_suffix('.datainfo')
            # TODO: Maybe catch FileNotFoundError and similar here (pass)
            curdi = DataInfo.read_json(dipath)
            # NOTE: Paths are not compared here
            if curdi == model.datainfo:
                datainfo = model.datainfo.replace(path=curdi.path)
                model = model.replace(datainfo=datainfo)
        else:
            h_dir.mkdir(parents=True, exist_ok=True)

            highest = 0
            for file in datasets_path.iterdir():
                name = file.name
                if name.startswith('data') and name.endswith('.csv'):
                    number = int(name[4:-4])  # Remove data and .csv
                    if number > highest:
                        highest = number

            dataset_basename = f'data{highest + 1}'
            dataset_filename = f'{dataset_basename}.csv'

            # NOTE: Create the index file at .datasets/.hash/<hash>/<dataset_filename>
            index_path = h_dir / dataset_filename
            index_path.touch()

            data_path = path_absolute(datasets_path / dataset_filename)
            datainfo = model.datainfo.replace(path=data_path)
            model = model.replace(datainfo=datainfo)
            model = write_csv(model, path=data_path, force=True)

            # NOTE: Write datainfo last so that we are "sure" dataset is there
            # if datainfo is there
            model.datainfo.to_json(datasets_path / (dataset_basename + '.datainfo'))

        # NOTE: Write the model
        model_path = self.database.path / str(self.key)
        model_path.mkdir(exist_ok=True)
        write_model(model, model_path / ("model" + model.filename_extension), force=True)
        return model

    def store_local_file(self, path, new_filename=None):
        if Path(path).is_file():
            destination = self.database.path / str(self.key)
            destination.mkdir(parents=True, exist_ok=True)
            if new_filename:
                destination = destination / new_filename
            shutil.copy2(path, destination)

    def store_metadata(self, metadata):
        destination = self.database.path / str(self.key) / DIRECTORY_PHARMPY_METADATA
        destination.mkdir(parents=True, exist_ok=True)
        with open(destination / FILE_METADATA, 'w') as f:
            json.dump(metadata, f, indent=2)

    def store_modelfit_results(self):
        destination = self.database.path / str(self.key) / DIRECTORY_PHARMPY_METADATA
        destination.mkdir(parents=True, exist_ok=True)

        modelfit_results = self.model_entry.modelfit_results

        if modelfit_results is not None:
            modelfit_results.to_json(destination / FILE_MODELFIT_RESULTS)

    def store_model_entry(self):
        if self.model_entry is None:
            raise ValueError('Transaction does not have `model_entry` attribute')
        # FIXME: Store parent
        self.store_model()
        self.store_modelfit_results()


class LocalModelDirectoryDatabaseSnapshot(ModelSnapshot):
    def retrieve_local_files(self, destination_path):
        path = self.database.path / str(self.key)
        files = path.glob('*')
        for f in files:
            if f.is_file():
                shutil.copy2(f, destination_path)
            else:
                shutil.copytree(f, Path(destination_path) / f.stem)

    def retrieve_file(self, filename):
        # Return path to file
        path = self.database.path / str(self.key) / filename
        if path.is_file() and stat(path).st_size > 0:
            return path
        else:
            raise FileNotFoundError(f"Cannot retrieve {filename} for {self.name}")

    def retrieve_model(self):
        path = self._find_full_model_path()

        # NOTE: This will guess the model type
        model = Model.parse_model(path)

        return model

    def _find_full_model_path(self):
        extensions = ('.mod', '.ctl')
        root = self.database.path / str(self.key)
        errors = []

        for extension in extensions:
            filename = "model" + extension
            path = root / filename

            if path.is_file():
                return path
            else:
                errors.append(path)
        else:
            raise KeyError(
                f'Could not find model in {self.database}.'
                f' Looked up {", ".join(map(lambda p: f"`{p}`", errors))}.'
            )

    def retrieve_modelfit_results(self):
        model = self.retrieve_model()
        path = self._find_full_model_path()
        res = get_modelfit_results(model, path)

        if res is not None:
            return res

        # FIXME: The following does not work because deserialization of modelfit
        # results is not generic enough. We only use it to make the resume_tool
        # test pass.
        path = (
            self.database.path / str(self.key) / DIRECTORY_PHARMPY_METADATA / FILE_MODELFIT_RESULTS
        )
        if path.is_file():
            return read_results(path)
        else:
            return None

    def retrieve_model_entry(self):
        model = self.retrieve_model()
        modelfit_results = self.retrieve_modelfit_results()
        return create_model_entry(model, modelfit_results)
