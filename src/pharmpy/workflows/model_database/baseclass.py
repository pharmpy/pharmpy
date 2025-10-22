from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import ContextManager, Union

from pharmpy.model import Model

from ...workflows.model_entry import ModelEntry
from ..hashing import ModelHash
from ..results import Results


class ModelTransaction(ABC):
    def __init__(self, database: ModelDatabase, obj: Union[Model, ModelEntry, ModelHash]):
        self.database = database
        if isinstance(obj, ModelEntry):
            self.model_entry = obj
            self.key = ModelHash(obj.model)
        elif isinstance(obj, Model):
            self.model_entry = ModelEntry.create(obj)
            self.key = ModelHash(obj)
        elif isinstance(obj, ModelHash):
            self.model_entry = None
            self.key = obj
        else:
            raise ValueError(
                f'Invalid type `model_or_model_entry`: got {type(obj)}, expected Model or ModelEntry'
            )

    @abstractmethod
    def store_model(self) -> None:
        """Store the model object bound to this transaction"""
        pass

    @abstractmethod
    def store_local_file(self, path: Path, new_filename: Union[str, None] = None):
        """Store a file from the local machine for the model bound to this
        transaction

        Parameters
        ----------
        path : Path
            Path to file
        new_filename: str|None
            Filename to give to the file. Optional, defaults to original
            filename given by path.
        """
        pass

    @abstractmethod
    def store_metadata(self, metadata) -> None:
        """Store metadata for the model bound to this transaction

        Parameters
        ----------
        metadata : Dict
            A dictionary with metadata
        """
        pass

    @abstractmethod
    def store_modelfit_results(self) -> None:
        """Store modelfit results of the model bound to this transaction"""
        pass

    @abstractmethod
    def store_model_entry(self) -> None:
        """Store model entry of the model entry bound to this transaction"""
        pass


class ModelSnapshot(ABC):
    def __init__(self, database: ModelDatabase, model: Union[Model, ModelHash]):
        self.database = database
        self.key = ModelHash(model)

    @abstractmethod
    def list_all_files(self) -> list[str]:
        """Lists all file names related to a model run bound to this snapshot

        Note that this will not return a list of full paths, only the name of the files

        Returns
        -------
        list[str]
            List of file names
        """
        pass

    @abstractmethod
    def retrieve_file(self, filename: str, destination_path: Path, force: bool = False) -> None:
        """Retrieve one file related to a model run bound to this snapshot

        Parameters
        ----------
        filename : str
            Name of file
        destination_path : Path
            Destination path
        force : bool
            Force overwrite of file (default False)
        """
        pass

    @abstractmethod
    def retrieve_all_files(self, destination_path: Path, force: bool = False) -> None:
        """Retrieve all files related to a model run bound to this snapshot

        Parameters
        ----------
        destination_path : Path
            Destination path
        force : bool
            Force overwrite of files (default False)
        """
        pass

    @abstractmethod
    def retrieve_model(self) -> Model:
        """Get the model bound to this snapshot

        Returns
        -------
        Model
            Retrieved model object
        """
        pass

    @abstractmethod
    def retrieve_modelfit_results(self) -> Results:
        """Read modelfit results from the database

        Parameters
        ----------
        name : str
            Name of the model

        Returns
        -------
        Results
            Retrieved model results object
        """
        pass

    @abstractmethod
    def retrieve_model_entry(self) -> ModelEntry:
        """Read model entry from the database

        Returns
        -------
        ModelEntry
            Retrieved model entry object
        """
        pass


class ModelDatabase(ABC):
    """Baseclass for databases for models and results of model runs

    Primary key is the model name, which means that models with
    the same name will be counted as the same model.

    Currently the model database is centered around storing and retrieving
    files belonging to the model or results from fitting the model. This
    doesn't mean that implementations have to use a file system to store the
    files.
    """

    @abstractmethod
    def store_model(self, model: Model) -> None:
        """Store a model object

        Parameters
        ----------
        model : Model
            Pharmpy model object
        """
        pass

    @abstractmethod
    def store_local_file(
        self, model: Union[Model, ModelHash], path: Path, new_filename: Union[str, None] = None
    ) -> None:
        """Store a file from the local machine

        Parameters
        ----------
        model : Model
            Pharmpy model object
        path : Path
            Path to file
        new_filename: str|None
            Filename to give to the file. Optional, defaults to original
            filename given by path.
        """
        pass

    @abstractmethod
    def store_metadata(self, model: Union[Model, ModelHash], metadata: dict) -> None:
        """Store metadata

        Parameters
        ----------
        model : Model
            Pharmpy model object
        metadata : Dict
            A dictionary with metadata
        """
        pass

    @abstractmethod
    def store_modelfit_results(self, model: Union[Model, ModelHash]) -> None:
        """Store modelfit results

        Parameters
        ----------
        model : Model
            Pharmpy model object
        """
        pass

    @abstractmethod
    def store_model_entry(self, model_entry: ModelEntry) -> None:
        """Store model entry of the model entry bound to this transaction

        Parameters
        ----------
        model_entry : ModelEntry
            Pharmpy ModelEntry object
        """
        pass

    @abstractmethod
    def list_all_files(self, model: Union[Model, ModelHash]) -> list[str]:
        """Lists all file names related to a model run

        Note that this will not return a list of full paths, only the name of the files

        Parameters
        ----------
        model : Model, ModelHash
            Model object or ModelHash

        Returns
        -------
        list[str]
            List of file names
        """
        pass

    @abstractmethod
    def retrieve_file(
        self,
        model: Union[Model, ModelHash],
        filename: str,
        destination_path: Path,
        force: bool = False,
    ) -> None:
        """Retrieve one file related to a model run

        Parameters
        ----------
        model : Model, ModelHash
            Model object or ModelHash
        filename : str
            Name of file
        destination_path : Path
            Destination path
        force : bool
            Force overwrite of file (default False)
        """
        pass

    @abstractmethod
    def retrieve_all_files(
        self, model: Union[Model, ModelHash], destination_path: Path, force: bool = False
    ) -> None:
        """Retrieve all files related to a model run

        Parameters
        ----------
        model : Model, ModelHash
            Model object or ModelHash
        destination_path : Path
            Destination path
        force : bool
            Force overwrite of files (default False)
        """
        pass

    @abstractmethod
    def retrieve_model(self, model: ModelHash) -> Model:
        """Read a model from the database

        Parameters
        ----------
        model_name : str
            Name of the model

        Returns
        -------
        Model
            Retrieved model object
        """
        pass

    @abstractmethod
    def retrieve_modelfit_results(self, model: Union[Model, ModelHash]) -> Results:
        """Read modelfit results from the database

        Parameters
        ----------
        model_name : str
            Name of the model

        Returns
        -------
        Results
            Retrieved model results object
        """
        pass

    @abstractmethod
    def retrieve_model_entry(self, model: Union[Model, ModelHash]) -> ModelEntry:
        """Read model entry from the database

        Parameters
        ----------
        model_name : str
            Name of the model

        Returns
        -------
        ModelEntry
            Retrieved model entry object
        """
        pass

    @abstractmethod
    def snapshot(self, model: Union[Model, ModelHash]) -> ContextManager[ModelSnapshot]:
        """Creates a readable snapshot context for a given model.

        Parameters
        ----------
        model_name : str
            Name of the Pharmpy model object
        """
        pass

    @abstractmethod
    def transaction(
        self, obj: Union[Model, ModelEntry, ModelHash]
    ) -> ContextManager[ModelTransaction]:
        """Creates a writable transaction context for a given model.

        Parameters
        ----------
        obj : Model | ModelEntry | ModelHash
            Pharmpy model, ModelEntry or ModelHash object
        """
        pass


class NonTransactionalModelDatabase(ModelDatabase):
    @contextmanager
    def snapshot(self, model: Union[Model, ModelHash]):
        yield DummySnapshot(self, model)

    @contextmanager
    def transaction(self, obj: Union[Model, ModelEntry, ModelHash]):
        yield DummyTransaction(self, obj)


class DummyTransaction(ModelTransaction):
    def __init__(self, database: ModelDatabase, obj: Union[Model, ModelEntry, ModelHash]):
        self.database = database
        if isinstance(obj, ModelEntry):
            self.model_entry = obj
        elif isinstance(obj, Model):
            self.model_entry = ModelEntry.create(obj)
        else:
            raise ValueError(
                f'Invalid type `obj`: got {type(obj)}, expected Model, ModelEntry ' 'or ModelHash'
            )

    def store_model(self) -> None:
        return self.database.store_model(self.model_entry.model)

    def store_local_file(self, path: Path, new_filename: Union[str, None] = None) -> None:
        return self.database.store_local_file(self.model_entry.model, path, new_filename)

    def store_metadata(self, metadata: dict) -> None:
        return self.database.store_metadata(self.model_entry.model, metadata)

    def store_modelfit_results(self) -> None:
        return self.database.store_modelfit_results(self.model_entry.model)

    def store_model_entry(self) -> None:
        return self.database.store_model_entry(self.model_entry)


class DummySnapshot(ModelSnapshot):
    def __init__(self, database: ModelDatabase, model: Union[Model, ModelHash]):
        self.database = database
        self.key = ModelHash(model)

    def list_all_files(self) -> list[str]:
        return self.database.list_all_files(self.key)

    def retrieve_file(self, filename: str, destination_path: Path, force: bool = False) -> None:
        self.database.retrieve_file(self.key, filename, destination_path, force)

    def retrieve_all_files(self, destination_path: Path, force: bool = False) -> None:
        self.database.retrieve_all_files(self.key, destination_path, force)

    def retrieve_model(self) -> Model:
        return self.database.retrieve_model(self.key)

    def retrieve_modelfit_results(self) -> Results:
        return self.database.retrieve_modelfit_results(self.key)

    def retrieve_model_entry(self) -> ModelEntry:
        return self.database.retrieve_model_entry(self.key)


class TransactionalModelDatabase(ModelDatabase):
    def store_model(self, model: Union[Model, ModelEntry]) -> None:
        with self.transaction(model) as txn:
            return txn.store_model()

    def store_local_file(
        self,
        model: Union[Model, ModelEntry, ModelHash],
        path: Path,
        new_filename: Union[str, None] = None,
    ) -> None:
        with self.transaction(model) as txn:
            return txn.store_local_file(path, new_filename)

    def store_metadata(self, model: Union[Model, ModelEntry, ModelHash], metadata: dict) -> None:
        with self.transaction(model) as txn:
            return txn.store_metadata(metadata)

    def store_modelfit_results(self, model: Union[Model, ModelEntry]) -> None:
        with self.transaction(model) as txn:
            return txn.store_modelfit_results()

    def store_model_entry(self, model_entry: ModelEntry) -> None:
        with self.transaction(model_entry) as txn:
            return txn.store_model_entry()

    def list_all_files(self, obj: Union[Model, ModelEntry, ModelHash]) -> list[str]:
        with self.snapshot(obj) as sn:
            return sn.list_all_files()

    def retrieve_file(
        self,
        obj: Union[Model, ModelEntry, ModelHash],
        filename: str,
        destination_path: Path,
        force: bool = False,
    ) -> None:
        with self.snapshot(obj) as sn:
            sn.retrieve_file(filename, destination_path, force)

    def retrieve_all_files(
        self, obj: Union[Model, ModelEntry, ModelHash], destination_path, force: bool = False
    ) -> None:
        with self.snapshot(obj) as sn:
            sn.retrieve_all_files(destination_path, force)

    def retrieve_model(self, obj: Union[Model, ModelEntry, ModelHash]) -> Model:
        with self.snapshot(obj) as sn:
            return sn.retrieve_model()

    def retrieve_modelfit_results(self, obj: Union[Model, ModelEntry, ModelHash]) -> Results:
        with self.snapshot(obj) as sn:
            return sn.retrieve_modelfit_results()

    def retrieve_model_entry(self, obj: Union[Model, ModelEntry, ModelHash]) -> ModelEntry:
        with self.snapshot(obj) as sn:
            return sn.retrieve_model_entry()


class PendingTransactionError(Exception):
    pass
