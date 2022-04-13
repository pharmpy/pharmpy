from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import ContextManager, Union

from pharmpy.model import Model, Results


class ModelTransaction(ABC):
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


class ModelSnapshot(ABC):
    @abstractmethod
    def retrieve_local_files(self, destination_path: Path) -> None:
        """Retrieve all files related to the model bound to this snapshot

        Parameters
        ----------
        destination_path : Path
            Local destination path
        """
        pass

    @abstractmethod
    def retrieve_file(self, filename: str) -> Path:
        """Retrieve one file related to the model bound to this snapshot

        Note that if the database is implemented in the local filesystem
        it is allowed to return a path directly to the file in the database.

        Parameters
        ----------
        filename : str
            Name of file

        Returns
        -------
        Path
            Path to local file
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
    def store_model(self, model: Model):
        """Store a model object

        Parameters
        ----------
        model : Model
            Pharmpy model object
        """
        pass

    @abstractmethod
    def store_local_file(self, model: Model, path: Path, new_filename: Union[str, None] = None):
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
    def store_metadata(self, model: Model, metadata: dict):
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
    def store_modelfit_results(self, model: Model):
        """Store modelfit results

        Parameters
        ----------
        model : Model
            Pharmpy model object
        """
        pass

    @abstractmethod
    def retrieve_local_files(self, model_name: str, destination_path: Path):
        """Retrieve all files related to a model

        Parameters
        ----------
        model_name : str
            Name of the model
        destination_path : Path
            Local destination path
        """
        pass

    @abstractmethod
    def retrieve_file(self, model_name: str, filename: str) -> Path:
        """Retrieve one file related to a model

        Note that if the database is implemented in the local filesystem
        it is allowed to return a path directly to the file in the database.

        Parameters
        ----------
        model_name : str
            Name of the model
        filename : str
            Name of file

        Returns
        -------
        Path
            Path to local file
        """
        pass

    @abstractmethod
    def retrieve_model(self, model_name: str) -> Model:
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
    def retrieve_modelfit_results(self, model_name: str) -> Results:
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
    def snapshot(self, model_name: str) -> ContextManager[ModelSnapshot]:
        """Creates a readable snapshot context for a given model.

        Parameters
        ----------
        model_name : str
            Name of the Pharmpy model object
        """
        pass

    @abstractmethod
    def transaction(self, model: Model) -> ContextManager[ModelTransaction]:
        """Creates a writable transaction context for a given model.

        Parameters
        ----------
        model : Model
            Pharmpy model object
        """
        pass


class NonTransactionalModelDatabase(ModelDatabase):
    @contextmanager
    def snapshot(self, model_name: str):
        yield DummySnapshot(self, model_name)

    @contextmanager
    def transaction(self, model: Model):
        yield DummyTransaction(self, model)


class DummyTransaction(ModelTransaction):
    def __init__(self, database: ModelDatabase, model: Model):
        self.db = database
        self.model = model

    def store_model(self) -> None:
        return self.db.store_model(self.model)

    def store_local_file(self, path: Path, new_filename: Union[str, None] = None) -> None:
        return self.db.store_local_file(self.model, path, new_filename)

    def store_metadata(self, metadata: dict) -> None:
        return self.db.store_metadata(self.model, metadata)

    def store_modelfit_results(self) -> None:
        return self.db.store_modelfit_results(self.model)


class DummySnapshot(ModelSnapshot):
    def __init__(self, database: ModelDatabase, model_name: str):
        self.db = database
        self.name = model_name

    def retrieve_local_files(self, destination_path: Path) -> None:
        return self.db.retrieve_local_files(self.name, destination_path)

    def retrieve_file(self, filename: str) -> Path:
        return self.db.retrieve_file(self.name, filename)

    def retrieve_model(self) -> Model:
        return self.db.retrieve_model(self.name)

    def retrieve_modelfit_results(self) -> Results:
        return self.db.retrieve_modelfit_results(self.name)


class TransactionalModelDatabase(ModelDatabase):
    def store_model(self, model: Model) -> None:
        with self.transaction(model) as txn:
            return txn.store_model()

    def store_local_file(
        self, model: Model, path: Path, new_filename: Union[str, None] = None
    ) -> None:
        with self.transaction(model) as txn:
            return txn.store_local_file(path, new_filename)

    def store_metadata(self, model: Model, metadata: dict) -> None:
        with self.transaction(model) as txn:
            return txn.store_metadata(metadata)

    def store_modelfit_results(self, model: Model) -> None:
        with self.transaction(model) as txn:
            return txn.store_modelfit_results()

    def retrieve_file(self, model_name: str, filename: str) -> Path:
        with self.snapshot(model_name) as sn:
            return sn.retrieve_file(filename)

    def retrieve_local_files(self, model_name: str, destination_path: Path) -> None:
        with self.snapshot(model_name) as sn:
            return sn.retrieve_local_files(destination_path)

    def retrieve_model(self, model_name: str) -> Model:
        with self.snapshot(model_name) as sn:
            return sn.retrieve_model()

    def retrieve_modelfit_results(self, model_name: str) -> Results:
        with self.snapshot(model_name) as sn:
            return sn.retrieve_modelfit_results()


class PendingTransactionError(Exception):
    pass
