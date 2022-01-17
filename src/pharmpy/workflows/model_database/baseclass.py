from abc import ABC, abstractmethod


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
    def store_local_file(self, model, path):
        """Store a file from the local machine

        Parameters
        ----------
        model : Model
            Pharmpy model object
        path : Path
            Path to file
        """
        pass

    @abstractmethod
    def retrieve_local_files(self, name, destination_path):
        """Retrieve all files related to a model

        Parameters
        ----------
        name : str
            Name of the model
        destination_path : Path
            Local destination path
        """
        pass

    @abstractmethod
    def retrieve_file(self, name, filename):
        """Retrieve one file related to a model

        Note that if the database is implemented in the local filesystem
        it is allowed to return a path directly to the file in the database.

        Parameters
        ----------
        name : str
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
    def get_model(self, name):
        """Read a model from the database

        Parameters
        ----------
        name : str
            Name of the model

        Returns
        -------
        Model
            Retrieved model object
        """
        pass
