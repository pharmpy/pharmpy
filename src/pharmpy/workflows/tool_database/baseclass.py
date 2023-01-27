from abc import ABC, abstractmethod
from pathlib import Path


class ToolDatabase(ABC):
    """Database of results, metadata and run files for one tool run

    Parameters
    ----------
    toolname : str
        Name of the tool to use the database
    """

    def __init__(self, toolname):
        self.toolname = toolname

    @property
    def path(self):
        """ModelDatabase to store results of models run by tool"""
        return self._path

    @path.setter
    def path(self, value: Path):
        self._path = value

    @property
    def model_database(self):
        """ModelDatabase to store results of models run by tool"""
        return self._model_database

    @model_database.setter
    def model_database(self, value):
        self._model_database = value

    @abstractmethod
    def store_local_file(self, source_path):
        """Store a local file in the database

        Parameters
        ----------
        source_path : str or Path
            Path to local file
        """
        pass

    @abstractmethod
    def store_results(self, res):
        """Store tool results

        Parameters
        ----------
        res : Results
            Tool results object
        """
        pass

    @abstractmethod
    def store_metadata(self, metadata):
        """Store tool metadata

        Parameters
        ----------
        metadata : dict
            Tool metadata dictionary
        """
        pass

    @abstractmethod
    def read_metadata(self):
        """Read tool metadata"""
        pass
