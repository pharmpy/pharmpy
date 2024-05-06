from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal, Optional, Union

from pharmpy.deps import pandas as pd
from pharmpy.model import Model
from pharmpy.workflows.hashing import ModelHash
from pharmpy.workflows.model_database import ModelDatabase
from pharmpy.workflows.model_entry import ModelEntry
from pharmpy.workflows.results import Results

FINAL_MODEL_NAME = 'final'
INPUT_MODEL_NAME = 'input'


class Context(ABC):
    """Context for runs

    A database of results, metadata and run files for one tool run

    Parameters
    ----------
    name : str
        Name of the context
    ref : str
        A reference (path) to the context
    """

    def __init__(self, name: str, ref: Optional[str] = None, common_options: dict[str, Any] = None):
        # If the context already exists it will be opened
        # otherwise a new top level context will be created
        # An implementation needs to create the model database here
        # If ref is None an implementation specific default ref will be used
        self._name = name

    @property
    def model_database(self) -> ModelDatabase:
        """ModelDatabase to store results of models run in context"""
        return self._model_database

    @staticmethod
    @abstractmethod
    def exists(name: str, ref: Optional[str] = None) -> bool:
        pass

    @abstractmethod
    def store_results(self, res: Results):
        """Store tool results

        Parameters
        ----------
        res : Results
            Tool results object
        """
        pass

    @abstractmethod
    def retrieve_results(self) -> Results:
        """Retrieve tool results

        Return
        ------
        Results
            Tool results object
        """

    @abstractmethod
    def store_metadata(self, metadata: dict):
        """Store tool metadata

        Parameters
        ----------
        metadata : dict
            Tool metadata dictionary
        """
        pass

    @abstractmethod
    def retrieve_metadata(self) -> dict:
        """Read tool metadata"""
        pass

    @abstractmethod
    def store_key(self, name: str, key: ModelHash):
        """Associate a key with a model name"""
        pass

    @abstractmethod
    def retrieve_key(self, name: str) -> ModelHash:
        """Retrive the key corresponding to a model name

        This key can be used to lookup the model in the model database
        """
        pass

    @abstractmethod
    def list_all_names(self) -> list(str):
        """Retrieve a list of all model names in the context"""
        pass

    @abstractmethod
    def list_all_subcontexts(self) -> list(str):
        """Retrieve a list of the names of all subcontexts in the context"""
        pass

    @abstractmethod
    def store_annotation(self, name: str, annotation: str):
        """Store an annotation string (description) for a model"""
        pass

    @abstractmethod
    def retrieve_annotation(self, name: str) -> str:
        """Retrieve an annotation for a model"""
        pass

    @abstractmethod
    def log_message(self, severity: Literal["error", "warning", "note"], message: str):
        """Add a message to the log"""
        pass

    @abstractmethod
    def retrieve_log(self, level: Literal['all', 'current', 'lower'] = 'all') -> pd.DataFrame:
        """Retrieve the entire log
        all - the entire log
        current - only the current Context level
        lower - current and sub levels
        """
        pass

    @abstractmethod
    def retrieve_common_options(self) -> dict[str, Any]:
        pass

    @abstractmethod
    def get_parent_context(self) -> Context:
        """Get the parent context of this context"""
        pass

    @abstractmethod
    def get_subcontext(self, name: str) -> Context:
        """Get one of the subcontexts of this context"""
        pass

    @abstractmethod
    def create_subcontext(self, name: str) -> Context:
        """Create a new subcontext of this context"""
        pass

    def _store_model(self, name: str, model: Union[Model, ModelEntry]):
        db = self.model_database
        with db.transaction(model) as txn:
            txn.store_model_entry()
            key = txn.key
        self.store_key(name, key)
        annotation = model.description if isinstance(model, Model) else model.model.description
        self.store_annotation(name, annotation)

    def _retrieve_me(self, name: str) -> ModelEntry:
        db = self.model_database
        key = self.retrieve_key(name)
        me = db.retrieve_model_entry(key)
        model = me.model
        annotation = self.retrieve_annotation(name)
        model = model.replace(name=name, description=annotation)
        new_me = ModelEntry(
            model=model, parent=me.parent, modelfit_results=me.modelfit_results, log=me.log
        )
        return new_me

    def store_model_entry(self, me: Union[Model, ModelEntry]) -> None:
        name = me.name if isinstance(me, Model) else me.model.name
        self._store_model(name, me)

    def retrieve_model_entry(self, name: str) -> ModelEntry:
        me = self._retrieve_me(name)
        return me

    def store_input_model_entry(self, me: Union[Model, ModelEntry]) -> None:
        self._store_model(INPUT_MODEL_NAME, me)

    def retrieve_input_model_entry(self) -> ModelEntry:
        me = self._retrieve_me(INPUT_MODEL_NAME)
        return me

    def store_final_model_entry(self, me: ModelEntry) -> None:
        self._store_model(FINAL_MODEL_NAME, me)

    def retrieve_final_model_entry(self) -> ModelEntry:
        model = self._retrieve_me(FINAL_MODEL_NAME)
        return model
