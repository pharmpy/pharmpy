from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Literal, Optional, Union

from pharmpy.deps import numpy as np
from pharmpy.deps import pandas as pd
from pharmpy.model import Model
from pharmpy.workflows.broadcasters import Broadcaster
from pharmpy.workflows.dispatchers import Dispatcher
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

    def __init__(
        self,
        name: str,
        ref: Optional[str] = None,
    ):
        # If the context already exists it will be opened
        # otherwise a new top level context will be created
        # An implementation needs to create the model database here
        # If ref is None an implementation specific default ref will be used
        self._name = name
        self._ref = ref

    @abstractmethod
    def __repr__(self) -> str:
        pass

    @property
    def name(self) -> str:
        return self._name

    @property
    def ref(self) -> str:
        return self._ref

    @property
    def model_database(self) -> ModelDatabase:
        """ModelDatabase to store results of models run in context"""
        return self._model_database

    @property
    def broadcaster(self) -> Broadcaster:
        if not hasattr(self, '_broadcaster'):
            options = self.retrieve_dispatching_options()
            if 'broadcaster' in options:
                name = options['broadcaster']
            else:
                from pharmpy import conf

                name = conf.broadcaster
            self._broadcaster = Broadcaster.select_broadcaster(name)
        return self._broadcaster

    @property
    def dispatcher(self) -> Dispatcher:
        if not hasattr(self, '_dispatcher'):
            options = self.retrieve_dispatching_options()
            if 'dispatcher' in options:
                name = options['dispatcher']
            else:
                from pharmpy import conf

                name = conf.dispatcher
            self._dispatcher = Dispatcher.select_dispatcher(name)
        return self._dispatcher

    @property
    def seed(self) -> int:
        if not hasattr(self, '_seed'):
            self._seed = self.retrieve_metadata()['seed']
        return self._seed

    @property
    @abstractmethod
    def context_path(self) -> str:
        pass

    def get_model_context_path(self, model: Model) -> str:
        ctxpath = f"{self.context_path}/@{model.name}"
        return ctxpath

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
        pass

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

    def get_ncores_for_execution(self):
        """Get number of cores for execution (using available cores among allocation)"""
        ncores = self.retrieve_dispatching_options()['ncores']
        return self.dispatcher.get_available_cores(ncores)

    @abstractmethod
    def store_message(self, severity, ctxpath: str, date, message: str):
        pass

    def log_message(
        self,
        severity: Literal["critical", "error", "warning", "info", "trace"],
        message: str,
        model: Optional[Model] = None,
    ):
        """Add a message to the log"""
        date = datetime.now()
        if model is None:
            ctxpath = self.context_path
        else:
            ctxpath = self.get_model_context_path(model)
        self.store_message(severity, ctxpath, date, message)
        self.broadcaster.broadcast_message(severity, ctxpath, date, message)

    def log_info(self, message: str, model: Optional[Model] = None):
        """Add an info message to the log

        Currently with echo to stdout. In the future this could be changed or be configurable.
        """
        self.log_message(severity="info", message=message, model=model)

    def log_error(self, message: str, model: Optional[Model] = None):
        """Add an error message to the log"""
        self.log_message(severity="error", message=message, model=model)

    def log_warning(self, message: str, model: Optional[Model] = None):
        """Add a warning message to the log"""
        self.log_message(severity="warning", message=message, model=model)

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
    def retrieve_dispatching_options(self) -> dict[str, Any]:
        pass

    @abstractmethod
    def get_parent_context(self) -> Context:
        """Get the parent context of this context"""
        pass

    @abstractmethod
    def get_top_level_context(self) -> Context:
        """Get the top level context of this context"""
        pass

    @abstractmethod
    def get_subcontext(self, name: str) -> Context:
        """Get one of the subcontexts of this context"""
        pass

    @abstractmethod
    def create_subcontext(self, name: str) -> Context:
        """Create a new subcontext of this context"""
        pass

    @abstractmethod
    def finalize(self):
        """Called after a tool has finished its run in a context
        can be implemented to do cleanup of the context
        """
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
        """Retrieve the ModelEntry of the model marked as input"""
        me = self._retrieve_me(INPUT_MODEL_NAME)
        return me

    def store_final_model_entry(self, me: ModelEntry) -> None:
        self._store_model(FINAL_MODEL_NAME, me)

    def retrieve_final_model_entry(self) -> ModelEntry:
        """Retrieve the ModelEntry of the model marked as final"""
        me = self._retrieve_me(FINAL_MODEL_NAME)
        return me

    def call_workflow(self, workflow, unique_name: str):
        """Ask the dispatcher to call a subworkflow"""
        res = self.dispatcher.call_workflow(workflow, unique_name, self)
        return res

    def abort_workflow(self, message):
        """Ask the dispatcher to abort the currently running workflow directly"""
        self.log_message("critical", message)
        self.dispatcher.abort_workflow()

    def has_started(self):
        """Check if the tool running in the context has started"""
        metadata = self.retrieve_metadata()
        return "stats" in metadata and "start_time" in metadata["stats"]

    def has_completed(self):
        """Check if the tool running in the context has completed"""
        metadata = self.retrieve_metadata()
        return "stats" in metadata and "end_time" in metadata["stats"]

    def create_rng(self, index: int):
        """Create a random number generator

        Creating the generator will be using the seed common option, the index and
        the context path to get a unique sequence.
        """
        ctxpath_bytes = bytes(self.context_path, encoding="utf-8")
        rng = np.random.default_rng([index, ctxpath_bytes, self.seed])
        return rng

    def spawn_seed(self, rng, n=128) -> int:
        """Spawn a new seed using a random number generator

        Parameters
        ----------
        rng : Random number generator
            Random number generator
        n : int
            Size of seed to generate in number of bits

        Returns
        -------
        int
            New random seed
        """
        n_full_words = n // 64
        a = rng.integers(2**64 - 1, size=n_full_words, dtype=np.uint64)
        x = 0
        m = 1
        for val in a:
            x += int(val) * m
            m *= 2**64
        remaining_bits = n % 64
        if remaining_bits > 0:
            b = rng.integers(2**remaining_bits - 1, size=1, dtype=np.uint64)
            x += int(b[0]) * m
        return x
