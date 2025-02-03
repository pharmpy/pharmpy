from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, TypeVar

from ..workflow import Workflow

T = TypeVar('T')

DISPATCHERS = ('local_dask',)


class Dispatcher(ABC):
    @staticmethod
    def canonicalize_dispatcher_name(name: Optional[str]) -> str:
        if name is None:
            from pharmpy import conf

            canon_name = conf.dispatcher.lower()
        else:
            canon_name = name.lower()
        if canon_name not in DISPATCHERS:
            raise ValueError(f"Unknown dispatcher {name}")
        return canon_name

    @staticmethod
    def select_dispatcher(name: Optional[str]) -> Dispatcher:
        """Create a new dispatcher given a dispatcher name"""
        Dispatcher.canonicalize_dispatcher_name(name)
        # Currently only one type supported
        from pharmpy.workflows.dispatchers.local_dask import LocalDaskDispatcher

        dispatcher = LocalDaskDispatcher()
        return dispatcher

    @abstractmethod
    def run(self, workflow: Workflow[T], context) -> T:
        pass

    @abstractmethod
    def call_workflow(self, wf: Workflow[T], unique_name, context) -> T:
        pass

    @abstractmethod
    def abort_workflow(self) -> None:
        pass
