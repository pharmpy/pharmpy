from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, TypeVar

from ..workflow import Workflow

T = TypeVar('T')

DISPATCHERS = ('local_dask', 'local_serial')


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
        dispatcher_name = Dispatcher.canonicalize_dispatcher_name(name)

        if dispatcher_name == 'local_dask':
            from pharmpy.workflows.dispatchers.local_dask import LocalDaskDispatcher

            dispatcher = LocalDaskDispatcher()
        else:
            from pharmpy.workflows.dispatchers.local_serial import LocalSerialDispatcher

            dispatcher = LocalSerialDispatcher()
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

    @abstractmethod
    def get_hosts(self) -> dict[str, int]:
        pass

    def get_available_cores(self, allocation: int):
        pass

    @abstractmethod
    def get_hostname(self) -> str:
        pass
