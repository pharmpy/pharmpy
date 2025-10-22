from __future__ import annotations

import os
import signal
from abc import ABC, abstractmethod
from typing import NoReturn, Optional, TypeVar

from ..workflow import Workflow
from .slurm_helpers import is_running_on_slurm

T = TypeVar('T')

DISPATCHERS = ('local_dask', 'local_serial')


class AbortWorkflowException(Exception):
    pass


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

    def canonicalize_ncores(self, ncores: Optional[int]) -> int:
        if ncores and ncores > 1:
            if is_running_on_slurm():
                raise ValueError(
                    f'Invalid `ncores`: must be 1 or None when running on slurm, got {ncores}'
                )
        if not ncores:
            hosts = self.get_hosts()
            ncores = sum(hosts.values())
        return ncores

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
    def run(self, workflow: Workflow[T], context) -> Optional[T]:
        pass

    @abstractmethod
    def call_workflow(self, wf: Workflow[T], unique_name: str, context) -> T:
        pass

    @abstractmethod
    def abort_workflow(self) -> NoReturn:
        # It is not safe to call this more than once per dispatched workflow
        # This is the responsibility of the Context
        raise AbortWorkflowException

    @abstractmethod
    def get_hosts(self) -> dict[str, int]:
        pass

    @abstractmethod
    def get_available_cores(self, allocation: int) -> int:
        pass

    @abstractmethod
    def get_hostname(self) -> str:
        pass


class SigHandler:
    def __init__(self, context):
        self.context = context

    def __enter__(self):
        def sigint_handler(sig, frame):
            self.context.abort_workflow("Workflow was interrupted by user (SIGINT)")

        def sigterm_handler(sig, frame):
            self.context.abort_workflow("Workflow was terminated (SIGTERM)")

        def sighup_handler(sig, frame):
            # If the calling shell dies but we are still alive
            # we want to block SIGPIPE in case we are writing
            # to the now broken stdout. This way we could
            # still be able to run the workflow to completion
            signal.signal(signal.SIGPIPE, signal.SIG_IGN)

        signal.signal(signal.SIGINT, sigint_handler)
        signal.signal(signal.SIGTERM, sigterm_handler)
        if os.name != 'nt':
            # Windows doesn't recognize the SIGHUP signal
            signal.signal(signal.SIGHUP, sighup_handler)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        signal.signal(signal.SIGTERM, signal.SIG_DFL)

        if os.name != 'nt':
            # Windows doesn't recognize the SIGHUP signal
            signal.signal(signal.SIGHUP, signal.SIG_DFL)
