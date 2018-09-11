# -*- encoding: utf-8 -*-
"""
Execution Environment
=====================

Platform (e.g. Linux) & system (e.g. SLURM) implementation.

The cluster/local or OS etc. to start jobs on.

Definitions
-----------
"""

import os

from .job import Job


class Environment:
    """Manages the execution of the engine on a platform/system.

    Subclasses may support e.g. Windows, Linux, SLURM or SGE.
    """

    def __new__(cls, *args, **kwargs):
        if cls is Environment or cls is SystemEnvironment:
            cls = WindowsSystemEnvironment if os.name == 'nt' else PosixSystemEnvironment
        self = cls._create(args, init=False)
        if self is None:
            raise NotImplementedError("Cannot instantiate %r on your system" % (cls.__name__,))
        self._init()
        return self

    def submit(self, *commands):
        """Starts job, possibly composed of several processes, and returns Job object.

        Each argument is (iterable of) arguments for separate process."""
        return Job(*commands)

    # def argparse_options(self):
    #     """Returns CLI options for argparse for this environment."""
    #     raise NotImplementedError

    @classmethod
    def _create(cls, args, init=True):
        pass

    def _init(self):
        pass


class SystemEnvironment(Environment):
    """Manages the system execution (not using SLURM or similar) of the engine on a platform."""
    pass


class PosixSystemEnvironment(SystemEnvironment):
    """Manages the system execution of the engine on a Posix-like platform."""
    is_supported = (os.name != 'nt')

    @classmethod
    def _create(cls, args, init=True):
        self = object.__new__(cls)
        if init:
            self._init()
        return self


class WindowsSystemEnvironment(SystemEnvironment):
    """Manages the system execution of the engine on a Windows platform."""
    is_supported = (os.name == 'nt')
