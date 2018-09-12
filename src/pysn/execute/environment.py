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
    """Manages execution of an engine on a platform/system.

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

    def submit(self, command, cwd):
        """Starts job and returns Job object."""
        return Job(command, cwd)

    # def argparse_options(self):
    #     """Returns CLI options for argparse for this environment."""
    #     raise NotImplementedError

    @classmethod
    def _create(cls, args, init=True):
        pass

    def _init(self):
        pass


class SystemEnvironment(Environment):
    """Manages system execution (not using SLURM or similar) of an engine on a platform."""
    pass


class PosixSystemEnvironment(SystemEnvironment):
    """Manages system execution of an engine on a Posix-like platform."""
    is_supported = (os.name != 'nt')

    @classmethod
    def _create(cls, args, init=True):
        self = object.__new__(cls)
        if init:
            self._init()
        return self


class WindowsSystemEnvironment(SystemEnvironment):
    """Manages system execution of an engine on a Windows platform."""
    is_supported = (os.name == 'nt')
