# -*- encoding: utf-8 -*-
"""
Execution Environment
=====================

Platform (e.g. Linux) & system (e.g. SLURM) implementation.

The cluster/local or OS etc. to start jobs on.

Definitions
-----------
"""

import asyncio
import logging
import os
import concurrent.futures

from .job import Job


class Environment:
    """Manages execution of an engine on a platform/system.

    Subclasses may support e.g. Windows, Linux, SLURM or SGE.
    """

    def __new__(cls, *args, **kwargs):
        if cls is Environment or cls is SystemEnvironment:
            cls = WindowsSystemEnvironment if os.name == 'nt' else PosixSystemEnvironment
        self = cls.create_new(args, init=False)
        if self is None or not self.supported:
            raise NotImplementedError("Cannot instantiate %r on your system" % (cls.__name__,))
        return self

    def submit(self, command, cwd):
        """Starts job and returns Job object."""
        raise NotImplementedError

    @property
    def supported(self):
        """True if environment supported on current platform."""
        pass

    @classmethod
    def create_new(cls, args, init=True):
        """Creates new object."""
        pass


class SystemEnvironment(Environment):
    """Manages system execution (not using SLURM or similar) of an engine on a platform."""

    def __init__(self, threads=None):
        self.pool = concurrent.futures.ThreadPoolExecutor(threads)
        self.jobs = []
        self.futures = []

    def submit(self, command, cwd):
        job = Job(command, cwd, stdout=self._stdout_handle, stderr=self._stderr_handle,
                  callback=self._callback, keepends=False)
        future = self.pool.submit(job.run)

        loop = job.queue.get()
        asyncio.get_child_watcher().attach_loop(loop)
        job.queue.task_done()

        self.jobs += [job]
        self.futures += [future]
        return job

    @classmethod
    def create_new(cls, args, init=True):
        self = object.__new__(cls)
        if init:
            self.__init__()
        return self

    def wait(self, timeout=None):
        """Blocks until all jobs are completed."""
        concurrent.futures.wait(self.futures, timeout=timeout)

    def _stdout_handle(self, line):
        print(line)
        logging.getLogger(__name__).info(line)

    def _stderr_handle(self, line):
        logging.getLogger(__name__).warning(line)

    def _callback(self, job):
        if job.rc == 0:
            logging.getLogger(__name__).info('%r finished.', job)
        else:
            logging.getLogger(__name__).error('%r error-exited, returned %d.', job, job.rc)


class PosixSystemEnvironment(SystemEnvironment):
    """Manages system execution of an engine on a Posix-like platform."""

    @property
    def supported(self):
        return (os.name != 'nt')


class WindowsSystemEnvironment(SystemEnvironment):
    """Manages system execution of an engine on a Windows platform."""

    @property
    def supported(self):
        return (os.name == 'nt')
