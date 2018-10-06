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
import threading
import concurrent.futures

from .job import Job


class Environment:
    """Manages execution of an engine on a platform/system.

    Subclasses may support e.g. Windows, Linux, SLURM or SGE.
    """

    def __new__(cls, *args, **kwargs):
        if cls is Environment:
            cls = SystemEnvironment
        self = cls.__new__(*args, **kwargs)
        if self is None or not self.supported:
            raise NotImplementedError("Cannot instantiate %r on your system" % (cls.__name__,))
        else:
            return self

    def submit(self, command, cwd):
        """Starts job and returns Job object."""
        raise NotImplementedError

    @property
    def supported(self):
        """True if environment supported on current platform."""
        pass


class SystemEnvironment(Environment):
    """Manages system execution (not using SLURM or similar) of an engine on a platform."""

    def __new__(cls, *args, **kwargs):
        if cls is SystemEnvironment:
            if os.name == 'nt':
                cls = WindowsSystemEnvironment
            else:
                cls = PosixSystemEnvironment
        self = object.__new__(cls)
        self.__init__(*args, **kwargs)
        return self

    def __init__(self, threads=None):
        self.jobs = list()
        self.futures = list()
        self.pool = concurrent.futures.ThreadPoolExecutor(threads)
        self._watcher = asyncio.get_child_watcher()

    async def submit(self, command, cwd=None):
        """Submits *command* to run as subprocess with *cwd* working directory."""
        logger = logging.getLogger(__name__)

        job = Job(command, cwd,  stdout=self._stdout_handle, stderr=self._stderr_handle,
                  callback=self._callback, keepends=False)
        future = asyncio.wrap_future(self.pool.submit(job.thread))
        logger.debug('Submitted job %r: future %r on %r', job, future, self)

        self.jobs += [job]
        self.futures += [future]

        logger.debug('Awaiting event loop (to watch) from job %r: %r', job, self)
        loop = job.queue.get()
        self._watcher.attach_loop(loop)
        job.queue.task_done()

    def close(self):
        """Stop accepting new jobs and wait for all to complete."""
        logger = logging.getLogger(__name__)
        for job in self.jobs:
            if not job.done:
                logger.info('Awaiting job %r: %r', job, self)
                job.wait(1)
        self.pool.shutdown(wait=True)
        self.pool = None

    @classmethod
    def _new(cls, *args, **kwargs):
        self = object.__new__(cls)
        return self

    @property
    def log(self):
        return logging.getLogger('%s.%s' % (__name__, self.__class__.__name__))

    @property
    def _loop(self):
        if threading.current_thread() != threading.main_thread():
            raise AssertionError("%s must only execute in main thread." % self.__class__.__name__)
        else:
            return asyncio.get_event_loop()

    def _stdout_handle(self, line):
        self.log.info(line)

    def _stderr_handle(self, line):
        self.log.warning(line)

    def _callback(self, job):
        if job.rc == 0:
            self.log.info('Finished job %r: %r', job, self)
        else:
            self.log.error('Failed (rc=%d) job %r: %r', job.rc, job, self)

    def __repr__(self):
        return '%s(%d)' % (self.__class__.__name__, self.pool._max_workers)


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

    @property
    def loop(self):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        return super()._init_loop()
