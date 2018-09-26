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
        self.jobs = []
        self.futures = []
        self.pool = concurrent.futures.ThreadPoolExecutor(threads)

    @property
    def log(self):
        return logging.getLogger('%s.%s' % (__name__, self.__class__.__name__))

    @property
    def loop(self):
        on_main_thread = (threading.current_thread() == threading.main_thread())
        assert on_main_thread, ("%s can only execute in main thread!" % self.__class__.__name__)
        return asyncio.get_event_loop()

    @property
    def watcher(self):
        return asyncio.get_child_watcher()

    async def submit(self, command, cwd):
        """Submits *command* to run as subprocess with *cwd* working directory."""
        logger = logging.getLogger(__name__)

        job = Job(command, cwd,  stdout=self._stdout_handle, stderr=self._stderr_handle,
                  callback=self._callback, keepends=False)

        future = asyncio.wrap_future(self.pool.submit(job.run))
        logger.debug('Submitting job %r, future %r: %r', job, future, self)

        self.jobs += [job]
        self.futures += [future]

        logger.debug('Awaiting event loop (to watch) from job %r: %r', job, self)
        loop = await job.queue.get()
        logger.debug('Received loop %r, from job %r: %r.', loop, job, self)

        self.watcher.attach_loop(loop)
        job.queue.task_done()

    async def wait(self):
        """Blocks until all jobs are completed."""
        logger = logging.getLogger(__name__)

        logger.info('Awaiting futures %r: %r', self.futures, self)
        for future in self.futures:
            await future

    @classmethod
    def create_new(cls, args, init=True):
        self = object.__new__(cls)
        if init:
            self.__init__()
        return self

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
