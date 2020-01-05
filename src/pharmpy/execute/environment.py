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
import concurrent.futures
import logging
import os

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
    """Manages system (direct subprocess) execution for an engine on some platform.

    Needs :mod:`asyncio` because a running event loop is required to monitor child processes.
    Execution machinery via :mod:`concurrent.futures` interface.

    Attributes:
        jobs: Submitted jobs.
        futures: Future promises of jobs, instances of :mod:`concurrent.futures.Future`.
        pool: Threading pool, instance of :class:`~concurrent.Futures.ThreadPoolExecutor`.
    """

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

    async def submit(self, command, cwd=None):
        """Submits *command* to run as subprocess in *cwd* working directory."""

        logger = logging.getLogger(__name__)

        asyncio.get_child_watcher()
        job = Job(command, cwd,  stdout=self._stdout_handle, stderr=self._stderr_handle,
                  callback=self._callback, keepends=False)

        future = self.pool.submit(job.run)
        logger.debug('Submitted job %r (future: %r): %r', job, future, self)

        self.jobs += [job]
        self.futures += [future]
        return job

    async def wait(self, timeout=None, poll=1):
        """Wait (block) for all jobs to complete (and stop accepting new ones).

        Arguments:
            timeout: Maximum wait time in seconds.
            poll: Wait between each status check of job.
        """

        coros = [job.wait(timeout, poll) for job in self.jobs]
        await asyncio.gather(*coros)
        self.pool.shutdown()
        self.pool = None

    @property
    def log(self):
        return logging.getLogger('%s.%s' % (__name__, self.__class__.__name__))

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
        return '<%s %s>' % (self.__class__.__name__, hex(id(self)))


class PosixSystemEnvironment(SystemEnvironment):
    """:class:`SystemEnvironment` on Linux/MacOS."""

    @property
    def supported(self):
        return (os.name != 'nt')


class WindowsSystemEnvironment(SystemEnvironment):
    """:class:`SystemEnvironment` on Windows."""

    def __init__(self, *args, **kwargs):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        return super().__init__(*args, **kwargs)

    @property
    def supported(self):
        return (os.name == 'nt')
