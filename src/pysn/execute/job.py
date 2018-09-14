# -*- encoding: utf-8 -*-
"""
============
Executed Job
============

A job unit.

Contains process-wrapping Job for non-blocking execution and communication.

.. note::
    :class:`Job` instance created by :class:`~pysn.execute.engine.Engine` implementation and already
    started before external API access gets access.

Definitions
-----------
"""

import asyncio
import locale
import os
from asyncio.subprocess import PIPE
from contextlib import closing


async def _read_stream(stream, callback):
    while True:
        line = await stream.readline()
        if not line:
            break
        callback(line)


class AsyncProcess:
    """An asynchronous process.

    A process launched asynchronously with stdout and stderr streams. Meant to be wrapped in
    :class:`Job` and never bare.

    Arguments:
        command: An iterable of the program and arguments to launch.
        cwd: The working directory to launch the process in.
        stdout: Optional handler for callback on stdout line receive.
        stderr: Optional handler for callback on stderr line receive.
    """

    _queue = dict(stdout=asyncio.Queue(), stderr=asyncio.Queue())
    _history = dict(stdout=[], stderr=[])
    _callback = dict(stdout=None, stderr=None)

    def __init__(self, command, cwd, stdout=None, stderr=None):
        self.command = tuple(str(x) for x in command)
        self.cwd = str(cwd)
        self._callback['stdout'] = stdout
        self._callback['stderr'] = stderr

    def run(self):
        """Construct main asyncio loop and launch. Returns returncode."""
        if os.name == 'nt':
            loop = asyncio.ProactorEventLoop()
            asyncio.set_event_loop(loop)
        else:
            loop = asyncio.get_event_loop()
        with closing(loop):
            rc = loop.run_until_complete(self._stream_subprocess())
        return rc

    @property
    def stdout(self):
        """(Current) stdout history."""
        return self._history['stdout']

    @property
    def stderr(self):
        """(Current) stderr history."""
        return self._history['stderr']

    def iter_stdout(self, block=True):
        """Returns generator for stdout.

        Arguments:
            block: If True and empty queue: Block until new line (or process completion).

        .. warning:: Does not work yet. I don't know why!
        """
        while True:
            try:
                line = self._queue['stdout'].get_nowait()
            except asyncio.QueueEmpty:
                if block:
                    continue
                else:
                    return
            yield line

    def iter_stderr(self, block=True):
        """Returns generator for stderr.

        Arguments:
            block: If True and empty queue: Block until new line (or process completion).

        .. warning:: Does not work yet. I don't know why!
        """
        while True:
            try:
                line = self._queue['stderr'].get_nowait()
            except asyncio.QueueEmpty:
                if block:
                    continue
                else:
                    return
            yield line

    def _stdout_handler(self, line):
        """Handle one stdout line: Put in queue, history and call external handler (if any)."""
        line = line.decode(locale.getpreferredencoding(False))
        self._queue['stdout'].put(line)
        self._history['stdout'] += [line]
        if self._callback['stdout'] is not None:
            self._callback['stdout'](line)

    def _stderr_handler(self, line):
        """Handle one stderr line: Put in queue, history and call external handler (if any)."""
        line = line.decode(locale.getpreferredencoding(False))
        self._queue['stderr'].put(line)
        self._history['stderr'] += [line]
        if self._callback['stderr'] is not None:
            self._callback['stderr'](line)

    async def _stream_subprocess(self):
        """The asynchronous coroutine. Bound by event loop in :func:`~self.run`."""
        # ref: https://stackoverflow.com/a/20697159
        proc = await asyncio.create_subprocess_exec(*self.command, stdout=PIPE, stderr=PIPE,
                                                    cwd=self.cwd)
        await asyncio.wait([
            _read_stream(proc.stdout, self._stdout_handler),
            _read_stream(proc.stderr, self._stderr_handler)
        ])
        return await proc.wait()


class Job:
    """A job of an Engine (running in an Environment).

    Responsible for generating and wrapping an :class:`AsyncProcess` object for asynchronous
    execution, status control and interface with stdout/stderr streams.

    Arguments:
        command: An iterable of the program and arguments to launch.
        cwd: The working directory to launch the process in.

    .. todo:: Add callback argument for job completion."""

    rc = None
    """Returncode of process wrapped by Job (or None if not completed)."""

    def __init__(self, command, cwd):
        self.process = AsyncProcess(command, cwd)
        self.rc = self.process.run()
