# -*- encoding: utf-8 -*-
"""
============
Executed Job
============

A job unit, for non-blocking execution and communication (of subprocesses)

.. todo:: Implement support for multiple in paralell, via :func:`asyncio.gather`.

Definitions
-----------
"""

import asyncio
import functools
import locale
import os
import queue
import threading
from asyncio.subprocess import PIPE
from contextlib import closing


async def _await_stream_lines(stream, func, loop, *args, **kwargs):
    """Awaits new line in *stream*, calling *func* (on *loop*) threadsafe."""
    while True:
        line = await stream.readline()
        if line:
            loop.call_soon_threadsafe(functools.partial(func, line, *args, **kwargs))
        else:
            break


class Job:
    """A job of an :class:`~pharmpy.execute.Engine` running in a
    :class:`~pharmpy.execute.RunDirectory` (on an :class:`~pharmpy.execute.Environment`).

    Responsible for generating and wrapping a task, and providing interfaces to stdout/stderr
    streams and result.

    Arguments:
        command: Program and arguments (iterable).
        cwd: Optional working directory to change to.
        stdout: Call, for each line of stdout.
        stderr: Call, for each line of stderr.
        done: Call, when task completes.
        keepends: Keep line breaks for individual lines?
    """

    _callback = dict(
        stdout=None,
        stderr=None,
        done=None,
    )

    _queue = dict(
        output=queue.Queue(),
        stdout=queue.Queue(),
        stderr=queue.Queue(),
    )

    _history = dict(
        output=list(),
        stdout=list(),
        stderr=list(),
    )

    def __init__(self, command, cwd=None, stdout=None, stderr=None, done=None, keepends=False):
        self.command = tuple(str(x) for x in command)
        self.wd = cwd if cwd else None
        self._callback['stdout'] = stdout
        self._callback['stderr'] = stderr
        self._callback['done'] = done
        self.keepends = keepends

    def start(self):
        """Starts job in a new thread (in other loop). Returns task."""

        self._started = threading.Event()
        self._ended = threading.Event()

        self.loop = asyncio.new_event_loop()
        asyncio.get_child_watcher().attach_loop(self.loop)

        def start_loop(loop):
            asyncio.set_event_loop(loop)
            loop.run_forever()

        self.thread = threading.Thread(target=start_loop, args=(self.loop,))
        self.thread.start()

        self.task = asyncio.run_coroutine_threadsafe(self._main(), self.loop)
        self.task.add_done_callback(self._done_callback)
        self._started.wait()

        return self.task

    async def start_loop(self, loop):
        """Starts job async on set event loop. Returns task."""

        self._started = asyncio.Event()
        self._ended = asyncio.Event()

        if loop:
            asyncio.set_event_loop(loop)
        self.loop = asyncio.get_event_loop()

        self.task = asyncio.ensure_future(self._main(), loop=self.loop)
        self.task.add_done_callback(self._done_callback)
        await self.task

        return self.task

    @property
    def started(self):
        """True if job is started."""
        try:
            return self._started.is_set()
        except AttributeError:
            return None

    @property
    def ended(self):
        """True if job is ended."""
        try:
            return self._ended.is_set()
        except AttributeError:
            return None

    def wait(self):
        """Waits for job completion."""
        return self._ended.wait()

    @property
    def output(self):
        """Current, non-linesplit stdout/stderr (mixed) history."""
        return ''.join(self._history['output'])

    @property
    def stdout(self):
        """Current, non-linesplit stdout history."""
        return ''.join(self._history['stdout'])

    @property
    def stderr(self):
        """Current, non-linesplit stderr history."""
        return ''.join(self._history['stderr'])

    @property
    def proc(self):
        """Started subprocess ``Process`` object."""
        return self._proc

    @property
    def rc(self):
        """Return code of subprocess."""
        return self.proc.returncode

    @property
    def pid(self):
        """PID of subprocess."""
        return self.proc.pid

    def iter_stream(self, stream_type, timeout=0, keepends=None):
        """Line-for-line generator for :attr:`~Job.stdout`, :attr:`~Job.stderr` or
        :attr:`~Job.output` of only non-processed history.

        Arguments:
            stream_type: *stdout*, *stderr* or *output* (mixed stream)
            timeout: Time to wait for new lines/process completion. Non-blocking if 0.
            keepends: Whether to keep line breaks (default: set at init).

        .. todo:: Needs reference counting for multiple instances.
        """

        keepends = self.keepends if keepends is None else keepends
        get_line = functools.partial(self._queue[stream_type].get, timeout != 0, timeout)
        while True:
            try:
                line = get_line()
            except queue.Empty:
                return
            if line is None:
                return
            if keepends:
                yield line
            else:
                yield ''.join(line.splitlines())

    async def _main(self):
        """Main coroutine.

        Subprocess running/scheduling. Only called by :func:`run`.

        See `reference <https://stackoverflow.com/a/20697159>`_ of implementation.
        """

        self._proc = await asyncio.create_subprocess_exec(*self.command, stdout=PIPE, stderr=PIPE,
                                                         cwd=self.wd)
        self.loop.call_soon_threadsafe(self._started.set)

        await asyncio.wait([
            _await_stream_lines(self._proc.stdout, self._stream_handler, self.loop, 'stdout'),
            _await_stream_lines(self._proc.stderr, self._stream_handler, self.loop, 'stderr'),
        ])

        return_code = await self._proc.wait()

        callback = self._callback['done']
        if callback is not None:
            self.loop.call_soon_threadsafe(functools.partial(callback, return_code))

        self.loop.call_soon_threadsafe(self._ended.set)
        return return_code

    def _stream_handler(self, line, stream_type):
        """Handler of all stdout/stderr lines.

        Put in queue, history and call external handler (if any).
        """

        line = line.decode(locale.getpreferredencoding(False))
        self._history['output'] += [line]
        self._history[stream_type] += [line]

        if not self.keepends:
            line = ''.join(line.splitlines())
        self._queue['output'].put(line)
        self._queue[stream_type].put(line)

        on_newline = self._callback[stream_type]
        if on_newline is not None:
            self.loop.call_soon_threadsafe(functools.partial(on_newline, line))

    def _done_callback(self, task):
        """Clean-up (kill loop, end queues)."""
        if hasattr(self, 'thread') and self.loop.is_running():
            self.loop.stop()
        for q in self._queue.values():
            q.put(None)
