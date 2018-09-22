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
from asyncio.subprocess import PIPE
from contextlib import closing
import queue
from threading import Thread


async def _await_stream_lines(stream, func, loop, *args, **kwargs):
    """Awaits new line in *stream*, calling *func* (on *loop*) threadsafe."""
    while True:
        line = await stream.readline()
        if not line:
            break
        loop.call_soon_threadsafe(functools.partial(func, line, *args, **kwargs))


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

    def __init__(self, command, cwd=None, stdout=None, stderr=None, done=None):
        self.command = tuple(str(x) for x in command)
        self.wd = cwd if cwd else None
        self._callback['stdout'] = stdout
        self._callback['stderr'] = stderr
        self._callback['done'] = done

    def run(self, block=False):
        """Construct main asyncio loop and run/schedule."""

        def start_loop(loop):
            asyncio.set_event_loop(loop)
            loop.run_forever()

        if block:
            if os.name == 'nt':
                self.loop = asyncio.ProactorEventLoop()
                asyncio.set_event_loop(self.loop)
            else:
                self.loop = asyncio.get_event_loop()

            with closing(self.loop):
                self.future = asyncio.ensure_future(self._coro())
                self.loop.run_until_complete(self.future)
        else:
            self.loop = asyncio.new_event_loop()
            asyncio.get_child_watcher().attach_loop(self.loop)
            self.thread = Thread(target=start_loop, args=(self.loop,))
            self.thread.start()
            self.future = asyncio.run_coroutine_threadsafe(self._coro(), self.loop)

        self.future.add_done_callback(self._post_hook)

    @property
    def done(self):
        """True if job is done."""
        try:
            fut = self.future
        except AttributeError:
            return None
        else:
            if fut is None:
                return True
            return fut.done()

    @property
    def rc(self):
        """Return code of subprocess."""
        assert self.done
        return self.future.result()

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

    def wait(self, timeout=None):
        """Waits for job completion and returns return code."""
        return self.future.result(timeout=None)

    def iter_stream(self, stream_type, timeout=None, keepends=False):
        """Line-for-line generator for :attr:`~Job.stdout`, :attr:`~Job.stderr` or
        :attr:`~Job.output`.

        Arguments:
            stream_type: *stdout*, *stderr* or *output* (mixed stream)
            timeout: Time to wait for new lines/process completion. Non-blocking if 0.
            keepends: Whether to keep line breaks.

        .. todo:: Needs reference counting for multiple instances.
        """

        get_line = functools.partial(self._queue[stream_type].get, timeout != 0, timeout)
        print(get_line)
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

    async def _coro(self):
        """Main coroutine.

        Subprocess running/scheduling. Only called by :func:`run`.

        See `reference <https://stackoverflow.com/a/20697159>`_ of implementation.
        """

        proc = await asyncio.create_subprocess_exec(*self.command, stdout=PIPE, stderr=PIPE,
                                                    cwd=self.wd)
        await asyncio.wait([
            _await_stream_lines(proc.stdout, self._stream_handler, self.loop, 'stdout'),
            _await_stream_lines(proc.stderr, self._stream_handler, self.loop, 'stderr'),
        ])
        return_code = await proc.wait()

        callback = self._callback['done']
        if callback is not None:
            self.loop.call_soon_threadsafe(functools.partial(callback, return_code))

    def _stream_handler(self, line, stream_type):
        """Handler of all stdout/stderr lines.

        Put in queue, history and call external handler (if any).
        """

        line = line.decode(locale.getpreferredencoding(False))

        self._queue['output'].put(line)
        self._queue[stream_type].put(line)
        self._history['output'] += [line]
        self._history[stream_type] += [line]

        on_newline = self._callback[stream_type]
        if on_newline is not None:
            self.loop.call_soon_threadsafe(functools.partial(on_newline, line))

    def _post_hook(self, future):
        """Clean-up (kill loop, end queues)."""
        loop = self.loop
        if loop.is_running():
            loop.stop()
        for q in self._queue.values():
            q.put(None)
