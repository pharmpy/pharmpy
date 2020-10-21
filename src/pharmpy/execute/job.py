# -*- encoding: utf-8 -*-
"""
============
Executed Job
============

A job unit, for non-blocking execution and communication (of subprocesses).

Can be used raw for running subprocesses asynchronously.

Definitions
-----------
"""

import asyncio
import functools
import locale
import logging
import os
import queue
import threading
import time
from asyncio.subprocess import PIPE


class Status(threading.Event):
    """A started/done status of a :class:`~Job` instance."""

    def __bool__(self):
        return self.is_set()

    def __repr__(self):
        return '<%s %r>' % (self.__class__.__name__, bool(self))


class Job(threading.Thread):
    """A job of an :class:`~pharmpy.execute.Engine` running in a
    :class:`~pharmpy.execute.RunDirectory` (via an :class:`~pharmpy.execute.Environment`).

    Responsible for generating and wrapping a subprocess for execution and monitoring. Inherits from
    :class:`~threading.Thread` to run in separate thread (non-blocking monitoring). Call
    :func:`~Job.start` to execute (:func:`~Job.run` considered low-level API).

    Args:
        command: Program and arguments (iterable).
        cwd: Working directory.
        stdout: Callback, for each line of stdout.
        stderr: Callback, for each line of stderr.
        callback: Callback, for when task completes.
        keepends: Keep line breaks for individual lines?

    .. note:: Callback functions (arguments *stdout*, *stderr* and *callback*) must be threadsafe.
    """

    def __init__(self, command, cwd=None, stdout=None, stderr=None, callback=None, keepends=False):
        threading.Thread.__init__(self)

        self.init = Status()  #: True if job has started. :class:`Status` instance.
        self.done = Status()  #: True if job is done. :class:`Status` instance.

        self.command = tuple(str(x) for x in command)
        self.working_dir = cwd if cwd else None
        self.keepends = keepends

        self._callback = dict(stdout=stdout, stderr=stderr, done=callback)
        self._queue = dict(output=queue.Queue(), stdout=queue.Queue(), stderr=queue.Queue())
        self._history = dict(output=[], stdout=[], stderr=[])

    def run(self):
        """Starts execution *in current thread*.

        .. note:: You probably want :func:`~Job.start`.
        """

        if os.name == 'nt':
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        try:
            self.loop = asyncio.get_event_loop()
        except RuntimeError:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
        if os.name == 'nt':
            assert isinstance(self.loop, asyncio.ProactorEventLoop), 'bad event loop for Windows'

        returncode = self.loop.run_until_complete(self._main())

        self._debug('Finishing job')
        for q in self._queue.values():
            q.put(None)
        return returncode

    async def wait(self, timeout=None, poll=1):
        """Awaits job completion.

        Arguments:
            timeout: Maximum wait time in seconds.
            poll: Wait between each status check of job.

        Raises:
            TimeoutError: Job did not complete in *timeout* seconds.

        Use ``None`` for *timeout* to wait forever. Polling impossible to avoid with Python's GIL
        (don't event try)!
        """

        if (timeout and timeout < 0) or (poll < 0):
            raise ValueError("arguments (timeout, poll) not positive numbers (seconds)")

        def fmt_sec(seconds):
            if not seconds:
                return 'infinite'
            elif seconds < 1:
                return '%.3f msec' % round(seconds * 1000, 3)
            elif seconds < 60:
                return '%.3f sec' % round(seconds, 3)
            else:
                m, s = divmod(seconds, 60)
                h, m = divmod(m, 60)
                return '%d:%02d:%02d' % (h, m, s)

        start_time = time.time()
        self._debug('Waiting (timeout: %s) on job (poll: %s)', fmt_sec(timeout), fmt_sec(poll))
        while not self.done:
            seconds = time.time() - start_time
            if (timeout is not None) and (seconds > timeout):
                msg = 'Timed out (%s > %s) while waiting on job' % (
                    fmt_sec(seconds),
                    fmt_sec(timeout),
                )
                self._log(logging.ERROR, '%s, terminating %r' % (msg, self.proc))
                self.proc.terminate()
                raise TimeoutError(msg)
            await asyncio.sleep(poll)

        self._info('Waited %s on job, now joining thread', fmt_sec(time.time() - start_time))
        if self.is_alive():
            self.join()

    @property
    def output(self):
        """Current, non-linesplit stdout/stderr (mixed) history."""
        if not self.proc:
            return None
        with threading.Lock():
            return ''.join(self._history['output'])

    @property
    def stdout(self):
        """Current, non-linesplit stdout history."""
        if not self.proc:
            return None
        with threading.Lock():
            return ''.join(self._history['stdout'])

    @property
    def stderr(self):
        """Current, non-linesplit stderr history."""
        if not self.proc:
            return None
        with threading.Lock():
            return ''.join(self._history['stderr'])

    @property
    def rc(self):
        """Return code of subprocess."""
        if not self.proc:
            return None
        with threading.Lock():
            return self.proc.returncode

    @property
    def pid(self):
        """PID of subprocess."""
        if not self.proc:
            return None
        with threading.Lock():
            return self.proc.pid

    @property
    def proc(self):
        """Started subprocess. :class:`Process` object."""
        with threading.Lock():
            try:
                return self._proc
            except AttributeError:
                return None

    @proc.setter
    def proc(self, value):
        with threading.Lock():
            self._proc = value

    def iter_stream(self, stream_type, timeout=0, keepends=None):
        """Line-for-line generator for :attr:`~Job.stdout`, :attr:`~Job.stderr` or
        :attr:`~Job.output` of non-processed lines in queue.

        Arguments:
            stream_type: *stdout*, *stderr* or *output* (mixed stream)
            timeout: Time to wait for new lines/process completion. Non-blocking if 0.
            keepends: Keep line breaks?

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

    def _warn(self, msg, *args):
        self._log(logging.WARNING, msg % args)

    def _info(self, msg, *args):
        self._log(logging.INFO, msg % args)

    def _debug(self, msg, *args):
        self._log(logging.DEBUG, msg % args)

    def _log(self, level, msg):
        with threading.Lock():
            logging.getLogger(__name__).log(
                level, '%s: %r on <Thread %s>', msg, self, threading.current_thread().getName()
            )

    async def _main(self):
        """Main coroutine.

        Subprocess running/scheduling. Only called internally.

        See `reference <https://stackoverflow.com/a/20697159>`_ of implementation.
        """

        if self.working_dir:
            working_dir = str(self.working_dir)
        else:
            working_dir = None

        try:
            self._debug('Starting subprocess %r', self.command)
            self.proc = await asyncio.create_subprocess_exec(
                *self.command, stdout=PIPE, stderr=PIPE, cwd=working_dir
            )
        except Exception as exc:
            self._warn('Job failed to start! Process raised <%r>', exc)
            self.init.set()
            self.done.set()
            raise exc
        else:
            self._info('Job %r started', self.proc)
            self.init.set()

        await asyncio.gather(
            self._await_stream_lines(self.proc.stdout, 'stdout'),
            self._await_stream_lines(self.proc.stderr, 'stderr'),
        )
        self._debug('%r pipes closed', self.proc)

        return_code = await self.proc.wait()
        if return_code == 0:
            self._info('Job %r exited normally', self.proc)
        else:
            self._warn('Job %r error exited (rc=%d)', self.proc, return_code)

        if self._callback['done']:
            self._debug('Scheduling callback (to %r)', self._callback['done'])
            callback = functools.partial(self._callback['done'], self)
            self.loop.call_soon_threadsafe(self.loop.run_in_executor, None, callback)
            # await loop.run_in_executor(None, callback)
            # asyncio.run_coroutine_threadsafe(callback, loop=self.loop)

        self.done.set()
        return return_code

    def _stream_handle(self, line, name):
        """Handler of all stdout/stderr lines.

        Put in queue, history and call external handler (if any).
        """

        line = line.decode(locale.getpreferredencoding(False))
        with threading.Lock():
            self._history['output'] += [line]
            self._history[name] += [line]

        if not self.keepends:
            line = ''.join(line.splitlines())
        self._queue['output'].put(line)
        self._queue[name].put(line)

        if self._callback[name]:
            callback = functools.partial(self._callback[name], line)
            asyncio.get_event_loop().call_soon_threadsafe(callback)

    async def _await_stream_lines(self, reader, name):
        """Awaits new line in *reader*, calling handler threadsafely."""
        while True:
            line = await reader.readline()
            if not line:
                break
            self._debug('Read %r [%s]', line, name.upper())
            callback = functools.partial(self._stream_handle, line, name)
            asyncio.get_event_loop().call_soon_threadsafe(callback)
        self._debug('Read EOF [%s]', name.upper())

    def __repr__(self):
        with threading.Lock():
            name = '%s (%s)' % (hex(id(self)), self.getName())
            if self.done.is_set():
                return '<done %s %s>' % (self.__class__.__name__, name)
            elif self.init.is_set():
                return '<busy %s %s, %s>' % (self.__class__.__name__, name, self.proc)
            else:
                return '<new %s %s>' % (self.__class__.__name__, name)
