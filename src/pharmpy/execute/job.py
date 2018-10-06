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
import logging
import os
import queue
import threading
# import contextlib
from asyncio.subprocess import PIPE


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
        callback: Call, when task completes.
        keepends: Keep line breaks for individual lines?
    """

    _callback = dict(
        stdout=None,
        stderr=None,
        finish=None,
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

    def __init__(self, command, cwd=None, stdout=None, stderr=None, callback=None, keepends=False):
        self._started = threading.Event()
        self._done = threading.Event()

        self.queue = queue.Queue()
        self.command = tuple(str(x) for x in command)
        self.working_dir = cwd if cwd else None
        self.keepends = keepends

        self._callback['stdout'] = stdout if stdout else lambda line: None
        self._callback['stderr'] = stderr if stderr else lambda line: None
        self._callback['finish'] = callback if callback else lambda job: None

    @property
    def loop(self):
        """Current event loop. Create new if none."""
        if os.name == 'nt':
            asyncio.set_event_loop_policy(asyncio. WindowsProactorEventLoopPolicy())
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        if os.name == 'nt':
            assert isinstance(loop, asyncio.ProactorEventLoop), 'bad event loop for Windows'
        return loop

    async def run(self):
        """Runs job asynchronously.

        Wraps coroutine :func:`~Job.start` in convenient, synchronously callable, unit. Note that
        call will block until caller has attached loop in :attr:`~Job.queue`, from the main thread's
        running event loop, if not calling from main thread. Otherwise, async subprocesses & streams
        is `impossible
        <https://docs.python.org/3/library/asyncio-subprocess.html#subprocess-and-threads>`_.
        Example::

            job = Job(command=['find', '.'])
            pool = concurrent.futures.ThreadPoolExecutor(4)
            future = pool.submit(job.run)

            # blocking to let caller attach loop, which MUST happen in main thread
            loop = job.queue.get()
            asyncio.get_child_watcher().attach_loop(loop)

            # unblocking now
            job.queue.task_done()

            print(future.result())
        """

        if threading.current_thread() == threading.main_thread():
            self._debug('Attaching loop to watcher')
            asyncio.get_child_watcher().attach_loop(self.loop)
        else:
            self.queue.put(self.loop)
            self._debug('Waiting on main thread to attach loop')
            self.queue.join()

        self.task = asyncio.ensure_future(self._main(), loop=self.loop)
        self.task.add_done_callback(self._cleanup)

        self._debug('Awaiting _main() future')
        return await self.task

    def thread(self):
        """Runs job in new thread."""

        if threading.current_thread() != threading.main_thread():
            self._debug('Thread not main thread. Executing loop here.')
            loop = self.loop
            assert not loop.is_running()
            return loop.run_until_complete(self.run())

        def thread(loop):
            asyncio.set_event_loop(loop)
            loop.run_forever()

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=thread, args=(self._loop,))
        self._thread.start()

        self._debug('Waiting on loop from new <Thread %s>', self._thread.getName())
        future = asyncio.run_coroutine_threadsafe(self.run(), loop=self._loop)
        loop = self.queue.get()
        asyncio.get_child_watcher().attach_loop(loop)
        self.queue.task_done()

        self._debug('Waiting until new <Thread %s> has started job', self._thread.getName())
        self._started.wait()
        return future

    @property
    def started(self):
        """True if job is started."""
        return self._started.is_set()

    @property
    def done(self):
        """True if job is done."""
        try:
            return self._done.is_set()
        except AttributeError:
            raise RuntimeError('%r has not started!', self)

    def wait(self, timeout=None):
        """Waits for job to complete."""
        self._done.wait(timeout=timeout)
        # try:
        #     self._thread.join(timeout=timeout)
        # except AttributeError:
        #     pass
        return self.rc

    @property
    def output(self):
        """Current, non-linesplit stdout/stderr (mixed) history."""
        with threading.Lock():
            return ''.join(self._history['output'])

    @property
    def stdout(self):
        """Current, non-linesplit stdout history."""
        with threading.Lock():
            return ''.join(self._history['stdout'])

    @property
    def stderr(self):
        """Current, non-linesplit stderr history."""
        with threading.Lock():
            return ''.join(self._history['stderr'])

    @property
    def proc(self):
        """Started subprocess :class:`Process` object."""
        with threading.Lock():
            return self._proc

    @property
    def rc(self):
        """Return code of subprocess."""
        with threading.Lock():
            return self.proc.returncode

    @property
    def pid(self):
        """PID of subprocess."""
        with threading.Lock():
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

    def _warn(self, msg, *args, **kwargs):
        self._log(logging.WARNING, msg, *args, **kwargs)

    def _info(self, msg, *args, **kwargs):
        self._log(logging.INFO, msg, *args, **kwargs)

    def _debug(self, msg, *args, **kwargs):
        self._log(logging.DEBUG, msg, *args, **kwargs)

    def _log(self, level, msg, *args, **kwargs):
        with threading.Lock():
            msg = '%s: %r on <Thread %s>' % (msg, self, threading.current_thread().getName())
            logging.getLogger(__name__).log(level, msg, *args, **kwargs)

    async def _main(self):
        """Main coroutine.

        Subprocess running/scheduling. Only called internally.

        See `reference <https://stackoverflow.com/a/20697159>`_ of implementation.
        """

        if self.working_dir:
            working_dir = str(self.working_dir)
        else:
            working_dir = None

        self._debug('Starting subprocess %r', self.command)
        try:
            proc = await asyncio.create_subprocess_exec(*self.command, stdout=PIPE, stderr=PIPE,
                                                        cwd=working_dir)
        except Exception as exc:
            self._warn('Subprocess raised exception %r', exc)
            self._started.set()
            self._done.set()
            raise exc
        else:
            with threading.Lock():
                self._proc = proc
            self._info('Started subprocess %r', self._proc)
            self._started.set()

        await asyncio.gather(self._await_stream_lines(self._proc.stdout, 'stdout'),
                             self._await_stream_lines(self._proc.stderr, 'stderr'))

        return_code = await self._proc.wait()
        self._debug('Awaited subprocess %r', self._proc)

        callback = functools.partial(self._callback['finish'], self)
        asyncio.get_event_loop().call_soon_threadsafe(callback)
        self._done.set()
        return return_code

    def _stream_handler(self, line, stream_type):
        """Handler of all stdout/stderr lines.

        Put in queue, history and call external handler (if any).
        """

        line = line.decode(locale.getpreferredencoding(False))
        with threading.Lock():
            self._history['output'] += [line]
            self._history[stream_type] += [line]

        if not self.keepends:
            line = ''.join(line.splitlines())
        self._queue['output'].put(line)
        self._queue[stream_type].put(line)

        callback = functools.partial(self._callback[stream_type], line)
        asyncio.get_event_loop().call_soon_threadsafe(callback)

    async def _await_stream_lines(self, stream_reader, stream_type):
        """Awaits new line in *stream_reader*, calling handler threadsafely."""
        while True:
            line = await stream_reader.readline()
            if line:
                self._debug('Read from %s', stream_type.upper())
                callback = functools.partial(self._stream_handler, line, stream_type)
                asyncio.get_event_loop().call_soon_threadsafe(callback)
            else:
                self._debug('Read end of %s', stream_type.upper())
                return stream_reader.close()

    def _cleanup(self, task=None):
        """Clean-up (stop loop if started, end queues)."""
        try:
            self._loop.call_soon_threadsafe(self._loop.stop)
        except AttributeError:
            pass
        for q in self._queue.values():
            q.put(None)

    def __repr__(self):
        hid = hex(id(self))
        if self._done.is_set():
            return '<done %s %s>' % (self.__class__.__name__, hid)
        elif self._started.is_set():
            task = '<%s %s>' % (self.task._state.lower(), self.task.__class__.__name__)
            return '<busy %s %s, %s, %s>' % (self.__class__.__name__, hid, task, self.proc)
        else:
            return '<new %s %s, %r>' % (self.__class__.__name__, hid, self.command)
