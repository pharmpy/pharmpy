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
from asyncio.subprocess import PIPE
from contextlib import closing


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

    def run(self, block=True):
        """Runs job & returns result.

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

        if os.name == 'nt':
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        if not block:
            self._loop = asyncio.new_event_loop()
            self._thread = threading.Thread(target=self.run, args=(True,))
            self._thread.start()

            future = asyncio.run_coroutine_threadsafe(self.start(), self._loop)
            if threading.current_thread() == threading.main_thread():
                self._debug('On main thread; Attaching loop to watcher')
                asyncio.get_child_watcher().attach_loop(self.queue.get())
                self.queue.task_done()
            self._debug('Blocking until new <Thread %s> running start()', self._thread.getName())
            self._started.wait()

            self._debug('Returning start() future (running in <Thread %s>)', self._thread.getName())
            return future
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if threading.current_thread() == threading.main_thread():
            self._debug('On main thread; Attaching loop to watcher')
            asyncio.get_child_watcher().attach_loop(loop)
        else:
            self._debug('Waiting on main thread (to attach loop)')
            self.queue.put(loop)
            self.queue.join()

        if loop.is_running():
            self._debug('Loop already running; Returning start() future')
            return asyncio.ensure_future(self.start(), loop=loop)
        else:
            self._debug('Starting new loop, to return result of start()')
            return loop.run_until_complete(self.start())

    async def start(self):
        """Starts job async on current event loop."""

        self.task = asyncio.ensure_future(self._main())
        self.task.add_done_callback(self._cleanup)
        try:
            return await self.task
        except KeyboardInterrupt:
            pass
        finally:
            self._cleanup()
        return None

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
        try:
            done = self._thread.join(timeout=timeout)
        except AttributeError:
            done = self._done.wait(timeout=timeout)
        if done and not self._done.is_set():
            self._warn("Thread done but event not set (bad cleanup)!")

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
        msg = '%s: %r on <Thread %s>' % (msg, self, threading.current_thread().getName())
        logging.getLogger("asyncio").log(level, msg, *args, **kwargs)

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
            asyncio.get_event_loop().call_soon_threadsafe(self._started.set)
            asyncio.get_event_loop().call_soon_threadsafe(self._done.set)
            raise exc
        else:
            with threading.Lock():
                self._proc = proc
            self._info('Started subprocess %r', self._proc)
            asyncio.get_event_loop().call_soon_threadsafe(self._started.set)

        await asyncio.wait([
            self._await_stream_lines(self._proc.stdout, 'stdout'),
            self._await_stream_lines(self._proc.stderr, 'stderr'),
        ])

        return_code = await self._proc.wait()
        self._debug('Awaited subprocess %r', self._proc)

        callback = functools.partial(self._callback['finish'], self)
        asyncio.get_event_loop().call_soon_threadsafe(callback)
        asyncio.get_event_loop().call_soon_threadsafe(self._done.set)
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

    async def _await_stream_lines(self, stream_reader, type):
        """Awaits new line in *stream_reader*, calling handler threadsafely."""
        self._debug('Reading from (%s) %r', type, stream_reader)
        while True:
            # line = await stream_reader.readline()
            line = None
            if line:
                self._debug('Read line from (%s) %r: %r', type, stream_reader, line)
                callback = functools.partial(self._stream_handler, line, type)
                asyncio.get_event_loop().call_soon_threadsafe(callback)
            else:
                self._debug('Read end of (%s) %r', type, stream_reader)
                break

    def _cleanup(self, task=None):
        """Clean-up (kill thread loop, end queues)."""
        try:
            self._loop.call_soon_threadsafe(self._loop.stop)
        except AttributeError:
            pass
        for q in self._queue.values():
            q.put(None)

    def __repr__(self):
        if self._done.is_set():
            status = 'done'
        elif self._started.is_set():
            status = 'busy'
        else:
            status = 'new'
        try:
            task = ' task %s' % self.task
        except AttributeError:
            task = ''
        return '<%s %s %s%s>' % (status, self.__class__.__name__, hex(id(self)), task)
