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
import logging
import queue
import threading
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
        self._ended = threading.Event()
        self.queue = queue.Queue()
        self.loop = None

        self.command = tuple(str(x) for x in command)
        self.wd = cwd if cwd else None
        self._callback['stdout'] = stdout if stdout else lambda line: None
        self._callback['stderr'] = stderr if stderr else lambda line: None
        self._callback['finish'] = callback if callback else lambda job: None
        self.keepends = keepends

    def _info(self, msg, *args, **kwargs):
        self._log(logging.INFO, msg, *args, **kwargs)

    def _debug(self, msg, *args, **kwargs):
        self._log(logging.DEBUG, msg, *args, **kwargs)

    def _log(self, level, msg, *args, **kwargs):
        if self._blocking:
            msg = '%r on <Thread %s>: %s' % (self, threading.current_thread().getName(), msg)
        else:
            msg = '%r: %s' % (self, msg)
        logging.getLogger("asyncio").log(level, msg, *args, **kwargs)

    def run(self, block=True):
        """Runs job & returns result.

        Wraps coroutine :func:`~Job.start` in convenient, synchronously callable, unit. Note that
        call must be main thread, or else call will block until caller has attached loop in
        :attr:`~Job.queue` (from the main thread's running event loop). Otherwise, async
        subprocesses & streams is `impossible
        <https://docs.python.org/3/library/asyncio-subprocess.html#subprocess-and-threads>`_.
        Example::

            job = Job(command=['find', '.'])
            pool = concurrent.futures.ThreadPoolExecutor(4)
            future = pool.submit(job.run, block=True)

            # blocking to let caller attach loop, which MUST happen in main thread
            loop = job.queue.get()
            asyncio.get_child_watcher().attach_loop(loop)

            # unblocking now
            job.queue.task_done()

            print(future.result())
        """

        self._blocking = block
        try:
            self.loop = asyncio.get_event_loop()
        except RuntimeError:
            self.loop = asyncio.new_event_loop()
        if os.name == 'nt':
            loop_cls = asyncio.ProactorEventLoop
            if not isinstance(self.loop, loop_cls):
                if self.loop.is_running():
                    raise RuntimeError("Running event loop not %s: %s. Required on Windows." %
                                       (loop_cls.__name__, self.loop))
                self.loop = asyncio.ProactorEventLoop()

        if threading.current_thread() == threading.main_thread():
            self._debug('On main thread. Attaching loop to watcher.')
            asyncio.get_child_watcher().attach_loop(self.loop)
        else:
            self._debug('Not main thread; Blocking while awaiting main thread to get loop...')
            self.queue.put(self.loop)
            self.queue.join()
            self._debug('Unblocking (loop processed by someone).')

        if self._blocking:
            if self.loop.is_running():
                self._debug('Loop running. Ensuring future of start().')
                result = asyncio.ensure_future(self.start(), loop=self.loop)
            else:
                self._debug('Switching & running loop until start() completes.')
                asyncio.set_event_loop(self.loop)
                result = self.loop.run_until_complete(self.start())
            self._debug('run() done. Returning result %r.', result)
            return result

        def start_loop(parent_id, loop):
            thread_id = '<Thread %s>' % threading.current_thread().getName()
            loop_id = '<loop %s>' % hex(id(loop))
            prefix = '%s of parent %s' % (thread_id, parent_id)

            logging.getLogger("asyncio").info('%s: Running %r forever.', prefix, loop_id)
            asyncio.set_event_loop(loop)
            loop.run_forever()
            logging.getLogger("asyncio").info('%s: Stopped %r.', prefix, loop_id)

        self.thread = threading.Thread(target=start_loop, args=(repr(self), self.loop))
        child_id = 'child <Thread %s>' % self.thread.getName()
        self._debug('Starting %s', child_id)
        self.thread.start()

        self._debug('Blocking until start() runs on %s.', child_id)
        future = asyncio.run_coroutine_threadsafe(self.start(), self.loop)
        self._started.wait()

        self._debug('run() done. Returning future %r.', future)
        return future

    async def start(self):
        """Starts & returns future, async on current event loop."""

        if not self.loop:
            self.loop = asyncio.get_event_loop()
        self.task = asyncio.ensure_future(self._main(), loop=self.loop)
        task_id = '<%s of coro %s()>' % (self.task.__class__.__name__, self.task._coro.__qualname__)
        self._debug('Has promised future %s.', task_id)
        self.task.add_done_callback(self._cleanup)
        return await self.task

    @property
    def started(self):
        """True if job is started."""
        try:
            return self._started.is_set()
        except AttributeError:
            raise RuntimeError('Not defined. Start method of %r not called.', self)

    @property
    def ended(self):
        """True if job is ended."""
        try:
            return self._ended.is_set()
        except AttributeError:
            raise RuntimeError('Not defined. %r has not started.', self)

    def wait(self, timeout=None):
        """Waits for job completion."""
        try:
            ended = self.thread.join(timeout=timeout)
        except AttributeError:
            ended = self._ended.wait(timeout=timeout)
        if ended:
            assert self._ended.is_set(), "%r ended but attr '_ended' not set (bad clean-up)"

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

        Subprocess running/scheduling. Only called internally.

        See `reference <https://stackoverflow.com/a/20697159>`_ of implementation.
        """

        wd = str(self.wd) if self.wd else None
        self._debug('Starting subprocess %r', self.command)
        try:
            self._proc = await asyncio.create_subprocess_exec(*self.command, stdout=PIPE,
                                                              stderr=PIPE, cwd=wd)
        except Exception as exc:
            self._warning('Subprocess raised %r', exc)
            self.loop.call_soon_threadsafe(self._started.set)
            self.loop.call_soon_threadsafe(self._ended.set)
            raise exc
        else:
            self._info('Started subprocess %r', self._proc)
            self.loop.call_soon_threadsafe(self._started.set)

        await asyncio.wait([
            self._await_stream_lines(self._proc.stdout, 'stdout'),
            self._await_stream_lines(self._proc.stderr, 'stderr'),
        ])

        return_code = await self._proc.wait()
        callback = self._callback['finish']
        self._debug('Awaited subprocess %r. Callback: %r.', self._proc, callback)
        callback = functools.partial(callback, self)
        self.loop.call_soon_threadsafe(callback)
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

        self.loop.call_soon_threadsafe(functools.partial(self._callback[stream_type], line))

    def _cleanup(self, task):
        """Clean-up (kill thread loop, end queues)."""
        if hasattr(self, 'thread') and self.loop.is_running():
            self.loop.stop()
        for q in self._queue.values():
            q.put(None)

    def __repr__(self):
        self_id = hex(id(self))
        loop_id = hex(id(self.loop)) if self.loop else None
        if self._ended.is_set():
            status = 'DONE'
        elif self._started.is_set():
            status = 'RUN'
        else:
            status = 'NEW'
        return '<%s %s %s loop %s>' % (status, self.__class__.__name__, self_id, loop_id)

    async def _await_stream_lines(self, stream_reader, type):
        """Awaits new line in *stream_reader*, calling handler threadsafely."""
        self._debug('Reading from (%s) %r', type, stream_reader)
        while True:
            line = await stream_reader.readline()
            if line:
                self._debug('Read line from (%s) %r: %r', type, stream_reader, line)
                self.loop.call_soon_threadsafe(functools.partial(self._stream_handler, line, type))
            else:
                self._debug('Read end of (%s) %r', type, stream_reader)
                break
