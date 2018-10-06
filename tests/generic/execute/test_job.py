# -*- encoding: utf-8 -*-
"""
.. todo:: Important! Exception handling of threaded processes! Main thread MUST raise again.
"""

import asyncio
import concurrent.futures
from functools import partial

import pytest

from pharmpy.execute.job import Job


def test_job_run(py_command_slow, py_output_slow, event_loop):
    """Test running job synchronously."""

    def callback(job):
        assert job.rc == 0

    def line(stream_type, line):
        assert line == py_output_slow[stream_type].pop(0)

    job = Job(py_command_slow, stdout=partial(line, 'stdout'),
              stderr=partial(line, 'stderr'), callback=callback)

    result = event_loop.run_until_complete(job.run())
    assert isinstance(result, int)
    assert job.started
    assert job.done
    assert job.rc == result == 0
    assert job.proc.pid > 0


@pytest.mark.asyncio
async def test_job_async(py_command_slow, py_output_slow):
    """Test running job asynchronously."""

    def callback(job):
        assert job.rc == 0

    def line(stream_type, line):
        assert line == py_output_slow[stream_type].pop(0)

    job = Job(py_command_slow, stdout=partial(line, 'stdout'),
              stderr=partial(line, 'stderr'), callback=callback)

    coro = job.run()
    assert asyncio.iscoroutine(coro)
    assert not job.started
    assert not job.done
    with pytest.raises(AttributeError):
        job.proc

    result = await coro
    assert isinstance(result, int)
    assert job.started
    assert job.done
    assert job.rc == result == 0
    assert job.proc.pid > 0


@pytest.mark.asyncio
async def test_job_threaded(py_command_slow, py_output_slow, event_loop):
    """Test running job completely asynchronously."""

    def callback(job):
        assert job.rc == 0

    def line(stream_type, line):
        assert line == py_output_slow[stream_type].pop(0)

    job = Job(py_command_slow, stdout=partial(line, 'stdout'),
              stderr=partial(line, 'stderr'), callback=callback)

    future = job.thread()
    assert not asyncio.isfuture(future)
    assert isinstance(future, concurrent.futures.Future)

    assert job.started
    assert not job.done
    assert job.rc is None
    assert job.proc.pid > 0

    result = future.result()
    assert job.started
    assert job.done
    assert job.rc == result == 0


@pytest.mark.asyncio
async def test_job_executor(py_command_slow, py_output_slow):
    """Test running job completely asynchronously via concurrent.futures."""

    def callback(job):
        assert job.rc == 0

    def line(stream_type, line):
        assert line == py_output_slow[stream_type].pop(0)

    # run job and assert results
    job = Job(py_command_slow, stdout=partial(line, 'stdout'),
              stderr=partial(line, 'stderr'), callback=callback)

    pool = concurrent.futures.ThreadPoolExecutor(4)
    watcher = asyncio.get_child_watcher()

    future = pool.submit(job.thread)
    assert isinstance(future, concurrent.futures.Future)

    assert not job.started
    assert not job.done
    with pytest.raises(AttributeError):
        job.proc

    loop = job.queue.get()
    watcher.attach_loop(loop)
    job.queue.task_done()

    assert not job.started
    assert not job.done
    with pytest.raises(AttributeError):
        job.proc

    result = future.result()
    assert job.started
    assert job.done
    assert job.rc == result == 0
    assert job.proc.pid > 0
