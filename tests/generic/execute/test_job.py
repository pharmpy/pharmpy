# -*- encoding: utf-8 -*-
"""
.. todo:: Important! Exception handling of threaded processes! Main thread MUST raise again.
"""

import asyncio
import concurrent.futures
from functools import partial

import pytest

from pharmpy.execute.job import Job


@pytest.mark.asyncio
async def test_job_blocking(py_command_slow, py_output_slow, event_loop):
    """Test py_command_slow execution, BLOCKING & ASYNC."""

    def callback(job):
        assert job.rc == 0

    def line(stream_type, line):
        assert line == py_output_slow[stream_type].pop(0)

    job = Job(py_command_slow, stdout=partial(line, 'stdout'),
              stderr=partial(line, 'stderr'), callback=callback)

    # run job and assert results
    await job.run(block=True)
    assert job.started
    assert job.ended
    assert job.rc == 0
    assert job.proc.pid > 0


def test_job_nonblocking(py_command_slow, py_output_slow):
    """Test py_command_slow execution, NONBLOCKING thread."""

    def callback(job):
        assert job.rc == 0

    def line(stream_type, line):
        assert line == py_output_slow[stream_type].pop(0)

    job = Job(py_command_slow, stdout=partial(line, 'stdout'),
              stderr=partial(line, 'stderr'), callback=callback)

    # run job and assert results
    job.run(block=False)
    assert job.started
    assert not job.ended
    assert job.rc is None
    assert job.proc.pid > 0

    job.wait()
    assert job.started
    assert job.ended
    assert job.rc == 0


def test_job_thread_executor(py_command_slow, py_output_slow):
    """Test py_command_slow execution, using ThreadPoolExecutor."""

    def callback(job):
        assert job.rc == 0

    def line(stream_type, line):
        assert line == py_output_slow[stream_type].pop(0)

    # run job and assert results
    job = Job(py_command_slow, stdout=partial(line, 'stdout'),
              stderr=partial(line, 'stderr'), callback=callback)
    pool = concurrent.futures.ThreadPoolExecutor(4)
    future = pool.submit(job.run, block=True)
    assert not job.started
    assert not job.ended
    with pytest.raises(AttributeError):
        job.proc

    loop = job.queue.get()
    asyncio.get_child_watcher().attach_loop(loop)
    job.queue.task_done()
    assert not job.started
    assert not job.ended

    future.result()
    assert job.started
    assert job.ended
    assert job.rc == 0
    assert job.proc.pid > 0
