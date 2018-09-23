
import asyncio
import concurrent.futures
import sys
import textwrap
from functools import partial

import pytest

from pharmpy.execute.job import Job


@pytest.fixture
def py_command():
    code = textwrap.dedent("""
        import sys
        import time

        for i in range(3):
            print('OUT[%d]' % i, file=sys.stdout)
            print('ERR[%d]' % i, file=sys.stderr)
            time.sleep(0.05)
    """)
    command = [sys.executable, '-c', code]
    return command


@pytest.fixture
def py_output(py_command):
    """Provides output of py_command, BLOCKING & not async."""

    def callback(job):
        assert job.rc == 0

    job = Job(py_command, stdout=partial(print, file=sys.stdout),
              stderr=partial(print, file=sys.stderr), callback=callback)

    # run job and assert results
    job.run(block=True)
    assert job.started
    assert job.ended
    assert job.rc == 0
    assert job.proc.pid > 0

    # provide output
    output = job.output.splitlines()
    reference = dict(output=output)
    reference['stdout'] = list(filter(lambda x: x.startswith('OUT'), output))
    reference['stderr'] = list(filter(lambda x: x.startswith('ERR'), output))
    return reference


@pytest.mark.asyncio
async def test_job_blocking(py_command, py_output, event_loop):
    """Test py_command execution, BLOCKING & ASYNC."""

    def callback(job):
        assert job.rc == 0

    def line(stream_type, line):
        assert line == py_output[stream_type].pop(0)

    job = Job(py_command, stdout=partial(line, 'stdout'),
              stderr=partial(line, 'stderr'), callback=callback)

    # run job and assert results
    await job.run(block=True)
    assert job.started
    assert job.ended
    assert job.rc == 0
    assert job.proc.pid > 0


def test_job_nonblocking(py_command, py_output):
    """Test py_command execution, NONBLOCKING thread."""

    def callback(job):
        assert job.rc == 0

    def line(stream_type, line):
        assert line == py_output[stream_type].pop(0)

    job = Job(py_command, stdout=partial(line, 'stdout'),
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


def test_job_thread_executor(py_command, py_output):
    """Test py_command execution, using ThreadPoolExecutor."""

    def callback(job):
        assert job.rc == 0

    def line(stream_type, line):
        assert line == py_output[stream_type].pop(0)

    # run job and assert results
    job = Job(py_command, stdout=partial(line, 'stdout'),
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
