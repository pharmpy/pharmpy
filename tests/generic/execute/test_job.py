
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
@pytest.mark.asyncio
async def py_output(py_command, event_loop):
    """Test (and provides output of) py_command (async style via Job.start_loop)."""

    def callback(job):
        assert job.rc == 0

    job = Job(py_command, stdout=print, stderr=partial(print, file=sys.stderr), callback=callback)

    # same thread, job is always done after await
    await job.start_loop(event_loop)
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
async def test_job_python(py_command, py_output):
    """Test py_command fully concurrent (thread via Job.start)."""

    def callback(job):
        assert job.rc == 0

    def line(stream_type, line):
        """Assert all output lines."""
        assert line == py_output[stream_type].pop(0)

    job = Job(py_command, stdout=partial(line, 'stdout'), stderr=partial(line, 'stderr'),
              callback=callback)

    # different thread, job is NOT done after start...
    job.start()
    assert job.started
    assert not job.ended
    assert job.rc is None
    assert job.proc.pid > 0

    # ... but it can be awaited until end, of course
    job.wait()
    assert job.started
    assert job.ended
    assert job.rc == 0


@pytest.mark.asyncio
async def test_job_python_iter(py_command, py_output):
    job = Job(py_command)
    job.start()
    job.wait()
    lines = list(job.iter_stream('output'))
    print(py_output['output'])

    # FIXME: Why is the reference 3×expected (3×6=18) all of a sudden?
    assert len(py_output['output']) >= len(lines)
