
import sys
import textwrap
from functools import partial

import pytest

from pharmpy.execute.job import Job


@pytest.fixture(scope='session')
def py_command():
    code = textwrap.dedent("""
        import sys
        print('OUT', file=sys.stdout, flush=True)
        print('ERR', file=sys.stderr, flush=True)
    """)
    command = [sys.executable, '-c', code]
    return command


@pytest.fixture(scope='session')
def py_command_slow():
    code = textwrap.dedent("""
        import sys
        import time

        for i in range(3):
            print('OUT[%d]' % (i+1), file=sys.stdout, flush=True)
            print('ERR[%d]' % (i+1), file=sys.stderr, flush=True)
            time.sleep(0.05)
    """)
    command = [sys.executable, '-c', code]
    return command


@pytest.fixture(scope='session')
def py_output_slow(py_command_slow):
    """Provides output of py_command_slow, BLOCKING & not async."""

    def callback(job):
        assert job.rc == 0

    job = Job(py_command_slow, stdout=partial(print, file=sys.stdout),
              stderr=partial(print, file=sys.stderr), callback=callback)

    # run job and assert results
    job.run(block=True)
    assert job.started
    assert job.done
    assert job.rc == 0
    assert job.proc.pid > 0

    # provide output
    output = job.output.splitlines()
    reference = dict(output=output)
    reference['stdout'] = list(filter(lambda x: x.startswith('OUT'), output))
    reference['stderr'] = list(filter(lambda x: x.startswith('ERR'), output))
    return reference
