
import sys

import pytest

from pharmpy.execute.environment import SystemEnvironment


@pytest.mark.asyncio
async def test_SystemEnvironment_submit(py_command):
    """Test SystemEnvironment.submit, single job."""

    envir = SystemEnvironment(threads=4)
    assert not envir.jobs
    assert not envir.futures

    await envir.submit(command=py_command)
    assert len(envir.jobs) == 1
    assert len(envir.futures) == 1

    job = envir.jobs[0]
    assert not job.started
    assert not job.done

    envir.close()
    assert job.started
    assert job.done
    assert job.rc == 0
    assert job.proc.pid > 0


@pytest.mark.asyncio
async def test_SystemEnvironment_submit_many():
    """Test SystemEnvironment.submit, many jobs."""

    envir = SystemEnvironment(threads=4)
    assert not envir.jobs
    assert not envir.futures

    py_commands = [
        [sys.executable, '-c', "import time; time.sleep(0.10); print('OUT[1]', flush=True)"],
        [sys.executable, '-c', "import time; time.sleep(0.05); print('OUT[2]', flush=True)"],
        [sys.executable, '-c', "print('OUT[3]', flush=True)"],
    ]

    for i, py_command in enumerate(py_commands):
        await envir.submit(command=py_command, cwd=None)
        assert len(envir.jobs) == (i+1)
        assert len(envir.futures) == (i+1)
        assert not envir.jobs[i].started
        assert not envir.jobs[i].done

    envir.close()
    for job in envir.jobs:
        assert job.started
        assert job.done
        assert job.rc == 0
        assert job.proc.pid > 0
