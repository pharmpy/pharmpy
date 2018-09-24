
import sys

# import pytest

from pharmpy.execute.environment import SystemEnvironment


def test_SystemEnvironment_submit(py_command):
    """Test SystemEnvironment.submit, single job."""

    envir = SystemEnvironment(threads=4)
    assert not envir.jobs and not envir.futures

    envir.submit(command=py_command, cwd=None)
    assert len(envir.jobs) == len(envir.futures) == 1

    job = envir.jobs[0]
    assert not job.started
    assert not job.ended

    envir.wait(timeout=None)
    assert job.started
    assert job.ended
    assert job.rc == 0
    assert job.proc.pid > 0


def test_SystemEnvironment_submit_many():
    """Test SystemEnvironment.submit, many jobs."""

    envir = SystemEnvironment(threads=4)
    assert not envir.jobs and not envir.futures

    py_commands = [
        [sys.executable, '-c', "print('OUT[1]', flush=True)"],
        [sys.executable, '-c', "print('OUT[2]', flush=True)"],
        [sys.executable, '-c', "print('OUT[3]', flush=True)"],
    ]

    for i, py_command in enumerate(py_commands):
        envir.submit(command=py_command, cwd=None)
        assert len(envir.jobs) == len(envir.futures) == (i+1)
        assert not envir.jobs[i].started
        assert not envir.jobs[i].ended

    envir.wait()

    for job in envir.jobs:
        assert job.started
        assert job.ended
        assert job.rc == 0
        assert job.proc.pid > 0
