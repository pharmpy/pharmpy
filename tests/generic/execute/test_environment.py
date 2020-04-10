# import sys

# import pytest

# from pharmpy.execute.environment import SystemEnvironment


# @pytest.mark.asyncio
# async def test_SystemEnvironment_submit_series(py_command):
#    """Test SystemEnvironment.submit, single job."""

#    command, output = py_command

#    envir = SystemEnvironment(threads=4)
#    assert not envir.jobs
#    assert not envir.futures

#    for i in range(5):
#        await envir.submit(command=command)
#        assert len(envir.jobs) == (i+1)
#        assert len(envir.futures) == (i+1)

#        job = envir.jobs[0]
#        await job.wait(1, 0.1)
#        assert job.init
#        assert job.done
#        assert job.rc == 0
#        assert job.proc.pid > 0


# @pytest.mark.asyncio
# async def test_SystemEnvironment_submit_parallel():
#    """Test SystemEnvironment.submit, many jobs."""

#    envir = SystemEnvironment(threads=4)
#    assert not envir.jobs
#    assert not envir.futures

#    commands = [
#        [sys.executable, '-c', "import time; time.sleep(0.20); print('OUT[5]')"],
#        [sys.executable, '-c', "import time; time.sleep(0.15); print('OUT[4]')"],
#        [sys.executable, '-c', "import time; time.sleep(0.10); print('OUT[3]')"],
#        [sys.executable, '-c', "import time; time.sleep(0.05); print('OUT[2]')"],
#        [sys.executable, '-c', "print('OUT[1]')"],
#    ]

#    for i, command in enumerate(commands):
#        await envir.submit(command=command, cwd=None)
#        assert len(envir.jobs) == (i+1)
#        assert len(envir.futures) == (i+1)
#        assert not envir.jobs[i].init
#        assert not envir.jobs[i].done

#    await envir.wait(1, 0.1)
#    for job in envir.jobs:
#        assert job.init
#        assert job.done
#        assert job.rc == 0
#        assert job.proc.pid > 0
