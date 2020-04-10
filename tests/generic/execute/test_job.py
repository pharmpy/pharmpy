# import asyncio
# import logging
# import threading

# import pytest

# from pharmpy.execute.job import Job


# class JobAssertions:
#    def __init__(self, stdout, stderr):
#        logging.getLogger().debug('Initializing: %r', self)
#        self.stdout_reference = stdout
#        self.stderr_reference = stderr
#        self.stdout_captured = []
#        self.stderr_captured = []

#    def stdout(self, line):
#        with threading.Lock():
#            self.stdout_captured += [line]

#    def stderr(self, line):
#        with threading.Lock():
#            self.stderr_captured += [line]

#    def callback(self, job):
#        logging.getLogger().debug('Callback (rc=%r) from job=%r: %r', job.rc, job, self)
#        with threading.Lock():
#            self.job = job

#    def assert_captures(self):
#        if hasattr(self, 'job'):
#            logging.getLogger().debug('Asserting job (rc=%r) %r: %r', self.job.rc, self.job, self)
#        else:
#            raise RuntimeError('No job has called back: %r' % self)
#        with threading.Lock():
#            assert self.job.rc == 0
#            assert self.stdout_reference == self.stdout_captured
#            assert self.stderr_reference == self.stderr_captured

#    def __repr__(self):
#        with threading.Lock():
#            return '<%s %s>' % (self.__class__.__name__, hex(id(self)))


# @pytest.mark.asyncio
# async def test_job_start_single(py_command):
#    command = py_command[0]
#    asyncio.get_child_watcher()

#    job = Job(command)
#    job.start()
#    job.init.wait()

#    await job.wait(3, 0.1)
#    assert job.init
#    assert job.done
#    assert job.rc == 0


# def test_job_start_series(py_command_slow, event_loop):
#    """Test running job completely asynchronously."""

#    command, output = py_command_slow

#    async def test_one():
#        watcher = asyncio.get_child_watcher()
#        watcher.attach_loop(asyncio.get_event_loop())

#        ref = JobAssertions(stdout=output['stdout'], stderr=output['stderr'])
#        job = Job(command, stdout=ref.stdout, stderr=ref.stderr, callback=ref.callback)
#        job.start()
#        assert not job.init

#        job.init.wait()
#        assert not job.done
#        assert job.rc is None
#        assert job.proc.pid > 0

#        await job.wait(3, 0.1)
#        assert job.init
#        assert job.done
#        ref.assert_captures()

#    async def test_series():
#        for i in range(5):
#            await test_one()

#    event_loop.run_until_complete(test_series())


# @pytest.mark.asyncio
# async def test_job_start_parallel(py_command_slow):
#    """Test running job completely asynchronously."""

#    command, output = py_command_slow
#    asyncio.get_child_watcher()

#    jobs, refs = [], []
#    for i in range(5):
#        ref = JobAssertions(stdout=output['stdout'], stderr=output['stderr'])
#        job = Job(command, stdout=ref.stdout, stderr=ref.stderr, callback=ref.callback)
#        jobs += [job]
#        refs += [ref]
#        job.start()
#        assert not job.init

#    for job in jobs:
#        job.init.wait()
#        assert not job.done

#    for ref, job in zip(refs, jobs):
#        await job.wait(3, 0.1)
#        ref.assert_captures()
