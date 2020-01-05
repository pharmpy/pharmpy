# import logging

# import pytest


# @pytest.mark.asyncio
# async def test_pheno_execute(pheno, event_loop):
#    """Test actual execution of pheno."""
#    logger = logging.getLogger()

#    copy = pheno.copy()
#    rundir = await copy.execute.estimate()
#    logger.debug('%r._jobs=%r', rundir, rundir._jobs)

#    job = rundir._jobs[0]
#    assert not job.done
#    await job.wait(10, 0.5)
#    assert job.done
# assert job.rc == 0

#    path = rundir.path
#    assert path.exists()
