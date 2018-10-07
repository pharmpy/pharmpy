
import asyncio
import logging

import pytest


@pytest.mark.asyncio
async def test_pheno_execute(pheno, event_loop):
    """Test actual execution of pheno."""
    logger = logging.getLogger()

    copy = pheno.copy()
    rundir = await copy.execute.estimate()
    logger.debug('%r._jobs=%r', rundir, rundir._jobs)
    await asyncio.gather(*[job.wait(5, 0.1) for job in rundir._jobs])
