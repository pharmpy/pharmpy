import asyncio
import logging
import sys
import textwrap

import pytest


@pytest.fixture(scope='function')
def py_command():
    code = textwrap.dedent("""
        import sys
        print('OUT', file=sys.stdout, flush=True)
        print('ERR', file=sys.stderr, flush=True)
    """)
    command = [sys.executable, '-c', code]

    output = dict(output=['OUT', 'ERR'])
    output['stdout'] = list(filter(lambda x: x.startswith('OUT'), output))
    output['stderr'] = list(filter(lambda x: x.startswith('ERR'), output))

    return (command, output)


@pytest.fixture(scope='function')
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

    output = textwrap.dedent("""
    OUT[1]
    ERR[1]
    OUT[2]
    ERR[2]
    OUT[3]
    ERR[3]
    """).strip().splitlines()
    output = dict(output=output)
    output['stdout'] = list(filter(lambda x: x.startswith('OUT'), output['output']))
    output['stderr'] = list(filter(lambda x: x.startswith('ERR'), output['output']))

    return (command, output)


def debuglog_jobs(section_title, jobs, logger=None):
    """Log debugging information on jobs, current event loop and child watcher."""

    if not logger:
        logger = logging.getLogger()
    logger.warning('DEBUG (%s)', section_title.upper())

    event_loop = asyncio.get_event_loop()
    logger.warning('event_loop=%r (%s)', event_loop, hex(id(event_loop)))
    for i, job in enumerate(jobs):
        job.init.wait()
        logger.warning('jobs[%d]: is_alive=%r job=%r', i, job.is_alive(), job)
        logger.warning('jobs[%d]: job.loop=%r (%s)', i, job.loop, hex(id(job.loop)))

    watcher = asyncio.get_child_watcher()
    logger.warning('watcher=%r: loop=%r (%s)', watcher, watcher._loop, hex(id(watcher._loop)))
    for pid, cb in watcher._callbacks.items():
        func = cb[0].__func__
        transport = cb[1][0]
        logger.warning('watcher._callbacks[%i] %r', pid, func)
        logger.warning('watcher._callbacks[%i] %r', pid, transport)
