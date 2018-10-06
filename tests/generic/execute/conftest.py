
import sys
import textwrap

import pytest


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
def py_output_slow():
    """Provides output of py_command_slow."""

    output = textwrap.dedent("""
    OUT[1]
    ERR[1]
    OUT[2]
    ERR[2]
    OUT[3]
    ERR[3]
    """).strip().splitlines()

    reference = dict(output=output)
    reference['stdout'] = list(filter(lambda x: x.startswith('OUT'), output))
    reference['stderr'] = list(filter(lambda x: x.startswith('ERR'), output))
    return reference
