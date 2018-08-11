
from pathlib import Path

import pytest


@pytest.fixture(scope='session')
def testdata():
    return Path(__file__).resolve().parent / 'testdata'
