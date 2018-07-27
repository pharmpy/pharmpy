from os.path import join, realpath

import pytest


@pytest.fixture(scope='module')
def records(nonmemAPI):
    return nonmemAPI.records
