from os.path import join, realpath

import pytest


@pytest.fixture(scope='module')
def records(api):
    return api.module.records
