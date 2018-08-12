
from functools import partial

import pytest

from pysn import getAPI


@pytest.fixture(scope='module')
def nonmemAPI():
    return getAPI('nonmem')


@pytest.fixture(scope='module')
def nm_testdata(testdata):
    return testdata / 'nonmem'


@pytest.fixture(scope='module')
def pheno_real(nm_testdata):
    return nm_testdata / 'pheno_real.mod'


@pytest.fixture(scope='module')
def nm_csv_read(nm_testdata, csv_read):
    return partial(csv_read, nm_testdata)
