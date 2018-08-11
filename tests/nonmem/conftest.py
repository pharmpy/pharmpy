
import pytest
from pathlib import Path
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
