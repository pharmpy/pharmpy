
import pytest

from pysn import getAPI


@pytest.fixture(scope='module')
def nm_api():
    return getAPI('nonmem')


@pytest.fixture(scope='module')
def nm_testdata(testdata):
    return testdata / 'nonmem'


@pytest.fixture(scope='module')
def pheno_real(nm_testdata):
    return nm_testdata / 'pheno_real.mod'
