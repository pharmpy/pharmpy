
import pytest

from pysn import getAPI


@pytest.fixture(scope='session')
def api():
    return getAPI('nonmem')


@pytest.fixture(scope='session')
def datadir(testdata):
    return testdata / 'nonmem'


@pytest.fixture(scope='session')
def pheno_real(datadir):
    return datadir / 'pheno_real.mod'
