
import pytest

from pysn import getAPI


@pytest.fixture
def api():
    return getAPI('nonmem')


@pytest.fixture
def datadir(testdata):
    return testdata / 'nonmem'


@pytest.fixture
def pheno_real(datadir):
    return datadir / 'pheno_real.mod'
