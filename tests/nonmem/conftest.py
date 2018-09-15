import pytest

from pharmpy.api_utils import getAPI


@pytest.fixture
def nonmem():
    return getAPI('nonmem')


@pytest.fixture
def datadir(testdata):
    return testdata / 'nonmem'


@pytest.fixture
def pheno_real(datadir):
    return datadir / 'pheno_real.mod'


@pytest.fixture
def pheno_data(datadir):
    return datadir / 'pheno.dta'
