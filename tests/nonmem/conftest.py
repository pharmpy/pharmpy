import pytest

from pharmpy.api_utils import getAPI


@pytest.fixture(scope='session')
def nonmem():
    return getAPI('nonmem')


@pytest.fixture(scope='session')
def datadir(testdata):
    return testdata / 'nonmem'


@pytest.fixture(scope='session')
def pheno_path(datadir):
    return datadir / 'pheno_real.mod'


@pytest.fixture(scope='session')
def pheno_ext(datadir):
    return datadir / 'pheno_real.ext'

@pytest.fixture(scope='session')
def pheno_lst(datadir):
    return datadir / 'pheno_real.lst'

@pytest.fixture(scope='session')
def pheno_data(datadir):
    return datadir / 'pheno.dta'


@pytest.fixture(scope='session')
def pheno(nonmem, pheno_path):
    model = nonmem.Model(str(pheno_path))
    assert model.path.samefile(pheno_path)
    with open(str(pheno_path), 'r') as f:
        assert str(model) == model.content == f.read()
    return model
