import pytest

import pharmpy


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
def pheno_phi(datadir):
    return datadir / 'pheno_real.phi'


@pytest.fixture(scope='session')
def pheno_cov(datadir):
    return datadir / 'pheno_real.cov'


@pytest.fixture(scope='session')
def pheno_lst(datadir):
    return datadir / 'pheno_real.lst'


@pytest.fixture(scope='session')
def pheno_data(datadir):
    return datadir / 'pheno.dta'


@pytest.fixture(scope='session')
def pheno(pheno_path):
    model = pharmpy.Model(pheno_path)
    return model
