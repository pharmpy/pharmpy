from pathlib import Path

import pytest


@pytest.fixture(scope='session')
def testdata():
    """Test data (root) folder."""
    return Path(__file__).resolve().parent / 'testdata'


@pytest.fixture(scope='session')
def datadir(testdata):
    return testdata / 'nonmem'


@pytest.fixture(scope='session')
def pheno_path(datadir):
    return datadir / 'pheno_real.mod'


@pytest.fixture(scope='session')
def pheno(pheno_path):
    import pharmpy

    return pharmpy.Model.create_model(pheno_path)
