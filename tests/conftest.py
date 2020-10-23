from pathlib import Path

import pytest
from pyfakefs.fake_filesystem_unittest import Patcher

# Do not let pyfakefs patch mpmath since it emits warnings
# and doesn't do anything with the filesystem anyway
Patcher.SKIPNAMES.update(['mpmath'])


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
