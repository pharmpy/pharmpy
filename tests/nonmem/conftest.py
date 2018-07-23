import pytest
from pysn import api_list
from testhelper_paths import *


@pytest.fixture(scope='module')
def api():
    return api_list.get('nonmem')

@pytest.fixture(scope='module')
def path_testdata(path_testdata):
    return dir_check(path_testdata, 'nonmem')

@pytest.fixture(scope='module')
def pheno_real(path_testdata):
    return file_check(path_testdata, 'pheno_real.mod')
