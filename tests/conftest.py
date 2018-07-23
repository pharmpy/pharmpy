import sys
import pytest
from testhelper.testhelper_paths import TESTHELPER_PATH, testdata_check


sys.path.append(TESTHELPER_PATH)


@pytest.fixture(scope='session')
def path_testdata():
    return testdata_check()
