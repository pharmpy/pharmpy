from os.path import realpath, dirname, abspath, join, isdir, isfile, basename
import pytest


TESTHELPER_PATH = realpath(dirname(abspath(__file__)))


def testdata_check():
    path = join(dirname(TESTHELPER_PATH), 'testdata')
    __tracebackhide__ = True
    if not isdir(path):
        dir = basename(path)
        pytest.fail("(common) testdata dir '%s' not found (path: %s)" % (dir, path,))
    return path


def dir_check(*paths):
    path = join(*paths)
    __tracebackhide__ = True
    if not isdir(path):
        dir = basename(path)
        pytest.fail("dir '%s' not found (path: %s)" % (dir, path,))
    return path


def file_check(*paths):
    path = join(*paths)
    __tracebackhide__ = True
    if not isfile(path):
        file = basename(path)
        pytest.fail("file '%s' not found (path: %s)" % (file, path))
    return path
