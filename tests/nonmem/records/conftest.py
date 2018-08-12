
import pytest


@pytest.fixture(scope='module')
def str_repr():
    def func(string):
        if not string:
            return '-- EMPTY --'
        return '"' + repr(string)[1:-1] + '"'
    return func


@pytest.fixture(scope='module')
def records(nonmemAPI):
    return nonmemAPI.records


@pytest.fixture(scope='module')
def parser(records):
    return records.parser
