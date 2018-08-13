
import csv
import re
from collections import namedtuple
from pathlib import Path

import pytest


tuple_matcher = re.compile(r'^\((.*)\)$')


@pytest.fixture(scope='session')
def testdata():
    return Path(__file__).resolve().parent / 'testdata'


# @pytest.fixture(scope='session')
# def SOURCE():
#     return Path(__file__).resolve().parent.parent / 'src'


@pytest.fixture(scope='session')
def csv_read():
    def func(root, file, names=None):
        TestData = tuple
        if names:
            TestData = namedtuple('TestData', names)

        def descape(item):
            return item.replace('\\n', '\n')

        def tupleize(item):
            m = tuple_matcher.match(item)
            if not m:
                return descape(item)
            if not m.group(1):
                return ()
            return tuple(descape(x.strip(' "')) for x in m.group(1).split(','))

        with open(Path(root, file), 'r') as f:
            dialect = csv.Sniffer().sniff(f.read(1024))
            f.seek(0)
            reader = csv.reader(f, dialect)
            return tuple(TestData(*tuple(map(tupleize, row))) for row in reader)

    return func


@pytest.fixture(scope='session')
def str_repr():
    def func(string):
        if not string:
            return '-- EMPTY --'
        return '"' + repr(string)[1:-1] + '"'
    return func
