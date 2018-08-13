# -*- encoding: utf-8 -*-

import pytest


@pytest.fixture(scope='class')
def create_record(api, request):
    """(Inject) record creating method, with some basic logging/asserts"""
    def func(cls, buf, fail=False):
        record = api.records.factory.create_record(buf)
        print(str(record))
        if fail:
            assert record.name is None
        else:
            assert record.name.startswith(record.raw_name)
            assert record.name == cls.canonical_name
        if buf.startswith('$'):
            assert str(record) == buf
        else:
            assert str(record) == '$' + buf
        return record
    request.cls.create_record = func
