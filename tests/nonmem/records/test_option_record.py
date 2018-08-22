from collections import OrderedDict


def test_create_record(api):
    record = api.records.create_record("ESTIMATION MAXEVALS=9999 INTER")
    pairs = record.option_pairs
    assert pairs == OrderedDict([('MAXEVALS', '9999'), ('INTER', None)])
