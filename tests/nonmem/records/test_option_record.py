from collections import OrderedDict


def test_create_record(nonmem):
    record = nonmem.records.create_record("ESTIMATION MAXEVALS=9999 INTER")
    pairs = record.option_pairs
    assert pairs == OrderedDict([('MAXEVALS', '9999'), ('INTER', None)])
