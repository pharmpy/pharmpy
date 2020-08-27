from collections import OrderedDict

import pytest


def test_create_record(parser):
    recs = parser.parse('$INPUT ID TIME DV WGT=DROP')
    rec = recs.records[0]
    pairs = rec.option_pairs
    assert pairs == OrderedDict([('ID', None), ('TIME', None), ('DV', None), ('WGT', 'DROP')])

    recs = parser.parse('$EST MAXEVAL=9999 INTERACTION')
    rec = recs.records[0]
    pairs = rec.option_pairs
    assert pairs == OrderedDict([('MAXEVAL', '9999'), ('INTERACTION', None)])


def test_all_options(parser):
    recs = parser.parse('$INPUT ID ID TIME DV WGT=DROP')
    rec = recs.records[0]
    pairs = rec.all_options
    assert pairs == [('ID', None), ('ID', None), ('TIME', None), ('DV', None), ('WGT', 'DROP')]


def test_set_option(parser):
    rec = parser.parse('$ETAS FILE=run1.phi').records[0]
    rec.set_option("FILE", "new.phi")
    assert rec.option_pairs == OrderedDict([('FILE', 'new.phi')])
    assert str(rec) == '$ETAS FILE=new.phi'

    rec = parser.parse('$EST METHOD=1 INTER ; my est\n').records[0]
    rec.set_option("METHOD", "0")
    assert str(rec) == '$EST METHOD=0 INTER ; my est\n'
    rec.set_option("CTYPE", "4")
    assert str(rec) == '$EST METHOD=0 INTER CTYPE=4 ; my est\n'


@pytest.mark.parametrize("buf,remove,expected", [
    ('$SUBS ADVAN1 TRANS2', 'TRANS2', '$SUBS ADVAN1'),
    ('$SUBS ADVAN1  TRANS2', 'TRANS2', '$SUBS ADVAN1'),
    ('$SUBS   ADVAN1 TRANS2 ;COMMENT', 'ADVAN1', '$SUBS TRANS2 ;COMMENT'),
])
def test_remove_option(parser, buf, remove, expected):
    rec = parser.parse(buf).records[0]
    rec.remove_option(remove)
    assert str(rec) == expected


@pytest.mark.parametrize("buf,remove,expected", [
    ('$SUBS ADVAN1 TRANS2', 'TRANS', '$SUBS ADVAN1'),
    ('$SUBS ADVAN1  TRANS2', 'TRANS', '$SUBS ADVAN1'),
    ('$SUBS   ADVAN1 TRANS2 ;COMMENT', 'ADVA', '$SUBS TRANS2 ;COMMENT'),
])
def test_remove_option_startswith(parser, buf, remove, expected):
    rec = parser.parse(buf).records[0]
    rec.remove_option_startswith(remove)
    assert str(rec) == expected
