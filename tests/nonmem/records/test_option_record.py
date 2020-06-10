from collections import OrderedDict


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
