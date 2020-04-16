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


def test_set_option(parser):
    rec = parser.parse('$ETAS FILE=run1.phi').records[0]
    rec.set_option("FILE", "new.phi")
    rec.root.treeprint()
    assert rec.option_pairs == OrderedDict([('FILE', 'new.phi')])
