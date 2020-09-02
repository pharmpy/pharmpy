from collections import OrderedDict

import pytest

from pharmpy.plugins.nonmem.records.option_record import OptionRecord


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


@pytest.mark.parametrize("buf,expected", [
    ('$MODEL COMP=1 COMP=2', [['1'], ['2']]),
    ('$MODEL COMP=(CENTRAL) COMP=(PERIPHERAL)', [['CENTRAL'], ['PERIPHERAL']]),
    ('$MODEL COMP=(CENTRAL DEFDOSE DEFOBS) COMP=(PERIPHERAL)',
        [['CENTRAL', 'DEFDOSE', 'DEFOBS'], ['PERIPHERAL']]),
])
def test_get_option_lists(parser, buf, expected):
    rec = parser.parse(buf).records[0]
    it = rec.get_option_lists('COMPARTMENT')
    assert list(it) == expected


@pytest.mark.parametrize("valid,opt,expected", [
    (['NOOFF', 'DEFDOSE', 'DEFOBS'], 'DEFDOSE', 'DEFDOSE'),
    (['NOOFF', 'DEFDOSE', 'DEFOBS'], 'DEFDOS', 'DEFDOSE'),
    (['NOOFF', 'DEFDOSE', 'DEFOBS'], 'DEFD', 'DEFDOSE'),
    (['NOOFF', 'DEFDOSE', 'DEFOBS'], 'DEF', None),
    (['NOOFF', 'DEFDOSE', 'DEFOBS'], 'DEFO', 'DEFOBS'),
    (['NOOFF', 'DEFDOSE', 'DEFOBS'], 'NO', None),
    (['NOOFF', 'DEFDOSE', 'DEFOBS'], 'NOOF', 'NOOFF'),
])
def test_match_option(parser, valid, opt, expected):
    match = OptionRecord.match_option(valid, opt)
    assert match == expected
