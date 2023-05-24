import pytest

from pharmpy.model.external.nonmem.records.option_record import OptionRecord, StrOpt, WildOpt
from pharmpy.model.external.nonmem.records.table_record import table_options


def test_create_record(parser):
    recs = parser.parse('$INPUT ID TIME DV WGT=DROP')
    rec = recs.records[0]
    pairs = rec.option_pairs
    assert list(pairs.items()) == [('ID', None), ('TIME', None), ('DV', None), ('WGT', 'DROP')]

    recs = parser.parse('$EST MAXEVAL=9999 INTERACTION')
    rec = recs.records[0]
    pairs = rec.option_pairs
    assert list(pairs.items()) == [('MAXEVAL', '9999'), ('INTERACTION', None)]


def test_all_options(parser):
    recs = parser.parse('$INPUT ID ID TIME DV WGT=DROP')
    rec = recs.records[0]
    pairs = rec.all_options
    assert pairs == [('ID', None), ('ID', None), ('TIME', None), ('DV', None), ('WGT', 'DROP')]


def test_set_option(parser):
    rec = parser.parse('$ETAS FILE=run1.phi').records[0]
    newrec = rec.set_option("FILE", "new.phi")
    assert newrec.option_pairs == {'FILE': 'new.phi'}
    assert str(newrec) == '$ETAS FILE=new.phi'

    rec = parser.parse('$EST METHOD=1 INTER ; my est\n').records[0]
    newrec = rec.set_option("METHOD", "0")
    assert str(newrec) == '$EST METHOD=0 INTER ; my est\n'
    newrec2 = newrec.set_option("CTYPE", "4")
    assert str(newrec2) == '$EST METHOD=0 INTER CTYPE=4 ; my est\n'


@pytest.mark.parametrize(
    "buf,remove,expected",
    [
        ('$SUBS ADVAN1 TRANS2', 'TRANS2', '$SUBS ADVAN1'),
        ('$SUBS ADVAN1  TRANS2', 'TRANS2', '$SUBS ADVAN1'),
        ('$SUBS   ADVAN1 TRANS2 ;COMMENT', 'ADVAN1', '$SUBS TRANS2 ;COMMENT'),
    ],
)
def test_remove_option(parser, buf, remove, expected):
    rec = parser.parse(buf).records[0]
    newrec = rec.remove_option(remove)
    assert str(newrec) == expected


@pytest.mark.parametrize(
    "buf,remove,expected",
    [
        ('$SUBS ADVAN1 TRANS2', 'TRANS', '$SUBS ADVAN1'),
        ('$SUBS ADVAN1  TRANS2', 'TRANS', '$SUBS ADVAN1'),
        ('$SUBS   ADVAN1 TRANS2 ;COMMENT', 'ADVA', '$SUBS TRANS2 ;COMMENT'),
    ],
)
def test_remove_option_startswith(parser, buf, remove, expected):
    rec = parser.parse(buf).records[0]
    newrec = rec.remove_option_startswith(remove)
    assert str(newrec) == expected


@pytest.mark.parametrize(
    "buf,expected",
    [
        ('$MODEL COMP=1 COMP=2', [['1'], ['2']]),
        ('$MODEL COMP 1 COMP 2', [['1'], ['2']]),
        ('$MODEL COMP=(CENTRAL) COMP=(PERIPHERAL)', [['CENTRAL'], ['PERIPHERAL']]),
        (
            '$MODEL COMP=(CENTRAL DEFDOSE DEFOBS) COMP=(PERIPHERAL)',
            [['CENTRAL', 'DEFDOSE', 'DEFOBS'], ['PERIPHERAL']],
        ),
        (
            '$MODEL COMP  (CENTRAL DEFDOSE DEFOBS) COMP=(PERIPHERAL)',
            [['CENTRAL', 'DEFDOSE', 'DEFOBS'], ['PERIPHERAL']],
        ),
    ],
)
def test_get_option_lists(parser, buf, expected):
    rec = parser.parse(buf).records[0]
    it = rec.get_option_lists('COMPARTMENT')
    assert list(it) == expected


@pytest.mark.parametrize(
    "buf,n,subopt,result",
    [
        (
            '$MODEL COMP=(CENTRAL) COMP=(PERIPHERAL)',
            0,
            'DEFDOSE',
            '$MODEL COMP=(CENTRAL DEFDOSE) COMP=(PERIPHERAL)',
        ),
        (
            '$MODEL COMP=(CENTRAL) COMP=(PERIPHERAL)',
            1,
            'DEFOBS',
            '$MODEL COMP=(CENTRAL) COMP=(PERIPHERAL DEFOBS)',
        ),
        (
            '$MODEL COMP=CENTRAL COMP=PERIPHERAL',
            1,
            'DEFOBS',
            '$MODEL COMP=CENTRAL COMP=(PERIPHERAL DEFOBS)',
        ),
    ],
)
def test_add_suboption_for_nth(parser, buf, n, subopt, result):
    rec = parser.parse(buf).records[0]
    newrec = rec.add_suboption_for_nth('COMPARTMENT', n, subopt)
    assert str(newrec) == result


@pytest.mark.parametrize(
    "valid,opt,expected",
    [
        (['NOOFF', 'DEFDOSE', 'DEFOBS'], 'DEFDOSE', 'DEFDOSE'),
        (['NOOFF', 'DEFDOSE', 'DEFOBS'], 'DEFDOS', 'DEFDOSE'),
        (['NOOFF', 'DEFDOSE', 'DEFOBS'], 'DEFD', 'DEFDOSE'),
        (['NOOFF', 'DEFDOSE', 'DEFOBS'], 'DEF', None),
        (['NOOFF', 'DEFDOSE', 'DEFOBS'], 'DEFO', 'DEFOBS'),
        (['NOOFF', 'DEFDOSE', 'DEFOBS'], 'NO', None),
        (['NOOFF', 'DEFDOSE', 'DEFOBS'], 'NOOF', 'NOOFF'),
        (['AAA', 'AAA'], 'AAA', None),
        (['AAA', 'AAA'], 'AAAA', None),
        (['AAAA', 'AAAA'], 'AAAA', None),
        (['ZZZZZZBAAA', 'ZZZZZZAAAA', 'ZZZZZZABCD'], 'ZZZZZZAAA', 'ZZZZZZAAAA'),
        (['AAAA', 'AAAA'], 'AAAAA', None),
        (['AAAAB', 'AAAA'], 'AAAAA', None),
        (['AAAAB', 'AAAA', 'AAAAA'], 'AAAAA', 'AAAAA'),
        (['AAAAB', 'AAAA', 'AAAAA'], 'AAAAAA', 'AAAAA'),
        (['AAAAB', 'AAAA', 'AAAAAA'], 'AAAAA', 'AAAAAA'),
    ],
)
def test_match_option(valid, opt, expected):
    match = OptionRecord.match_option(valid, opt)
    assert match == expected


def test_append_node(parser):
    rec = parser.parse('$ESTIMATION METH=0 MAXEVALS=0').records[0]
    newrec = rec.append_option('INTERACTION')
    assert str(newrec) == '$ESTIMATION METH=0 MAXEVALS=0 INTERACTION'
    newrec2 = newrec.append_option('MCETA', '100')
    assert str(newrec2) == '$ESTIMATION METH=0 MAXEVALS=0 INTERACTION MCETA=100'


def test_prepend_node(parser):
    rec = parser.parse('$ESTIMATION METH=0 MAXEVALS=0').records[0]
    newrec = rec.prepend_option('INTERACTION')
    assert str(newrec) == '$ESTIMATION INTERACTION METH=0 MAXEVALS=0'
    newrec2 = newrec.prepend_option('MCETA', '250')
    assert str(newrec2) == '$ESTIMATION MCETA=250 INTERACTION METH=0 MAXEVALS=0'


def test_remove_subotion_for_all(parser):
    rec = parser.parse('$MODEL COMP=(COMP1 DEFDOSE) COMP=(COMP2)').records[0]
    newrec = rec.remove_suboption_for_all('COMPARTMENT', 'DEFDOSE')
    assert str(newrec) == '$MODEL COMP=(COMP1) COMP=(COMP2)'


def test_options(parser):
    assert table_options['NOPRINT'].abbreviations == ['NOPRINT', 'NOPRIN', 'NOPRI', 'NOPR', 'NOP']
    rec = parser.parse('$TABLE IPRED DV FILE=sdtab').records[0]
    assert rec.parse_options(nonoptions=set(), netas=2) == [
        (WildOpt(), 'IPRED', None),
        (WildOpt(), 'DV', None),
        (StrOpt('FILE'), 'FILE', 'sdtab'),
    ]


@pytest.mark.usefixtures('parser')
@pytest.mark.parametrize(
    "buf",
    [
        '$TABLE CLOCKSEED=2',
        '$TABLE ESAMPLE=str',
        '$TABLE ID TIME PRINT NOPRINT',
        '$TABLE CLOCKSEED=0 CLOCK=0',
    ],
)
def test_option_errors(parser, buf):
    with pytest.raises(ValueError):
        parser.parse(buf).records[0].parse_options(nonoptions=set(), netas=2)


@pytest.mark.usefixtures('parser')
@pytest.mark.parametrize(
    "buf,netas,correct",
    [
        ('$TABLE ID TIME', 1, ['ID', 'TIME']),
        ('$TABLE ID TIME ETAS(1:2)', 2, ['ID', 'TIME', 'ETA1', 'ETA2']),
        ('$TABLE ID ETAS(1:2) TIME', 2, ['ID', 'ETA1', 'ETA2', 'TIME']),
        ('$TABLE ID ETAS (1 : 2   ) TIME', 2, ['ID', 'ETA1', 'ETA2', 'TIME']),
        ('$TABLE ID ETAS(1:LAST) TIME', 3, ['ID', 'ETA1', 'ETA2', 'ETA3', 'TIME']),
        ('$TABLE ID ETAS(1 TO LAST) TIME', 3, ['ID', 'ETA1', 'ETA2', 'ETA3', 'TIME']),
        ('$TABLE ID ETAS(1 TO LAST BY 2) TIME', 4, ['ID', 'ETA1', 'ETA3', 'TIME']),
        ('$TABLE ID ETAS(1,2,4) TIME', 4, ['ID', 'ETA1', 'ETA2', 'ETA4', 'TIME']),
        ('$TABLE ID ETAS(3:1) TIME', 3, ['ID', 'ETA3', 'ETA2', 'ETA1', 'TIME']),
        ('$TABLE ID ETAS(4:1 BY -2) TIME', 4, ['ID', 'ETA4', 'ETA2', 'TIME']),
        ('$TABLE ID ETAS(1:4 BY -1) TIME', 4, ['ID', 'ETA4', 'ETA3', 'ETA2', 'ETA1', 'TIME']),
    ],
)
def test_table_parsing(parser, buf, netas, correct):
    res = parser.parse(buf).records[0].parse_options(nonoptions=set(), netas=netas)
    names = [col[1] for col in res]
    assert names == correct
