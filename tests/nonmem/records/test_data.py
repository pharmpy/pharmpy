import pytest
from lark.exceptions import UnexpectedToken

from pharmpy.basic import Expr
from pharmpy.model import Ignore, ModelSyntaxError


def test_data_filename_get(parser):
    record = parser.parse('$DATA "pheno.dta"').records[0]
    assert record.filename == 'pheno.dta'

    record = parser.parse('$DATA /home/full/pheno.dta').records[0]
    assert record.filename == '/home/full/pheno.dta'

    record = parser.parse("$DATA 'pheno.dta'").records[0]
    assert str(record.root.subtree('filename')) == "'pheno.dta'"
    assert record.filename == "pheno.dta"

    record = parser.parse(r'$DATA "C:\windowspath\with space in.csv"').records[0]
    assert record.filename == r'C:\windowspath\with space in.csv'

    record = parser.parse('$DATA \n pheno.dta \n; comment\n').records[0]
    assert record.filename == 'pheno.dta'

    record = parser.parse('$DATA ; comment\n ; some comment line\n pheno.dta\n\n').records[0]
    assert record.filename == 'pheno.dta'


def test_data_filename_set(parser):
    record = parser.parse('$DATA DUMMY ; comment').records[0]
    assert record.filename == 'DUMMY'
    assert str(record) == '$DATA DUMMY ; comment'

    # Simple replace
    record = record.set_filename('/new/path/to_file.txt')
    assert record.filename == '/new/path/to_file.txt'
    assert str(record) == '$DATA /new/path/to_file.txt ; comment'

    # Force quoting
    record = record.set_filename('MUST=QUOTE')
    assert record.filename == 'MUST=QUOTE'
    assert str(record) == "$DATA 'MUST=QUOTE' ; comment"

    # More complex example
    text = '$DATA ; comment\n ; some comment line\n pheno.dta\n\n'
    record = parser.parse(text).records[0]
    assert record.filename == 'pheno.dta'
    assert str(record) == text

    # More complex replace
    record = record.set_filename("'IGNORE'")
    assert record.filename == "'IGNORE'"
    assert str(record) == text.replace('pheno.dta', '"\'IGNORE\'"')

    # *
    record = parser.parse('$DATA DUMMY ; comment').records[0]
    record = record.set_filename(None)
    assert str(record) == '$DATA * ; comment'


def test_option_record(parser):
    record = parser.parse('$DATA pheno.dta NOWIDE').records[0]
    assert record.option_pairs == {'NOWIDE': None}


def test_ignore_character(parser):
    record = parser.parse('$DATA pheno.dta').records[0]
    assert record.ignore_character is None
    record = record.set_ignore_character('I')
    assert record.ignore_character == 'I'

    record = parser.parse('$DATA pheno.dta IGNORE=@').records[0]
    assert record.filename == 'pheno.dta'
    assert record.ignore_character == '@'
    record = record.set_ignore_character('K')
    assert record.ignore_character == 'K'

    record = parser.parse('$DATA pheno.dta IGNORE="I"').records[0]
    assert record.ignore_character == 'I'

    record = parser.parse('$DATA pheno.dta IGNORE=\'"\'').records[0]
    assert record.ignore_character == '"'

    record = parser.parse('$DATA pheno.dta IGNORE=K IGNORE=(ID.EQ.2)').records[0]
    assert record.ignore_character == 'K'

    record = parser.parse('$DATA pheno.dta IGNORE=(DV==3) IGNORE=C').records[0]
    assert record.ignore_character == 'C'
    record = record.set_ignore_character('@')
    assert record.ignore_character == '@'
    assert str(record.ignore[0]) == 'DV==3'

    record = parser.parse('$DATA pheno.dta IGNORE=,').records[0]
    assert record.ignore_character == ','

    record = parser.parse('$DATA pheno.dta IGNORE="').records[0]
    assert record.ignore_character == '"'
    record = record.set_ignore_character('"')
    assert record.ignore_character == '"'
    assert str(record) == '$DATA pheno.dta IGNORE="'

    with pytest.raises(UnexpectedToken):
        record = parser.parse('$DATA pheno.dta IGNORE=""').records[0]

    record = parser.parse('$DATA pheno.dta IGNORE=c IGNORE=@').records[0]
    with pytest.raises(ModelSyntaxError):
        record.ignore_character


def test_ignore_character_from_header(parser):
    record = parser.parse('$DATA pheno.dta').records[0]
    assert record.ignore_character is None
    record = record.set_ignore_character_from_header("ID")
    assert record.ignore_character == '@'
    record = record.set_ignore_character_from_header("_ID")
    assert record.ignore_character == '_'


def test_null_value(parser):
    record = parser.parse('$DATA pheno.dta NULL=1').records[0]
    assert record.null_value == 1
    record = parser.parse('$DATA pheno.dta NULL=+').records[0]
    assert record.null_value == 0


def test_ignore_accept(parser):
    record = parser.parse('$DATA pheno.dta IGNORE=(DV.EQ.1)').records[0]
    assert str(record.ignore[0]) == 'DV.EQ.1'
    assert record.accept == []
    record = record.remove_ignore()
    assert record.ignore == []
    assert record.accept == []

    record = parser.parse('$DATA pheno.dta ACCEPT=(DV.EQ.1,    MDV.NEN.23)').records[0]
    assert str(record.accept[0]) == 'DV.EQ.1'
    assert str(record.accept[1]) == 'MDV.NEN.23'
    assert record.ignore == []
    record = record.remove_accept()
    assert record.ignore == []
    assert record.accept == []
    record = parser.parse('$DATA pheno.dta IGNORE=(WGT  < 1  ,\n  ID\n.EQ."lk")').records[0]
    assert str(record.ignore[0]) == 'WGT  < 1', 'ID\n.EQ."lk"'

    record = parser.parse('$DATA      ../pheno.dta IGNORE=@ IGNORE(APGR.GT.23)\n').records[0]
    record = record.remove_ignore().remove_accept()
    assert str(record) == '$DATA      ../pheno.dta IGNORE=@ \n'


@pytest.mark.parametrize(
    'rec, expected',
    [
        (
            'IGNORE=(DV.EQN.1)',
            ['DV == 1'],
        ),
        (
            'ACCEPT=(DV.EQN.1, MDV.NEN.23)',
            ['DV != 1', 'MDV == 23'],
        ),
        (
            'IGNORE=(WGT < 1, ID.NEN.1)',
            ['WGT < 1', 'ID != 1'],
        ),
        (
            'IGNORE(APGR.GT.23)',
            ['APGR > 23'],
        ),
        (
            'IGNORE=(APGR.GE.23)',
            ['APGR >= 23'],
        ),
        (
            'IGNORE(APGR.LT.23)',
            ['APGR < 23'],
        ),
        (
            'IGNORE(APGR.LE.23)',
            ['APGR <= 23'],
        ),
        (
            'IGNORE=(DV.EQN.0.1)',
            ['DV == 0.1'],
        ),
        (
            'IGNORE=(DV.EQN.1,ID.NEN.1)',
            ['DV == 1', 'ID != 1'],
        ),
        (
            'IGNORE=(DV.EQN.1) ACCEPT=(APGR.LE.23, MDV.NEN.23) IGNORE=(ID.NEN.1)',
            ['DV == 1', 'APGR > 23', 'MDV == 23', 'ID != 1'],
        ),
    ],
)
def test_get_filters(parser, rec, expected):
    record = parser.parse(f'$DATA pheno.dta {rec}').records[0]
    selects = record.get_filters()
    assert len(selects) == len(expected)
    assert selects == [Ignore.create(expr) for expr in expected]


@pytest.mark.parametrize(
    'rec, expr, strings',
    [
        (
            'IGNORE=(DV.EQ.1)',
            'DV == S',
            {Expr.symbol('S'): '1'},
        ),
        (
            'IGNORE=(DV.NE.1)',
            'DV != S',
            {Expr.symbol('S'): '1'},
        ),
        (
            'IGNORE=(ID.EQ."lk")',
            'ID == S',
            {Expr.symbol('S'): '"lk"'},
        ),
        (
            'ACCEPT=(DV.EQ.1)',
            'DV != S',
            {Expr.symbol('S'): '1'},
        ),
    ],
)
def test_get_filters_strings(parser, rec, expr, strings):
    record = parser.parse(f'$DATA pheno.dta {rec}').records[0]
    selects = record.get_filters()
    assert selects == [Ignore.create(expr, strings=strings)]


def test_comments(parser):
    record = parser.parse('$DATA pheno.dta IGNORE=@;MYCOMMENT').records[0]
    assert str(record) == '$DATA pheno.dta IGNORE=@;MYCOMMENT'


def test_data_infile(parser):
    record = parser.parse('$INFILE pheno.dta').records[0]
    assert record.name == 'DATA'
    assert record.filename == 'pheno.dta'
    assert record.raw_name == '$INFILE'


def test_comment(parser):
    contents = r"""$DATA     cpt7.dta IGNORE= #
;         Dataset
"""
    record = parser.parse(contents).records[0]
    record = record.set_ignore_character("A")
    assert str(record) == '$DATA     cpt7.dta \n;         Dataset\nIGNORE=A\n'


@pytest.mark.parametrize(
    'new, expected',
    [
        (Ignore.create('DV == 2'), 'DV.EQN.2'),
        (Ignore.create('DV == S', strings={Expr.symbol('S'): '2'}), 'DV.EQ.2'),
        (Ignore.create('DV != 2'), 'DV.NEN.2'),
        (Ignore.create('DV != S', strings={Expr.symbol('S'): '2'}), 'DV.NE.2'),
        (Ignore.create('DV < 2'), 'DV.LT.2'),
        (Ignore.create('DV <= 2'), 'DV.LE.2'),
        (Ignore.create('DV > 2'), 'DV.GT.2'),
        (Ignore.create('DV >= 2'), 'DV.GE.2'),
        (Ignore.create('DV == 2.0'), 'DV.EQN.2.00000000000000'),
    ],
)
def test_update_filters_single(parser, new, expected):
    rec_old = parser.parse('$DATA pheno.dta IGNORE=(DV.EQ.1) ACCEPT=(APGR.LE.23)').records[0]
    rec_new = rec_old.update_filters([new])
    assert str(rec_new) == str(rec_old) + f' IGNORE=({expected})'
    assert str(rec_new) == str(rec_new.update_filters([new]))


def test_update_filters_existing(parser):
    rec_old = parser.parse('$DATA pheno.dta IGNORE=(DV.EQN.1)').records[0]
    rec_new = rec_old.update_filters([Ignore.create('DV == 1')])
    assert str(rec_new) == str(rec_old)


def test_update_filters_no_new(parser):
    rec_old = parser.parse('$DATA pheno.dta IGNORE=(DV.EQN.1)').records[0]
    rec_new = rec_old.update_filters([])
    assert str(rec_new) == str(rec_old)


def test_update_filters_from_empty(parser):
    rec_old = parser.parse('$DATA pheno.dta').records[0]
    rec_new = rec_old.update_filters([Ignore.create('DV == 1')])
    assert str(rec_new) == str(rec_old) + ' IGNORE=(DV.EQN.1)'


def test_update_filters_multiple(parser):
    rec_old = parser.parse('$DATA pheno.dta IGNORE=(DV.EQ.1)').records[0]
    new = [Ignore.create('DV == 2'), Ignore.create('APGR > 23')]
    rec_new = rec_old.update_filters(new)
    assert str(rec_new) == str(rec_old) + ' IGNORE=(DV.EQN.2) IGNORE=(APGR.GT.23)'


def test_update_filters_stepwise(parser):
    rec_old = parser.parse('$DATA pheno.dta IGNORE=@').records[0]
    assert rec_old.ignore == []
    ignore1 = Ignore.create('DV == S', strings={Expr.symbol('S'): '1'})
    ignore2 = Ignore.create('DVID != 1')
    rec_1 = rec_old.update_filters([ignore1, ignore2])
    assert len(rec_1.ignore) == 2
    assert str(rec_1.ignore[0]) == 'DV.EQ.1'
    assert str(rec_1.ignore[1]) == 'DVID.NEN.1'
    ignore3 = Ignore.create('APGR > 23')
    rec_2 = rec_1.update_filters([ignore3])
    assert len(rec_2.ignore) == 3
    assert str(rec_2.ignore[2]) == 'APGR.GT.23'
    rec_3 = rec_2.update_filters([ignore1])
    assert len(rec_3.ignore) == 3
    assert 'IGNORE=(DV.EQ.1) IGNORE=(DVID.NEN.1) IGNORE=(APGR.GT.23)' in str(rec_3)
