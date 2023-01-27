import pytest
from lark.exceptions import UnexpectedToken

from pharmpy.model import ModelSyntaxError


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

    # simple replace
    record = record.set_filename('/new/path/to_file.txt')
    assert record.filename == '/new/path/to_file.txt'
    assert str(record) == '$DATA /new/path/to_file.txt ; comment'

    # force quoting
    record = record.set_filename('MUST=QUOTE')
    assert record.filename == 'MUST=QUOTE'
    assert str(record) == "$DATA 'MUST=QUOTE' ; comment"

    # more complex example
    text = '$DATA ; comment\n ; some comment line\n pheno.dta\n\n'
    record = parser.parse(text).records[0]
    assert record.filename == 'pheno.dta'
    assert str(record) == text

    # more complex replace
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
