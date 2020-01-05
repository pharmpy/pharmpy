from collections import OrderedDict


def test_data_filename_get(parser):
    record = parser.parse('$DATA "pheno.dta"').records[0]
    assert record.filename == 'pheno.dta'

    record = parser.parse('$DATA /home/full/pheno.dta').records[0]
    assert record.filename == '/home/full/pheno.dta'

    record = parser.parse("$DATA 'pheno.dta'").records[0]
    assert str(record.root.filename) == "'pheno.dta'"
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
    record.filename = '/new/path/to_file.txt'
    assert record.filename == '/new/path/to_file.txt'
    assert str(record) == '$DATA /new/path/to_file.txt ; comment'

    # force quoting
    record.filename = 'MUST=QUOTE'
    assert record.filename == 'MUST=QUOTE'
    assert str(record) == "$DATA 'MUST=QUOTE' ; comment"

    # more complex example
    text = '$DATA ; comment\n ; some comment line\n pheno.dta\n\n'
    record = parser.parse(text).records[0]
    assert record.filename == 'pheno.dta'
    assert str(record) == text

    # more complex replace
    record.filename = "'IGNORE'"
    assert record.filename == "'IGNORE'"
    assert str(record) == text.replace('pheno.dta', '"\'IGNORE\'"')


def test_option_record(parser):
    record = parser.parse('$DATA pheno.dta NOWIDE').records[0]
    assert record.option_pairs == OrderedDict([('NOWIDE', None)])


def test_ignore_character(parser):
    record = parser.parse('$DATA pheno.dta IGNORE=@').records[0]
    assert record.filename == 'pheno.dta'
    assert record.ignore_character == '@'

    record = parser.parse('$DATA pheno.dta IGNORE="I"').records[0]
    assert record.ignore_character == 'I'


def test_null_value(parser):
    record = parser.parse('$DATA pheno.dta NULL=1').records[0]
    assert record.null_value == 1


def test_ignore_accept(parser):
    record = parser.parse('$DATA pheno.dta IGNORE=(DV.EQ.1)').records[0]
    assert str(record.ignore[0]) == 'DV.EQ.1'
    assert record.accept == []
    record = parser.parse('$DATA pheno.dta ACCEPT=(DV.EQ.1,    MDV.NEN.23)').records[0]
    assert str(record.accept[0]) == 'DV.EQ.1'
    assert str(record.accept[1]) == 'MDV.NEN.23'
    assert record.ignore == []
    record = parser.parse('$DATA pheno.dta IGNORE=(WGT  < 1  ,\n  ID\n.EQ."lk")').records[0]
    assert str(record.ignore[0]) == 'WGT  < 1', 'ID\n.EQ."lk"'


def test_data_infile(parser):
    record = parser.parse('$INFILE pheno.dta').records[0]
    assert record.name == 'DATA'
    assert record.filename == 'pheno.dta'
    assert record.raw_name == '$INFILE'
