from collections import OrderedDict

#from pharmpy.input import InputFilterOperator, InputFilter, InputFilters


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
"""

def test_filter(nonmem):
    record = nonmem.records.create_record('DATA pheno.dta NOWIDE')
    print(record.parser)
    assert str(record.root.filename) == 'pheno.dta'
    assert str(record.root.find('option')) == 'NOWIDE'
    assert record.ignore_character is None

    record = nonmem.records.create_record("DATA  'pheno.dta'  IGNORE=(ID.EQ.1,MDV.NEN.0) ")
    print(record.parser)
    assert str(record.root.filename) == "'pheno.dta'"
    assert str(record.root.ignore) == 'IGNORE=(ID.EQ.1,MDV.NEN.0)'
    assert record.ignore_character is None
    filters = record.filters
    assert filters[0].symbol == "ID"
    assert filters[0].value == "1"
    assert filters[0].operator == InputFilterOperator.STRING_EQUAL
    assert filters[1].symbol == "MDV"
    assert filters[1].value == "0"
    assert filters[1].operator == InputFilterOperator.NOT_EQUAL

    record = nonmem.records.create_record('DATA "pheno.dta" NOWIDE IGNORE=(ID==1)')
    print(record.parser)
    assert str(record.root.filename) == '"pheno.dta"'
    assert str(record.root.find('option')) == 'NOWIDE'
    assert str(record.root.ignore) == 'IGNORE=(ID==1)'
    assert record.ignore_character is None

    record = nonmem.records.create_record("DATA      pheno.dta IGNORE=@\n")
    assert str(record.root.filename) == 'pheno.dta'
    assert record.ignore_character == '@'


def test_set_filter(nonmem):
    record = nonmem.records.create_record("DATA  'pheno.dta' IGNORE=@ IGNORE=(ID.EQ.1,MDV.NEN.0) ")
    filters = InputFilters([InputFilter("WGT", InputFilterOperator.EQUAL, "28")])
    record.filters = filters
    read_filters = record.filters
    assert len(read_filters) == 1
    assert read_filters[0].symbol == "WGT"
    assert read_filters[0].value == "28"
    assert read_filters[0].operator == InputFilterOperator.EQUAL

    filters = InputFilters([InputFilter("APGR", InputFilterOperator.LESS_THAN, 2),
                            InputFilter("DOSE", InputFilterOperator.NOT_EQUAL, 20)])
    record.filters = filters
    assert str(record.root) == "  'pheno.dta' IGNORE=@  IGNORE=(APGR.LT.2,DOSE.NEN.20)"
    read_filters = record.filters
    assert len(read_filters) == 2
"""
