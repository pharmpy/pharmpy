from collections import OrderedDict

from pharmpy.input import InputFilterOperator, InputFilter, InputFilters


def test_data_filename_get(nonmem):
    record = nonmem.records.create_record('DATA "pheno.dta"')
    assert record.filename == 'pheno.dta'

    record = nonmem.records.create_record('DATA /home/full/pheno.dta')
    assert record.filename == '/home/full/pheno.dta'

    record = nonmem.records.create_record("DATA 'pheno.dta'")
    assert str(record.root.filename) == "'pheno.dta'"
    assert record.filename == "pheno.dta"

    record = nonmem.records.create_record('DATA "C:\windowspath\with space in.csv"')
    assert record.filename == 'C:\windowspath\with space in.csv'

    record = nonmem.records.create_record("DATA \n pheno.dta \n; comment\n")
    assert record.filename == 'pheno.dta'

    record = nonmem.records.create_record('DATA ; comment\n ; some comment line\n pheno.dta\n\n')
    assert record.filename == 'pheno.dta'


def test_data_filename_set(nonmem):
    record = nonmem.records.create_record('DATA DUMMY ; comment')
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

    # FIXME: Code for below SHOULD work, but GRAMMAR prioritizes comment parse before filename!

    # more complex example
    text = 'DATA ; comment\n ; some comment line\n pheno.dta\n\n'
    record = nonmem.records.create_record(text)
    assert record.filename == 'pheno.dta'
    assert str(record) == ('$%s' % text)

    # more complex replace
    record.filename = "'IGNORE'"
    assert record.filename == "'IGNORE'"
    assert str(record) == ('$%s' % text).replace('pheno.dta', '"\'IGNORE\'"')


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
    print(record.parser)
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
    # assert read_filters[0].symbol == "WGT"
    # assert read_filters[0].value == "2"
    # assert read_filters[0].operator == InputFilterOperator.EQUAL


def test_option_record(nonmem):
    record = nonmem.records.create_record('DATA pheno.dta NOWIDE')
    assert record.option_pairs == OrderedDict([('NOWIDE', None)])
